# -*- coding: utf-8 -*-
""" Custom environment """

from gym.spaces import Box, Discrete
from gym import Env
import numpy as np

# Map cell states
EMPTY    = 0
WALL     = 1
THIEF    = 2
GUARDIAN = 3
TREASURE = 4

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
# NOOP  = 4

action_idx2delta = {
    UP:    np.array([-1,  0]),
    DOWN:  np.array([+1,  0]),
    LEFT:  np.array([ 0, -1]),
    RIGHT: np.array([ 0, +1]),
    # NOOP:  np.array([ 0,  0]),
}


# (thief reward, guardian reward)
REWARDS = {
    'killed':      (-1, +1),
    'out_of_time': (0, +5),
    'treasure':    (+9,  0),  # TODO: should guardians get a negative reward here?
}

DEBUG = False
dump_path = 'outputs/execution-#%d'


class ThievesGuardiansEnv(Env):
    """ Thieves aim to reach a treasure, guardians aim to catch the thieves """

    """
    A single controllable character; either a thief or a guardian
    Not to be confused with an Agent (controls all avatars on a team),
    or an Actor (executes for all Agents)
    """

    metadata = {'render.modes': ['rgb_array']}

    def __init__(self, scenario: str, env_id: int):
        self.env_id = env_id

        if scenario.startswith('random'):
            from environment.scenarios import random_scenario_configs
            config = random_scenario_configs[scenario[-1]]
        else:
            from environment.scenarios import fixed_scenario_config
            config = fixed_scenario_config(scenario)

        self._width = config.width
        self._height = config.height
        self._fixed_original_map = config.fixed_map

        if self._fixed_original_map is None:
            self._quadrant_ranges = self._compute_quadrants()
            self._wall_density = config.wall_density

        self.time_limit = config.time_limit
        self.elapsed_time = None

        self._n_thieves = config.n_thieves
        self._n_guardians = config.n_guardians

        self.num_avatars = self._n_thieves + self._n_guardians
        self.action_space = Discrete(len(action_idx2delta))
        self.observation_space = Box(low=0, high=1, shape=(5, self._width, self._height), dtype=np.uint8)
        self._dummy_dead_state = np.full((5, self._width, self._height), np.nan)

        self._id2team = np.array([THIEF] * self._n_thieves + [GUARDIAN] * self._n_guardians)
        self._controller = (self._id2team == GUARDIAN).astype(np.uint8)  # 0 for thieves, 1 for guardians

        self._n_remaining_thieves = None
        self._avatar_alive = None
        self._map = None
        self._id2pos = None
        self._pos2id = None
        self._walls_channel = None
        self._treasure_channel = None
        self.reset()

        if DEBUG:
            with open(dump_path % env_id, 'w') as f:
                f.write('')

    def reset(self):
        self.elapsed_time = 0
        self._n_remaining_thieves = self._n_thieves

        self._map = np.full((self._width, self._height), EMPTY)
        self._id2pos = {}
        self._pos2id = {}

        self._reset_map()
        self._avatar_alive = np.ones(self.num_avatars, bool)

        return self._compute_all_states()

    def random_cell(self, x_range, y_range) -> (int, int):
        x = np.random.randint(*x_range)
        y = np.random.randint(*y_range)
        return x, y

    def _random_empty_cell(self, quadrant_idx) -> (int, int):
        ranges = self._quadrant_ranges[quadrant_idx]

        failsafe = 0
        x, y = self.random_cell(*ranges)
        while self._map[x, y] != EMPTY:
            x, y = self.random_cell(*ranges)
            failsafe += 1
            if failsafe == 100:
                raise Exception(f'Could not find an open space in quadrant {quadrant_idx}')

        return x, y

    def _compute_quadrants(self) -> [((int, int), (int, int))]:
        """
        Split the map into four equal quadrants

        Returns:
            for each of the four quadrants, an x range and a y range
        """
        W = self._width
        H = self._height
        H2 = H//2
        W2 = W//2
        top    = ( 0, H2)
        bottom = (H2, H )
        left   = ( 0, W2)
        right  = (W2, W )
        return [
            (left, top),
            (left, bottom),
            (right, top),
            (right, bottom),
        ]

    def _reset_map(self):
        """ Place wall pieces randomly, and then the treasure, thieves and guardians in different quadrants """
        if self._fixed_original_map is not None:
            self._map = self._fixed_original_map.copy()

        else:
            # Place walls
            wall_mask = np.random.rand(self._width, self._height) < self._wall_density
            self._map[wall_mask] = WALL

            # Pick areas for the teams and treasure
            thieves_quad, guardians_quad, treasure_quad = np.random.choice(4, size=3, replace=False)

            # Place treasure
            treasure_pos = self._random_empty_cell(treasure_quad)
            self._map[treasure_pos] = TREASURE  # TODO walls don't block off objects

            # Place the thieves and guardians
            for avatar_id in range(self._n_thieves):
                x, y = self._random_empty_cell(thieves_quad)
                self._map[x, y] = THIEF

            for avatar_id in range(self._n_thieves, self._n_thieves + self._n_guardians):
                x, y = self._random_empty_cell(guardians_quad)
                self._map[x, y] = GUARDIAN

        # Precompute treasure and wall channels since it's static
        self._treasure_channel = (self._map == TREASURE).astype(int)
        self._walls_channel    = (self._map == WALL)    .astype(int)

        # Set avatar position bidirectional caches (first thieves then guardians)
        xs_t, ys_t = np.where(self._map == THIEF)
        xs_g, ys_g = np.where(self._map == GUARDIAN)
        xs = np.concatenate([xs_t, xs_g])
        ys = np.concatenate([ys_t, ys_g])
        for avatar_id, (x, y) in enumerate(zip(xs, ys)):
            self._id2pos[avatar_id] = x, y
            self._pos2id[(x, y)] = avatar_id

    def in_bounds(self, x, y):
        return 0 <= x < self._width and 0 <= y < self._height

    def _move_or_kill(self, avatar_id, avatar_team, old_pos, new_pos=None):
        self._map[old_pos] = EMPTY
        del self._pos2id[old_pos]

        # When moved out of the map (killed)
        if new_pos is None:
            self._avatar_alive[avatar_id] = False
            #self._id2team[avatar_id] = None

        # When moved to a new valid position
        else:
            self._map[new_pos] = avatar_team
            self._id2pos[avatar_id] = new_pos
            self._pos2id[new_pos] = avatar_id

    def step(self, actions: [int]):
        """
        actions shape: (num_avatars,)
            one for each avatar, can ignore the actions for dead avatars

        reward shape: (num_avatars,)
            -inf if the avatar is already dead

        done shape: bool
            when all are done

        info: dict
            infos['individual_done']: (num_avatars,)

        """
        info = {'controller': self._controller}  # for two thieves and one guardian: [0, 0, 1]
        individual_done = np.zeros(self.num_avatars, bool)
        reward          = np.zeros(self.num_avatars, float)

        avatars_alive = self._avatar_alive.nonzero()[0]

        for avatar_id in avatars_alive:
            action_idx = actions[avatar_id]
            delta = action_idx2delta[action_idx]
            old_pos = self._id2pos[avatar_id]
            new_pos = tuple(old_pos + delta)  # NOTE: make sure self.map[pos] the arg is a tuple, not a (2,) array

            # No team can move out of bounds, just ignore the action
            if not self.in_bounds(*new_pos):
                continue
            # TODO? (idea): allow screen wrap-around? for only one team?
            # new_pos[0] %= self.width
            # new_pos[1] %= self.height

            avatar_team  = self._map[old_pos]  # the character that is currently moving
            new_pos_type = self._map[new_pos]

            # Trying to step onto a teammate, just ignore the action
            if new_pos_type == avatar_team:
                continue

            # Trying to run into a wall, just ignore the action
            if new_pos_type == WALL:
                continue

            # A guardian is trying to step on the treasure, ignore the action
            if avatar_team == GUARDIAN and new_pos_type == TREASURE:
                continue
            # TODO? (idea): negative reward if a guardian touches the treasure?

            # A thief managed to reach the treasure, the game is over, punish all guardians
            if avatar_team == THIEF and new_pos_type == TREASURE:
                individual_done[:] = True
                info['end_reason'] = f'A thief (id={avatar_id}) reached the treasure'

                thief_reward, guardian_reward = REWARDS['treasure']
                reward[avatar_id] += thief_reward

                # Punish all guardians
                for pos, id in self._pos2id.items():
                    if self._map[pos] == GUARDIAN:
                        reward[id] += guardian_reward

                continue

            # Any team can move freely to an empty cell
            if new_pos_type == EMPTY:
                self._move_or_kill(avatar_id, avatar_team, old_pos, new_pos)
                continue

            thief_reward, guardian_reward = REWARDS['killed']
            # A thief is (stupidly) bumping into a guardian, kill the thief and apply rewards
            if avatar_team == THIEF and new_pos_type == GUARDIAN:
                guardian_id = self._pos2id[new_pos]

                self._move_or_kill(avatar_id, THIEF, old_pos)
                individual_done[avatar_id] = True

                reward[avatar_id]   += thief_reward
                reward[guardian_id] += guardian_reward
                continue

            # A guardian managed to catch a thief, kill the thief and apply rewards
            if avatar_team == GUARDIAN and new_pos_type == THIEF:
                thief_id = self._pos2id[new_pos]

                self._move_or_kill(thief_id, THIEF, new_pos)
                individual_done[thief_id] = True

                self._move_or_kill(avatar_id, GUARDIAN, old_pos, new_pos)

                reward[thief_id]  += thief_reward
                reward[avatar_id] += guardian_reward
                continue

        # No more thieves alive, the game is over (thieves and guardians have been rewarded at the moments of killing)
        if sum(self._avatar_alive) == self._n_guardians:
            info['end_reason'] = 'All thieves dead'
            individual_done[:] = True

        self.elapsed_time += 1
        if self.elapsed_time == self.time_limit:
            thief_reward, guardian_reward = REWARDS['out_of_time']
            info['end_reason'] = 'Ran out of time'
            individual_done[:] = True
            # Apply reward to all avatars alive
            for id in avatars_alive:
                team = self._id2team[id]
                if team == GUARDIAN:
                    reward[id] += guardian_reward
                if team == THIEF:
                    reward[id] += thief_reward

        state = self._compute_all_states()

        info['individual_done'] = individual_done
        all_done = all(individual_done)

        if DEBUG:
            self.render('file')
        
        return state, reward, all_done, info

    def _compute_all_states(self):
        return np.stack([
            self._compute_state(id) if alive else self._dummy_dead_state
            for id, alive in enumerate(self._avatar_alive)
        ])

    def _compute_state(self, for_id):
        """
        Five channels, each of size width by height, each cell having values 0 or 1
            1. own position
            2. teammate(s)
            3. opposing team
            4. walls
            5. treasure
        """
        own_pos  = self._id2pos [for_id]
        own_team = self._id2team[for_id]
        opposing_team = GUARDIAN if own_team == THIEF else THIEF

        own = np.zeros_like(self._map, float)
        own[own_pos] = 1

        teammates = np.zeros_like(self._map, float)
        opponents = np.zeros_like(self._map, float)
        for pos, id in self._pos2id.items():
            if self._id2team[id] == own_team:
                teammates[pos] = 1
            if self._id2team[id] == opposing_team:
                opponents[pos] = 1

        # Channels first
        return np.stack([own, teammates, opponents, self._walls_channel, self._treasure_channel])

    def render(self, mode='print'):
        CELL_TYPE2LETTER = {
            EMPTY:    '.',
            WALL:     'W',
            THIEF:    'T',
            GUARDIAN: 'G',
            TREASURE: 'S',
        }

        s = ''
        for row in self._map:
            for cell in row:
                s += CELL_TYPE2LETTER[cell] + ' '
            s += '\n'

        if mode == 'print':
            print(s)

        if mode == 'file':
            with open(f'env-{self.env_id}', 'a') as f:
                f.write(s + '\n\n')


if __name__ == '__main__':
    e = ThievesGuardiansEnv('4x4-thief-treasure', env_id=0)
    e.render()
    print()
