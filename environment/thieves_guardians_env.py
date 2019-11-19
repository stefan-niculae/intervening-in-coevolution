""" Custom environment """
import numpy as np

from configs.structure import Config


# Map cell states
THIEF    = 0  # thief must be zero
GUARDIAN = 1  # guardian must be 1
EMPTY    = 2
WALL     = 3
TREASURE = 4

# Actions
UP    = 0
DOWN  = 1
LEFT  = 2
RIGHT = 3
NOOP  = 4  # must be last one


DEAD = -1
ACTION_IDX2SYMBOL = {
    UP:     '⬆️',
    DOWN:   '⬇️',
    LEFT:   '⬅️',
    RIGHT:  '➡️️',
    NOOP:   'n',
    DEAD:   '◼️',
}

action_idx2delta = {
    UP:    np.array([-1,  0]),
    DOWN:  np.array([+1,  0]),
    LEFT:  np.array([ 0, -1]),
    RIGHT: np.array([ 0, +1]),
    NOOP:  np.array([ 0,  0]),
}


# (thief reward, guardian reward)
REWARDS = {
    'killed':      (-.5, +.5),
    'out_of_time': (-.5,   0),
    'time':        (  0,   0),
    'treasure':    ( +1,  -1),
}


def _coords_where(grid: np.array):
    """ A position (x, y) of an arbitrary one in the grid """
    xs, ys = np.where(grid == 1)
    return xs[0], ys[0]


class TGEnv:
    """
    Thieves aim to reach a treasure, guardians aim to catch the thieves

    An avatar is a single controllable character; either a thief or a guardian
    An agent controls all avatars on a team,
    (An actor executes the environment)
    """
    def __init__(self, config: Config):
        self.scenario_name = config.scenario
        if self.scenario_name.startswith('random'):
            from environment.scenarios import random_scenario_configs
            scenario = random_scenario_configs[self.scenario_name[-1]]
        else:
            from environment.scenarios import generate_fixed_scenario
            scenario = generate_fixed_scenario(self.scenario_name)

        self.time_limit = config.time_limit

        self._width = scenario.width
        self._height = scenario.height
        self._fixed_original_map = scenario.fixed_map

        self.id2team = np.array([THIEF] * scenario.n_thieves + [GUARDIAN] * scenario.n_guardians)
        self.num_avatars = scenario.n_thieves + scenario.n_guardians
        self.num_teams = int(scenario.n_thieves != 0) + int(scenario.n_guardians != 0)

        self.state_shape = (5, self._width, self._height)
        self.num_actions = len(action_idx2delta)
        if not config.allow_noop:
            self.num_actions -= 1  # the last one is NOOP

        self._num_thieves = scenario.n_thieves
        self.elapsed_time = None
        self._num_remaining_thieves = None
        self.avatar_alive = None
        self._map = None
        self._id2pos = None
        self._pos2id = None
        self._walls_channel = None
        self._treasure_channel = None
        self._chased_thief = None
        self._thief_target = None
        self.reset()

    def reset(self):
        self.elapsed_time = 0
        self._num_remaining_thieves = self._num_thieves

        self._id2pos = {}
        self._pos2id = {}

        self._reset_map()
        self.avatar_alive = np.ones(self.num_avatars, bool)

        return [self._compute_state(i) for i in range(self.num_avatars)]

    def _reset_map(self):
        """ Place wall pieces randomly, and then the treasure, thieves and guardians in different quadrants """
        if not self.scenario_name.startswith('random'):
            self._map = self._fixed_original_map.copy()
        else:
            from environment.scenarios import generate_random_map
            self._map = generate_random_map(self.scenario_name)

        # Precompute treasure and wall channels since they're static
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

        self._thief_target = _coords_where(self._treasure_channel)
        self._chased_thief = 0

    def _move_or_kill(self, avatar_id, avatar_team, old_pos, new_pos=None):
        self._map[old_pos] = EMPTY
        del self._pos2id[old_pos]

        # When moved out of the map (killed)
        if new_pos is None:
            assert avatar_team == THIEF
            self._num_remaining_thieves -= 1

            # Find a new target for the guardians: the first thief that is not alive
            if avatar_id == self._chased_thief and self._num_remaining_thieves > 0:
                for thief_id in range(self._num_thieves):
                    if self.avatar_alive[thief_id]:
                        self._chased_thief = thief_id
                        break

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

        done shape: (num_avatars,)
            alive or time out

        info: dict
            infos['individual_done']: (num_avatars,)

        """
        info = {}
        done   = np.zeros(self.num_avatars, bool)
        reward = np.zeros(self.num_avatars, float)

        for avatar_id in range(self.num_avatars):
            if not self.avatar_alive[avatar_id]:
                continue
            team = self.id2team[avatar_id]
            reward[avatar_id] += REWARDS['time'][team]

        for avatar_id in range(self.num_avatars):
            if not self.avatar_alive[avatar_id]:
                continue

            action_idx = actions[avatar_id]
            delta = action_idx2delta[action_idx]
            old_pos = self._id2pos[avatar_id]
            new_pos = tuple(old_pos + delta)  # NOTE: make sure self.map[pos] the arg is a tuple, not a (2,) array

            # No team can move out of bounds, just ignore the action
            if not (0 <= new_pos[0] < self._width and 0 <= new_pos[1] < self._height):
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
                done[:] = True
                info['end_reason'] = f'Treasure reached'

                thief_reward, guardian_reward = REWARDS['treasure']
                #thief_reward *= 5 / self.elapsed_time  # TODO generalize
                reward[avatar_id] += thief_reward

                # Punish all guardians
                for avatar_id in range(self.num_avatars):
                    if self.id2team[avatar_id] == GUARDIAN:
                        reward[avatar_id] += guardian_reward
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
                done[avatar_id] = True

                reward[avatar_id]   += thief_reward
                reward[guardian_id] += guardian_reward
                continue

            # A guardian managed to catch a thief, kill the thief and apply rewards
            if avatar_team == GUARDIAN and new_pos_type == THIEF:
                thief_id = self._pos2id[new_pos]

                self._move_or_kill(thief_id, THIEF, new_pos)
                done[thief_id] = True

                self._move_or_kill(avatar_id, GUARDIAN, old_pos, new_pos)

                reward[thief_id]  += thief_reward
                reward[avatar_id] += guardian_reward
                continue

        # No more thieves alive, the game is over (thieves and guardians have been rewarded at the moments of killing)
        if self._num_remaining_thieves == 0:
            info['end_reason'] = 'All thieves dead'
            done[:] = True

        self.elapsed_time += 1
        if self.elapsed_time == self.time_limit:
            thief_reward, guardian_reward = REWARDS['out_of_time']
            info['end_reason'] = 'Out of time'
            done[:] = True
            # Apply reward to all avatars alive
            for avatar_id in range(self.num_avatars):
                if not self.avatar_alive[avatar_id]:
                    continue
                team = self.id2team[avatar_id]
                if team == GUARDIAN:
                    reward[avatar_id] += guardian_reward
                if team == THIEF:
                    reward[avatar_id] += thief_reward

        state = [self._compute_state(i) for i in range(self.num_avatars)]

        # Update for next step: alive if alive before and not done
        self.avatar_alive &= ~done

        return state, reward, done, info

    def _compute_state(self, for_id):
        """
        Five channels, each of size width by height, each cell having values 0 or 1
            1. own position
            2. teammate(s)
            3. opposing team
            4. walls
            5. treasure

        None if the avatar is dead
        """
        if not self.avatar_alive[for_id]:
            return None

        own_pos  = self._id2pos[for_id]
        own_team = self.id2team[for_id]
        opposing_team = GUARDIAN if own_team == THIEF else THIEF

        own = np.zeros_like(self._map, float)
        own[own_pos] = 1

        teammates = np.zeros_like(self._map, float)
        opponents = np.zeros_like(self._map, float)
        for pos, id in self._pos2id.items():
            if self.id2team[id] == own_team:
                teammates[pos] = 1
            if self.id2team[id] == opposing_team:
                opponents[pos] = 1

        # Channels first
        return np.stack([own, teammates, opponents, self._walls_channel, self._treasure_channel])

    def scripted_action(self, avatar_id):
        r, c = self._id2pos[avatar_id]
        team = self.id2team[avatar_id]
        if team == THIEF:
            tr, tc = self._thief_target
        else:
            tr, tc = self._id2pos[self._chased_thief]

        if tr > r:
            return DOWN
        if tr < r:
            return UP
        if tc > c:
            return RIGHT
        if tc < c:
            return LEFT

        print('id', avatar_id, 'self', r, c, 'target', tr, tc)
        return NOOP

    def __str__(self):
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
        return s


if __name__ == '__main__':
    # e = TGEnv('4x4-thief-treasure')
    env = TGEnv('test-kill')
    print(env.avatar_alive)
    print(env)

    env.step([UP, UP, LEFT])
    print(env.avatar_alive)
    print(env)

    env.step([UP, None, UP])
    print(env.avatar_alive)
    print(env)
