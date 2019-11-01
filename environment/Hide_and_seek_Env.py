""" Custom environment """

from gym.spaces import Box
from gym import Env
import numpy as np

# Map cell states
EMPTY    = 0
WALL     = 1
THIEF    = 2
GUARDIAN = 3
TREASURE = 4

# Actions
NOOP  = 0
UP    = 1
DOWN  = 2
LEFT  = 3
RIGHT = 4

action_idx2delta = {
    NOOP:  np.array([ 0,  0]),
    UP:    np.array([-1,  0]),
    DOWN:  np.array([+1,  0]),
    LEFT:  np.array([ 0, -1]),
    RIGHT: np.array([ 0, +1]),
}


# (thief reward, guardian reward)
REWARDS = {
    'killed':      (-1, +1),
    'out_of_time': (-5, +5),
    'treasure':    (+9,  0),
}


class Hide_and_seek_Env(Env):
    """ Thieves aim to reach a trueasure, guardians aim to catch the thieves """

    """
    A single controllable character; either a thief or a guardian
    Not to be confused with an Agent (controls all avatars on a team),
    or an Actor (executes for all Agents)
    """

    def __init__(self, width=8, height=8, time_limit=30, n_thieves=2, n_guardians=2, wall_density=0.):
        """

        Args:
            width: number of horizontal cells
            height: number of vertical cells
            time_limit: one time step is when all avatars (thieves and guardians) moved
            n_thieves:
            n_guardians:
            wall_density:
        """
        self.width = width
        self.height = height
        self._quadrant_ranges = self.compute_quadrants()

        self.time_limit = time_limit  # TODO look at TimeLimitMask EnvWrapper to see expected names
        self.elapsed_time = None

        self.n_thieves = n_thieves
        self.n_guardians = n_guardians
        self.wall_density = wall_density
        self.n_remaining_thieves = self.n_thieves

        self.action_space      = Box(low=0, high=len(action_idx2delta), shape=(self.n_thieves + self.n_guardians,), dtype=int)
        self.observation_space = Box(low=EMPTY, high=TREASURE, shape=(3, self.height, self.width), dtype=int)

        self.map = None
        self.thieves_alive = None
        self.id2pos = None
        self.pos2id = None
        self.walls_channel = None
        self.treasure_channel = None
        self.reset()

    def reset(self):
        self.elapsed_time = 0

        self.map = np.full((self.width, self.height), EMPTY)
        self.id2pos = {}
        self.pos2id = {}

        self.generate_positions()
        self.thieves_alive = self.n_thieves

    def random_cell(self, x_range, y_range) -> (int, int):
        x = np.random.randint(*x_range)
        y = np.random.randint(*y_range)
        return x, y

    def random_empty_cell(self, quadrant_idx) -> (int, int):
        ranges = self._quadrant_ranges[quadrant_idx]

        failsafe = 0
        x, y = self.random_cell(*ranges)
        while self.map[x, y] != EMPTY:
            x, y = self.random_cell(*ranges)
            failsafe += 1
            if failsafe == 100:
                raise Exception(f'Could not find an open space in quadrant {quadrant_idx}')

        return x, y

    def compute_quadrants(self) -> [((int, int), (int, int))]:
        """
        Split the map into four equal quadrants

        Returns:
            for each of the four quadrants, an x range and a y range
        """
        W = self.width
        H = self.height
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

    def generate_positions(self):
        """ Place wall pieces randomly, and then the treasure, thieves and guardians in different quadrants """
        # TODO handcrafted wall positions (otherwise it'll sometimes be unsolvable, and main task is not generalizing to unseen maps necessarily, instead, learning to out-maneuver the other team around obstacles that allow that
        wall_mask = np.random.rand(self.height, self.width) < self.wall_density
        self.map[wall_mask] = WALL
        self.walls_channel = wall_mask.astype(float)

        thieves_quad, guardians_quad, treasure_quad = np.random.choice(4, size=3, replace=False)

        treasure_pos = self.random_empty_cell(treasure_quad)
        self.map[treasure_pos] = TREASURE
        self.treasure_channel = np.zeros_like(self.map, float)
        self.treasure_channel[treasure_pos] = 1

        for avatar_id in range(self.n_thieves):
            x, y = self.random_empty_cell(thieves_quad)
            self.id2pos[avatar_id] = x, y
            self.pos2id[(x, y)] = avatar_id
            self.map[x, y] = THIEF

        for avatar_id in range(self.n_thieves, self.n_thieves + self.n_guardians):
            x, y = self.random_empty_cell(guardians_quad)
            self.id2pos[avatar_id] = x, y
            self.pos2id[(x, y)] = avatar_id
            self.map[x, y] = GUARDIAN

    def in_bounds(self, x, y):
        return 0 <= x < self.width and 0 <= y < self.height

    def step(self, actions: dict):
        """
        Advance every avatar on the board
            eg: for 1 thief and 2 guardians, there are 3 actions

        actions: {avatar_id : action_idx}, one for each avatar_id in self.avatar_positions
        """
        reward = {id: 0     for id in self.id2pos}  # one reward for each remaining avatar
        done    = {id: False for id in self.id2pos}
        info = []

        for avatar_id, old_pos in self.id2pos.items():
            # Thieves move first, so they'll just die if they encounter a guardian
            # meaning thieves cannot die before they make their move
            action_idx = actions[avatar_id]
            delta = action_idx2delta[action_idx]
            new_pos = tuple(old_pos + delta)  # NOTE: make sure self.map[pos] the arg is a tuple, not a (2,) array

            # No team can move out of bounds, just ignore the action
            if not self.in_bounds(*new_pos):
                continue
            # TODO? (idea): allow screen wrap-around? for only one team?
            # new_pos[0] %= self.width
            # new_pos[1] %= self.height

            avatar_team  = self.map[old_pos]  # the character that is currently moving
            new_pos_type = self.map[new_pos]

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
                done = {id: True for id in self.id2pos}
                info.append(f'A thief (id={avatar_id}) reached the treasure')

                thief_reward, guardian_reward = REWARDS['treasure']
                reward[avatar_id] += thief_reward

                # Punish all guardians
                for pos, id in self.pos2id.items():
                    if self.map[pos] == GUARDIAN:
                        reward[id] += guardian_reward

                continue

            # Any team can move freely to an empty cell
            if new_pos_type == EMPTY:
                self.map[old_pos] = EMPTY
                self.map[new_pos] = avatar_team

                self.id2pos[avatar_id] = new_pos
                self.pos2id[new_pos] = avatar_id

                continue

            thief_reward, guardian_reward = REWARDS['killed']
            # A thief is (stupidly) bumping into a guardian, kill the thief and apply rewards
            if avatar_team == THIEF and new_pos_type == GUARDIAN:
                guardian_id = self.pos2id[new_pos]

                del self.id2pos[avatar_id]
                del self.pos2id[old_pos]
                done[avatar_id] = True
                self.thieves_alive -= 1
                self.map[old_pos] = EMPTY

                reward[avatar_id]   += thief_reward
                reward[guardian_id] += guardian_reward
                continue

            # A guardian managed to catch a thief, kill the thief and apply rewards
            if avatar_team == GUARDIAN and new_pos_type == THIEF:
                thief_id = self.pos2id[new_pos_type]

                del self.id2pos[thief_id]
                del self.pos2id[new_pos]
                done[thief_id] = True
                self.thieves_alive -= 1
                self.map[old_pos] = EMPTY

                reward[thief_id]  += thief_reward
                reward[avatar_id] += guardian_reward
                continue

        # No more thieves alive, the game is over (thieves and guardians have been rewarded at the moments of killing)
        if self.thieves_alive == 0:
            info.append('All thieves dead')
            done = {id: True for id in self.id2pos}

        self.elapsed_time += 1
        if self.elapsed_time == self.time_limit:
            thief_reward, guardian_reward = REWARDS['out_of_time']
            info.append('Ran out of time')

            # Apply reward to all avatars alive
            for pos, id in self.pos2id.items():
                if self.map[pos] == GUARDIAN:
                    reward[id] += guardian_reward
                if self.map[pos] == THIEF:
                    reward[id] += thief_reward

        state = {id: self.compute_state(id) for id in self.id2pos}
        return state, reward, done, info

    def compute_state(self, for_id):
        """
        Five channels, each of size width by height, each cell having values 0 or 1
            1. own position
            2. teammate(s)
            3. opposing team
            4. walls
            5. treasure
        """
        # TODO flat version of the state, but somehow handle the fact that thieves can die; also must have a fixed number of obstacles; and the coordinates should be normalized to (-1, +1) on both x and y axes
        own_pos = self.id2pos[for_id]
        own_team = self.map[own_pos]
        opposing_team = GUARDIAN if own_team == THIEF else THIEF

        own = np.zeros_like(self.map, float)
        own[own_pos] = 1

        teammates = np.zeros_like(self.map, float)
        opponents = np.zeros_like(self.map, float)
        for pos, id in self.pos2id.items():
            if self.map[pos] == own_team:
                teammates[pos] = 1
            if self.map[pos] == opposing_team:
                opponents[pos] = 1

        # Channels first
        return np.stack([own, teammates, opponents, self.walls_channel, self.treasure_channel])

    def render(self, mode='matrix'):
        CELL_TYPE2LETTER = {
            EMPTY:    '·',
            WALL:     '□',
            THIEF:    'T',
            GUARDIAN: 'G',
            TREASURE: '☆',
        }

        for row in self.map:
            for cell in row:
                print(CELL_TYPE2LETTER[cell], end=' ')
            print()

        # TODO matplotlib grid plot (EMPTY: white, WALL: grey, GOAL: yellow, THIEF: red, GUARDIAN: blue) with the id of the thief/guardian written in the middle of the respective cells; also compatible with the MonitorEnvWrapper that creates short videos (like in the homework)


if __name__ == '__main__':
    e = Hide_and_seek_Env()
    e.render()
    print()

    e.step({
        0: UP,
        1: UP,
        2: UP,
        3: UP,
    })
    e.render()
