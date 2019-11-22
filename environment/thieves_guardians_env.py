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
UP         = 0
DOWN       = 1
LEFT       = 2
RIGHT      = 3
UP_LEFT    = 4
UP_RIGHT   = 5
DOWN_LEFT  = 6
DOWN_RIGHT = 7
NOOP       = 8  # must be last one


DEAD = -1
ACTION_IDX2SYMBOL = {
    UP:         '⬆️',
    DOWN:       '⬇️',
    LEFT:       '⬅️',
    RIGHT:      '➡️️',
    UP_LEFT:    '⬉',
    UP_RIGHT:   '⬈',
    DOWN_LEFT:  '⬋',
    DOWN_RIGHT: '⬊',
    NOOP:       'N',
    DEAD:       'D',
}

action_idx2delta = {
    UP:    np.array([-1,  0]),
    DOWN:  np.array([+1,  0]),
    LEFT:  np.array([ 0, -1]),
    RIGHT: np.array([ 0, +1]),
    NOOP:  np.array([ 0,  0]),
}
action_idx2delta[UP_RIGHT]   = action_idx2delta[UP]   + action_idx2delta[RIGHT]
action_idx2delta[UP_LEFT]    = action_idx2delta[UP]   + action_idx2delta[LEFT]
action_idx2delta[DOWN_RIGHT] = action_idx2delta[DOWN] + action_idx2delta[RIGHT]
action_idx2delta[DOWN_LEFT]  = action_idx2delta[DOWN] + action_idx2delta[LEFT]


# (thief reward, guardian reward)
REWARDS = {
    'killed':      (  0,  +1),
    'out_of_time': (  0,   0),
    'time':        (  0,   0),
    'treasure':    ( +1,   0),
}

DEAD_COORDS = (-1, -1)


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
        self.num_walls = scenario.num_walls
        self.num_treasures = scenario.num_treasures
        self.num_teams = int(scenario.n_thieves != 0) + int(scenario.n_guardians != 0)

        self._state_representation = config.state_representation
        if self._state_representation == 'grid':
            self.state_shape = (5, self._width, self._height)
        elif self._state_representation == 'coordinates':
            num_objects = self.num_avatars + self.num_treasures + self.num_walls
            self.state_shape = (num_objects, 2)  # x and y for each
        self.allow_wraparound = config.allow_wraparound

        self.allow_noops = config.allow_noop
        self.allow_diagonals = config.allow_diagonals
        self.num_actions = [4] * self.num_teams  # by default they can all move in four directions
        for team, (noop, diagonals) in enumerate(zip(self.allow_noops, self.allow_diagonals)):
            if noop:
                self.num_actions[team] += 1
            if diagonals:
                self.num_actions[team] += 4

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

        return self._compute_state()

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

        xs, ys = np.where(self._walls_channel)
        self._wall_positions = list(zip(xs, ys))
        xs, ys = np.where(self._treasure_channel)
        self._treasures_positions = list(zip(xs, ys))

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

    def _interpret_action(self, action_idx: int, team: int):
        """
        0,1,2,3 are always up,down,left,right
        if noop is enabled, but diagonals disabled: 4 is noop
        if noop is disabled, but diagonals enabled: 4,5,6,7 are upleft,upright,downleft,downright
        if both noop and diagonals are enabled: 4,5,6,7 are upleft,upright,downleft,downright and 8 is noop
        """
        if action_idx < 4:
            return action_idx

        noops = self.allow_noops[team]
        diags = self.allow_diagonals[team]
        assert noops or diags

        if noops and not diags:
            assert action_idx == 4
            return NOOP

        if not noops and diags:
            assert action_idx < 8
            return action_idx

        if noops and diags:
            assert action_idx < 9
            return action_idx

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

            # Apply every-timestep reward
            team = self.id2team[avatar_id]
            reward[avatar_id] += REWARDS['time'][team]

            # Interpret actions
            action_idx = self._interpret_action(actions[avatar_id], team)
            delta = action_idx2delta[action_idx]
            old_pos = self._id2pos[avatar_id]
            new_pos = tuple(old_pos + delta)  # NOTE: make sure self.map[pos] the arg is a tuple, not a (2,) array

            # Trying to go over the edge
            if not (0 <= new_pos[0] < self._width and 0 <= new_pos[1] < self._height):
                # If allowed to wrap around, teleport to the other side of the screen
                if self.allow_wraparound[team]:
                    new_pos[0] %= self._width
                    new_pos[1] %= self._height

                # Otherwise, just ignore the action
                else:
                    continue

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

            # A thief managed to reach the treasure, the game is over, punish all guardians
            if avatar_team == THIEF and new_pos_type == TREASURE:
                done[:] = True
                info['end_reason'] = f'Treasure reached'

                thief_reward, guardian_reward = REWARDS['treasure']
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

        # Update for next step: alive if alive before and not done
        self.avatar_alive &= ~done
        print(self._compute_state())
        return self._compute_state(), reward, done, info

    def _compute_state(self):
        if self._state_representation == 'grid':
            f = self._compute_grid_state
        else:
            f = self._compute_coordinates_state
        return [
            f(i) if alive else None
            for i, alive in enumerate(self.avatar_alive)
        ]

    def _compute_grid_state(self, for_id):
        """
        Five channels, each of size width by height, each cell having values 0 or 1
            1. own position
            2. teammate(s)
            3. opponent(s)
            4. walls
            5. treasure(s)
        """
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

    def _compute_coordinates_state(self, for_id):
        """
        List of coordinates (x, y):
            1. own position
            2. teammate(s)
            3. opponent(s)
            4. walls
            5. treasure(s)

        # TODO if we want to use the same network on multiple scenarios, or want to use Transformers, a third component would need to be added (x, y, object_kind_identifier)\
        """
        own_team = self.id2team[for_id]
        teammates = []
        opponents = []
        for id, alive in enumerate(self.avatar_alive):
            if id == for_id:
                continue
            if self.id2team[id] == own_team:
                team_list = teammates
            else:
                team_list = opponents
            if alive:
                team_list.append(self._id2pos[id])
            else:
                team_list.append(DEAD_COORDS)

        coords = np.array([
            self._id2pos[for_id],  # own positions

            # Sorting folds the state space by always showing avatars closer to a corner first
            # with no side effects since all avatars in a team are equivalent to each-other
            *sorted(teammates, reverse=True),  # but place deads ones last
            *sorted(opponents, reverse=True),

            *self._wall_positions,
            *self._treasures_positions,  # TODO adapt to multiple treasures with DEAD as well
        ], float)

        # Scale into [0, 1] range
        coords /= [self._width - 1, self._height - 1]
        # to scale into [-1, +1] range: ((coords / [self._width - 1, self._height - 1]) * 2) - 1
        coords[coords < 0] = -1  # reset scaling on the dead state (-1)

        return coords

    def scripted_action(self, avatar_id):
        r, c = self._id2pos[avatar_id]
        team = self.id2team[avatar_id]

        # Thieves' target is a treasure
        if team == THIEF:
            tr, tc = self._thief_target

        # Guardians' target is a thief
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

        print(f'Warning: could not pick action for avatar #{avatar_id}, of team {team}'
              f'(current pos = ({r}, {c}), target pos = ({tr}, {tc}).'
              f'Defaulting to UP ({UP}).')
        return UP

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
