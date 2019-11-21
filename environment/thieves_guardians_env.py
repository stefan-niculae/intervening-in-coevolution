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
    'killed':      (  -.1,  +1),
    'out_of_time': (  0,   0),
    'time':        (  0,   0),
    'treasure':    ( +1,   0),
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

        # Scenario map
        self._width = scenario.width
        self._height = scenario.height
        self._fixed_original_map = scenario.fixed_map

        self.id2team = np.array([THIEF] * scenario.num_thieves + [GUARDIAN] * scenario.num_guardians)
        self.num_avatars = scenario.num_thieves + scenario.num_guardians
        self.num_teams = int(scenario.num_thieves != 0) + int(scenario.num_guardians != 0)

        self.state_shape = (5, self._width, self._height)

        self._allow_wraparound = config.allow_wraparound
        self._allow_noops = config.allow_noop
        self._allow_diagonals = config.allow_diagonals
        self.num_actions = [4] * self.num_teams  # by default they can all move in four directions
        for team, (noop, diagonals) in enumerate(zip(self._allow_noops, self._allow_diagonals)):
            if noop:
                self.num_actions[team] += 1
            if diagonals:
                self.num_actions[team] += 4

        self._teamwide_rewards = config.teamwide_rewards

        # Episode end variables
        self._num_thieves = scenario.num_thieves
        self._num_remaining_thieves = None

        self._num_treasures = scenario.num_treasures
        if config.treasure_collection_limit == -1:
            self._treasure_collection_limit = self._num_treasures
        else:
            self._treasure_collection_limit = config.treasure_collection_limit
        self._num_treasures_to_collect = None

        self._time_left = None
        self._time_limit = config.time_limit

        # Avatar statuses variables
        self.avatar_alive = None
        self._map = None
        self._id2pos = None
        self._pos2id = None

        # Precomputed channels
        self._walls_channel = None

        # Action scripting
        self._chased_thief_id = None
        self._chased_treasure_pos = None

        # Assign values to volatile variables
        self.reset()

    def reset(self):
        # Will be decremented when appropriate
        self._time_left = self._time_limit
        self._num_remaining_thieves = self._num_thieves
        self._num_treasures_to_collect = self._treasure_collection_limit

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
        self._walls_channel    = (self._map == WALL)    .astype(int)

        # Set avatar position bidirectional caches (first thieves then guardians)
        xs_t, ys_t = np.where(self._map == THIEF)
        xs_g, ys_g = np.where(self._map == GUARDIAN)
        xs = np.concatenate([xs_t, xs_g])
        ys = np.concatenate([ys_t, ys_g])
        for avatar_id, (x, y) in enumerate(zip(xs, ys)):
            self._id2pos[avatar_id] = x, y
            self._pos2id[(x, y)] = avatar_id

        self._chased_treasure_pos = _coords_where(self._map == TREASURE)
        self._chased_thief_id = 0

    def _move_or_kill(self, avatar_id, avatar_team, old_pos, new_pos=None):
        self._map[old_pos] = EMPTY
        del self._pos2id[old_pos]

        # When moved out of the map (killed)
        if new_pos is None:
            assert avatar_team == THIEF
            self._num_remaining_thieves -= 1
            # del self._id2pos[avatar_id]  # TODO keep all these variables in sync of movement and deaths (with numpy masks?)

            # Find a new target for the guardians: the first thief that is not alive
            if avatar_id == self._chased_thief_id:
                if self._num_remaining_thieves > 0:
                    for thief_id in range(self._num_thieves):
                        if self.avatar_alive[thief_id]:
                            self._chased_thief_id = thief_id
                            break
                else:
                    self._chased_thief_id = None

        # When moved to a new valid position
        else:
            # Decrement amount of treasures left to be collected
            collected_treasure = (self._map[new_pos] == TREASURE)

            self._map[new_pos] = avatar_team
            self._id2pos[avatar_id] = new_pos
            self._pos2id[new_pos] = avatar_id

            if collected_treasure:
                self._num_treasures_to_collect -= 1
                if self._num_treasures_to_collect > 0:
                    self._chased_treasure_pos = _coords_where(self._map == TREASURE)
                else:
                    self._chased_treasure_pos = None

    def _interpret_action(self, action_idx: int, team: int):
        """
        0,1,2,3 are always up,down,left,right
        if noop is enabled, but diagonals disabled: 4 is noop
        if noop is disabled, but diagonals enabled: 4,5,6,7 are upleft,upright,downleft,downright
        if both noop and diagonals are enabled: 4,5,6,7 are upleft,upright,downleft,downright and 8 is noop
        """
        if action_idx < 4:
            return action_idx

        noops = self._allow_noops[team]
        diags = self._allow_diagonals[team]
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

    def _iterate_avatars_alive(self, team=None):
        for id in range(self.num_avatars):
            if self.avatar_alive[id] and (team is None or self.id2team[id] == team):
                yield id

    def _rewards_mask(self, avatar_id) -> np.array:
        """ either to just the avatar or the entire team  """
        team = self.id2team[avatar_id]
        mask = np.zeros(self.num_avatars, bool)

        if not self._teamwide_rewards[team]:
            mask[avatar_id] = True
        else:
            for id in self._iterate_avatars_alive(team=team):
                mask[id] = True

        return mask

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

        for avatar_id in self._iterate_avatars_alive():
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
                if self._allow_wraparound[team]:
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

            # A thief managed to reach a treasure
            if avatar_team == THIEF and new_pos_type == TREASURE:
                self._move_or_kill(avatar_id, avatar_team, old_pos, new_pos)

                thief_reward, guardian_reward = REWARDS['treasure']
                reward[self._rewards_mask(avatar_id)] += thief_reward

                # Punish all guardians
                for id in self._iterate_avatars_alive(team=GUARDIAN):
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
                done[avatar_id] = True

                reward[self._rewards_mask(avatar_id)]   += thief_reward
                reward[self._rewards_mask(guardian_id)] += guardian_reward
                continue

            # A guardian managed to catch a thief, kill the thief and apply rewards
            if avatar_team == GUARDIAN and new_pos_type == THIEF:
                thief_id = self._pos2id[new_pos]

                self._move_or_kill(thief_id, THIEF, new_pos)
                done[thief_id] = True

                self._move_or_kill(avatar_id, GUARDIAN, old_pos, new_pos)

                reward[self._rewards_mask(thief_id)]  += thief_reward
                reward[self._rewards_mask(avatar_id)] += guardian_reward
                continue

        # No more thieves alive, the game is over (thieves and guardians have been rewarded at the moments of killing)
        if self._num_remaining_thieves == 0:
            done[:] = True
            info['end_reason'] = 'All thieves dead'

        # Zero disables episode end when treasures are collected
        if self._treasure_collection_limit != 0 and self._num_treasures_to_collect == 0:
            done[:] = True
            info['end_reason'] = 'Treasure(s) collected'

        self._time_left -= 1
        if self._time_left == 0:
            thief_reward, guardian_reward = REWARDS['out_of_time']
            info['end_reason'] = 'Out of time'
            done[:] = True
            # Apply reward to all avatars alive
            for id in self._iterate_avatars_alive():
                team = self.id2team[id]
                if team == GUARDIAN:
                    reward[id] += guardian_reward
                if team == THIEF:
                    reward[id] += thief_reward

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

        own = np.zeros_like(self._map, float)
        own[own_pos] = 1

        thieves   = (self._map == THIEF   ).astype(float)
        guardians = (self._map == GUARDIAN).astype(float)
        if own_team == THIEF:
            teammates = thieves
            opponents = guardians
        else:
            teammates = guardians
            opponents = thieves

        treasure_channel = (self._map == TREASURE).astype(float)

        # Channels first
        return np.stack([own, teammates, opponents, self._walls_channel, treasure_channel])

    def scripted_action(self, avatar_id):
        r, c = self._id2pos[avatar_id]
        team = self.id2team[avatar_id]

        # Thieves' target is a treasure
        if team == THIEF:
            if self._chased_treasure_pos is None:
                print(f'Warning: thief #{avatar_id} trying to chase a treasure, but there are none left.'
                      f'Defaulting to UP ({UP}).')
                return UP
            tr, tc = self._chased_treasure_pos

        # Guardians' target is a thief
        else:
            tr, tc = self._id2pos[self._chased_thief_id]

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
