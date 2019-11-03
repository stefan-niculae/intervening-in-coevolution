import numpy as np
from gym import Env
from gym.spaces import Discrete, Box


ACTIONS = {
    0: +1,
    1: -1,
}

class DummyMultiAgentEnv(Env):
    """ Two avatars, starting at zero, can increment or decrement each timestep, between -10 and +10
        one gets rewards for going positive, the other for going negative. """

    def __init__(self, env_id):
        self.env_id = env_id
        self.num_avatars = 2

        # each of them "move" -1 or +1
        self.action_space      = Discrete(2)
        # TODO it's Discrete, not Box for our thief&guardian env!!!!
        # each of them move between -10 and +10, and also have an id of either -1 or +1
        self.observation_space = Box(low=-10, high=+10, shape=(2,), dtype=int)
        self.reward_range = (0, 10)

        self.positions = None
        self.reset()

    def compute_state(self):
        """ state shape: (num_avatars, 2) """
        p1, p2 = self.positions
        return np.array([
            [+1, p1],
            [-1, p2],
        ])

    def reset(self):
        self.positions = np.zeros(2)
        return self.compute_state()

    def step(self, action: [int]):
        """
        actions shape: (num_avatars, 1)

        reward shape: (num_avatars,)
        done shape: bool
            done is used to reset the environment and other things.. should have another thing internally to see when individual avatars area "done"

        info: dict
        """
        [a1, a2] = action
        d1 = ACTIONS[a1]
        d2 = ACTIONS[a2]
        self.positions += [d1, d2]

        state = self.compute_state()
        p1, p2 = self.positions
        reward = np.array([+p1, -p2])
        done = all(abs(self.positions) == 10)
        info = {}

        return state, reward, done, info

    def render(self, mode='human'):
        print(self.positions)
