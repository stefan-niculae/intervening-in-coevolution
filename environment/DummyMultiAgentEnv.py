import numpy as np
from gym import Env
from gym.spaces import Box


class DummyMultiAgentEnv(Env):
    """ Two avatars, starting at zero, can increment or decrement each timestep, between -10 and +10
        one gets rewards for going positive, the other for going negative. """

    def __init__(self, env_id):
        self.env_id = env_id
        self.num_avatars = 2

        # each of them "move" -1, 0 or +1
        self.action_space      = Box(low=-1,  high=+1, shape=(1,), dtype=int)
        # each of them move between -10 and +10, and also have an id of either -1 or +1
        self.observation_space = Box(low=-10, high=+10, shape=(4,), dtype=int)
        self.reward_range = (0, 10)

        self.positions = None
        self.reset()

    def compute_state(self):
        p1, p2 = self.positions
        return np.array([
            [+1, 0, 0, p1],
            [-1, 0, 0, p2],
        ])

    def reset(self):
        self.positions = np.zeros(2)
        return self.compute_state()

    def step(self, action: [int]):
        print('actions received', action.shape)
        self.positions += action

        state = self.compute_state()
        p1, p2 = self.positions
        reward = np.array([+p1, -p2])
        done = abs(self.positions) == 10
        info = {}

        return state, reward, done, info

    def render(self, mode='human'):
        print(self.positions)
