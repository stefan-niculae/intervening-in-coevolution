from gym.spaces import Discrete
from gym.spaces import Box
from gym import Env
import numpy as np


class Hide_and_seek_Env(Env):

    def __init__(self, w=5, h=5, time_limit=30, n_hiders=2, n_seekers=2, wall_density=0.0):

        self.hider_type = "hider"
        self.seeker_type = "seeker"
        self.w = w
        self.h = h
        self.time_limit = time_limit
        self.n_hiders = n_hiders
        self.n_seekers = n_seekers
        self.wall_density = wall_density
        self.time = 0
        self.init_map()
        self.movements = [(0, 0), (0, 1), (1, 0), (0, -1), (-1, 0)]
        self.remaining_hiders = self.n_hiders

        self.action_space = Box(low=0, high=len(self.movements)-1, shape=(self.n_hiders+self.n_seekers), dtype=int)
        self.observation_space = Box(low=-1, high=self.n_hiders+self.n_seekers, shape=(self.h, self.w), dtype=int)

    def init_map(self):
        self.map = np.zeros((self.h, self.w))
        a = np.random.rand(self.h, self.w)
        self.map[a < self.wall_density] = -1
        self.init_agents()

    def init_agents(self):
        hiders_list = []
        seekers_list = []
        starting_quad = np.random.randint(2, size=2)
        for i in range(self.n_hiders):
            posx = int(np.random.randint(self.w / 2) + starting_quad[0] * self.w / 2)
            posy = int(np.random.randint(self.h / 2) + starting_quad[1] * self.h / 2)
            while self.map[posy, posx] != 0:
                posx = int(np.random.randint(self.w / 2) + starting_quad[0] * self.w / 2)
                posy = int(np.random.randint(self.h / 2) + starting_quad[1] * self.h / 2)
            self.map[posy, posx] = i + 1
            hiders_list.append(Agent(i + 1, self.hider_type, [posx, posy]))

        for i in range(self.n_seekers):
            posx = int(np.random.randint(self.w / 2) + (1 - starting_quad[0]) * self.w / 2)
            posy = int(np.random.randint(self.h / 2) + (1 - starting_quad[1]) * self.h / 2)
            while self.map[posy, posx] != 0:
                posx = int(np.random.randint(self.w / 2) + (1 - starting_quad[0]) * self.w / 2)
                posy = int(np.random.randint(self.h / 2) + (1 - starting_quad[1]) * self.h / 2)
            self.map[posy, posx] = i + 1 + self.n_hiders
            seekers_list.append(Agent(i + 1 + self.n_hiders, self.seeker_type, [posx, posy]))

        self.agents = hiders_list + seekers_list

    def reset(self):
        self.init_map()

    def step(self, action):
        for idx, act in enumerate(action):
            agent = self.agents[idx]
            if agent.alive:
                init_pos_x, init_pos_y = agent.pos
                pos_x, pos_y = self.movements[act][0] + init_pos_x, self.movements[act][1] + init_pos_y
                if 0 <= pos_x < self.w and 0 <= pos_y < self.h:
                    if self.map[pos_y, pos_x] == 0:
                        self.map[init_pos_y, init_pos_x] = 0
                        self.map[pos_y, pos_x] = agent.id
                        agent.pos = pos_x, pos_y
                    #     case when seeker catches a hider
                    elif 0 < self.map[pos_y, pos_x] <= self.n_hiders and agent.agent_type == self.seeker_type:
                        self.agents[int(self.map[pos_y, pos_x]-1)].alive = False
                        self.remaining_hiders -= 1
                        self.map[init_pos_y, init_pos_x] = 0
                        self.map[pos_y, pos_x] = agent.id
                        agent.pos = pos_x, pos_y
        done = False
        reward = 0
        inf = {}
        self.time += 1
        if self.remaining_hiders == 0:
            done = True
            reward = -1
        elif self.time == self.time_limit:
            done = True
            reward = 1
        return np.copy(self.map), reward, done, inf

    def render(self, mode='matrix'):
        print(self.map)

    def close(self):
        pass

    def seed(self, seed=None):
        pass

class Agent:
    def __init__(self, id, agent_type, pos):
        self.id = id
        self.agent_type = agent_type
        self.alive = True
        self.pos = pos


e = Hide_and_seek_Env()

e.render()
e.step([1,1,4,4])
e.render()