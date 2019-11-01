""" Functions to instantiate and handle environments """

import os

import gym
from baselines import bench


def make_env(env_id, seed, rank, log_dir, allow_early_resets):
    def _thunk():
        env = gym.make(env_id)  # TODO add our custom env here
        env.seed(seed + rank)

        # TODO understand what's happening with 'bad_transition' in main.py, and consider using this, by adding to our Env_max_episode_steps and _elapsed_steps to use this
        # env = TimeLimitMask(env)

        if log_dir is not None:
            env = bench.Monitor(
                env,
                os.path.join(log_dir, str(rank)),
                allow_early_resets=allow_early_resets)
        return env

    return _thunk

# TODO create a wrapper for an agent that picks action, and learns from on each avatar that is not `done`
# TODO either two separate policies or the same policy but with some kind of differentiator between the two teams

class TimeLimitMask(gym.Wrapper):
    """ Checks whether done was caused by time limits """
    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        if done and self.env._max_episode_steps == self.env._elapsed_steps:
            info['bad_transition'] = True

        return obs, rew, done, info


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None
