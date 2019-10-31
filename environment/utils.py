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

# TODO make a FlattenObservationWrapper and apply it if (len(observation_shape) > 1 and args.policy_base == 'fc'); and announce it
# TODO otherwise check that len(observation_shape) == 3 and issue a warning that images are interpreted as channel first

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
