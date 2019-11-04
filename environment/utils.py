""" Functions to instantiate and handle environments """

import os
import time

import gym
from baselines import bench

from environment.DummyMultiAgentEnv import DummyMultiAgentEnv
from environment.Hide_and_seek_Env import Hide_and_seek_Env


def make_env(env_name, seed, env_id, log_dir, allow_early_resets):
    def _thunk():
        env = Hide_and_seek_Env(env_id)
        env.seed(seed + env_id)

        # TODO understand what's happening with 'bad_transition' in main.py, and consider using this, by adding to our Env_max_episode_steps and _elapsed_steps to use this
        # env = TimeLimitMask(env)

        # TODO implement custom monitor, with vectorized reward
        if log_dir is not None:
            env = MultiAgentMonitor(
                env,
                os.path.join(log_dir, str(env_id)),
                allow_early_resets=allow_early_resets)
        return env

    return _thunk


class MultiAgentMonitor(bench.Monitor):
    def update(self, ob, rew, done, info):
        self.rewards.append(rew)
        if done:
            self.needs_reset = True
            eprew = sum(self.rewards)
            eplen = len(self.rewards)
            epinfo = {"r": eprew, "l": eplen, "t": round(time.time() - self.tstart, 6)}
            for k in self.info_keywords:
                epinfo[k] = info[k]
            self.episode_rewards.append(eprew)
            self.episode_lengths.append(eplen)
            self.episode_times.append(time.time() - self.tstart)
            epinfo.update(self.current_reset_info)
            if self.results_writer:
                self.results_writer.write_row(epinfo)
            assert isinstance(info, dict)
            if isinstance(info, dict):
                info['episode'] = epinfo

        self.total_steps += 1


# class DummyMultiAvatarWrapper(gym.Wrapper):
#     """ for testing on classic environments (one avatar), but for the algorithm that can handle multiple avatars """
#     def reset(self):
#         state = self.env.reset()
#         print('reset expand', np.expand_dims(state, axis=0).shape)
#         return np.expand_dims(state, axis=0)
#
#     def step(self, action):
#         action = action[0]
#         state, reward, done, info = self.env.step(action)
#         return np.expand_dims(state, axis=0), np.expand_dims(reward, axis=0), np.expand_dims(done, axis=0), info


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
