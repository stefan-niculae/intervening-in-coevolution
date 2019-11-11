""" Functions to instantiate and handle environments """
import numpy as np
import gym
from baselines import bench

from environment.ThievesGuardiansEnv import ThievesGuardiansEnv


def make_env(scenario, seed, env_id):
    def _thunk():
        env = ThievesGuardiansEnv(scenario, env_id)
        env.seed(seed + env_id)
        return env

    return _thunk


# class MultiAgentMonitor(bench.Monitor):
#     pass
#     def update(self, ob, rew, done, info):
#         self.rewards.append(rew)
#         if done:
#             self.needs_reset = True
#             eprew = sum(self.rewards)
#             eplen = len(self.rewards)
#             epinfo = {
#                 "r": eprew,
#                 "l": eplen,
#                 "t": time.time() - self.tstart,
#             }
#             for k in self.info_keywords:
#                 epinfo[k] = info[k]
#             self.episode_rewards.append(eprew)
#             self.episode_lengths.append(eplen)
#             self.episode_times.append(time.time() - self.tstart)
#             epinfo.update(self.current_reset_info)
#             if self.results_writer:
#                 self.results_writer.write_row(epinfo)
#             info['episode'] = epinfo
#
#         self.total_steps += 1


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


