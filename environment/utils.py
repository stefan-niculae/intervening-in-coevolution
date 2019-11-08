""" Functions to instantiate and handle environments """
import os
import time
from baselines import bench

from environment.ThievesGuardiansEnv import ThievesGuardiansEnv
from environment.visualization import EnvVisualizationWrapper


def make_env(env_name, seed, env_id, log_dir, allow_early_resets, visualization_wrapper=False):
    def _thunk():
        env = ThievesGuardiansEnv(env_name, env_id)
        env.seed(seed + env_id)

        if visualization_wrapper:
            env = EnvVisualizationWrapper(env)
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


# TODO understand what's happening with 'bad_transition' in main.py, and consider using this, by adding to our Env_max_episode_steps and _elapsed_steps to use this
# class TimeLimitMask(gym.Wrapper):
#     """ Checks whether done was caused by time limits """
#     def step(self, action):
#         obs, rew, done, info = self.env.step(action)
#         if done and self.env._max_episode_steps == self.env._elapsed_steps:
#             info['bad_transition'] = True
#
#         return obs, rew, done, info


def get_render_func(venv):
    if hasattr(venv, 'envs'):
        return venv.envs[0].render
    elif hasattr(venv, 'venv'):
        return get_render_func(venv.venv)
    elif hasattr(venv, 'env'):
        return get_render_func(venv.env)

    return None


