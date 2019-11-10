""" Run multiple copies of an environment in parallel """

import torch
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv

from environment.utils import make_env


def make_vec_envs(config, device) -> 'VecEnv':
    envs = [
        make_env(config.scenario, config.seed, proc_number)
        for proc_number in range(config.num_processes)
    ]

    dummy_env = envs[0]()
    num_avatars = dummy_env.num_avatars

    if len(envs) > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    envs = VecEnv(envs, device)
    envs.num_avatars = num_avatars

    return envs


class VecEnv(VecEnvWrapper):
    """
    Vectorized environment: batches data from multiple copies of an
    environment so that each observation becomes a batch of observations
    and expected action is a batch of actions to be applied per-env.
    """
    def __init__(self, venv, device):
        super().__init__(venv)
        self.device = device

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(-1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


# TODO reimplement this (?)
# class VecNormalize(VecNormalize_):
#     """ Normalize observations and returns
#     :param gamma (float): discount factor (default 0.99)
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.training = True
#
#     def _obfilt(self, obs, update=True):
#         if not self.ob_rms:
#             return obs
#
#         if self.training and update:
#             self.ob_rms.update(obs)
#
#         obs = np.clip((obs - self.ob_rms.mean) /
#                       np.sqrt(self.ob_rms.var + self.epsilon),
#                       -self.clipob, self.clipob)
#         return obs
#
#     def train(self):
#         self.training = True
#
#     def eval(self):
#         self.training = False


# TODO reimplement this (?)
# class VecFrameStack(VecEnvWrapper):
#     """ Instead of an observation being just the latest timestep,
#         it is the latest n timesteps """
#     def __init__(self, venv, n_timesteps: int, device):
#         self.venv = venv
#         self.n_timesteps = n_timesteps
#
#         wos = venv.observation_space  # wrapped ob space
#         self.shape_dim0 = wos.shape[0]
#
#         low  = np.repeat(wos.low,  self.n_timesteps, axis=0)
#         high = np.repeat(wos.high, self.n_timesteps, axis=0)
#
#         self.stacked_obs = torch.zeros((venv.num_envs, ) +
#                                        low.shape).to(device)
#
#         observation_space = gym.spaces.Box(
#             low=low, high=high, dtype=venv.observation_space.dtype)
#         VecEnvWrapper.__init__(self, venv, observation_space=observation_space)
#
#     def step_wait(self):
#         obs, rews, news, infos = self.venv.step_wait()
#         self.stacked_obs[:, :-self.shape_dim0] = \
#             self.stacked_obs[:, self.shape_dim0:]
#         for (i, new) in enumerate(news):
#             if new:
#                 self.stacked_obs[i] = 0
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs, rews, news, infos
#
#     def reset(self):
#         obs = self.venv.reset()
#         if torch.backends.cudnn.deterministic:
#             self.stacked_obs = torch.zeros(self.stacked_obs.shape)
#         else:
#             self.stacked_obs.zero_()
#         self.stacked_obs[:, -self.shape_dim0:] = obs
#         return self.stacked_obs
#
#     def close(self):
#         self.venv.close()


# def get_vec_normalize(venv):
#     if isinstance(venv, VecNormalize):
#         return venv
#     elif hasattr(venv, 'venv'):
#         return get_vec_normalize(venv.venv)
#
#     return None
