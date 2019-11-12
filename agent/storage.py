""" Container for RL traces (env state, action, reward, recurrent state) and returns computation """

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from gym.spaces import Discrete
import numpy as np


class RolloutStorage:
    def __init__(self, config, num_avatars, obs_shape, action_space, recurrent_hidden_state_size):
        self.num_transitions = config.num_transitions
        self.num_processes = config.num_processes
        self.num_avatars = num_avatars
        self.batch_size = config.batch_size

        self.discount = config.discount
        # TODO GAE

        self.env_state   = torch.zeros(self.num_transitions + 1, self.num_processes, num_avatars, *obs_shape)
        self.rec_state   = torch.zeros(self.num_transitions + 1, self.num_processes, num_avatars, recurrent_hidden_state_size)
        self.reward      = torch.zeros(self.num_transitions, self.num_processes, num_avatars, 1)
        self.returns     = torch.zeros(self.num_transitions + 1, self.num_processes, num_avatars, 1)
        self.value_pred  = torch.zeros(self.num_transitions + 1, self.num_processes, num_avatars, 1)
        self.action_prob = torch.zeros(self.num_transitions, self.num_processes, num_avatars, 1)  # log probs actually
        self.done        = torch.ones (self.num_transitions + 1, self.num_processes, num_avatars, 1)
        self.controller  = np.zeros(  (self.num_transitions + 1, self.num_processes, num_avatars), np.uint8)
        if isinstance(action_space, Discrete):
            self.action  = torch.zeros(self.num_transitions, self.num_processes, num_avatars, 1).long()  # action picked
        else:
            self.action  = torch.zeros(self.num_transitions, self.num_processes, num_avatars, *action_space.shape)

        self.step = 0

    def to(self, device):
        self.env_state   = self.env_state   .to(device)
        self.rec_state   = self.rec_state   .to(device)
        self.reward      = self.reward     .to(device)
        self.value_pred  = self.value_pred .to(device)
        self.returns     = self.returns     .to(device)
        self.action_prob = self.action_prob.to(device)
        self.action      = self.action     .to(device)
        self.done        = self.done        .to(device)

    def insert(self, env_state, rec_state, action, action_prob, value_preds, rewards, done, controller):
        """ Insert one transition and advance the step """
        self.env_state  [self.step + 1].copy_(env_state)
        self.rec_state  [self.step + 1].copy_(rec_state)
        self.action     [self.step    ].copy_(action)
        self.action_prob[self.step    ].copy_(action_prob)
        self.value_pred [self.step    ].copy_(value_preds)
        self.reward     [self.step    ].copy_(rewards)
        self.done       [self.step + 1].copy_(done)
        self.controller [self.step + 1] = controller.copy()

        self.step += 1

    def clear(self):
        """ Place the last ones on the first position, for when we finished collecting transitions but the env is not done  """
        self.env_state [0].copy_(self.env_state[-1])
        self.rec_state [0].copy_(self.rec_state[-1])
        self.done      [0].copy_(self.done[-1])
        self.controller[0] = self.controller[-1].copy()
        self.step = 1

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for step in reversed(range(self.num_transitions)):
            self.returns[step] = self.reward[step] + \
                                 (1-self.done[step]) * self.discount * self.returns[step + 1]
        # if use_gae:
        #     self.value_preds[-1] = next_value
        #     gae = 0
        #     for step in reversed(range(self.num_steps)):
        #         delta = self.rewards[step] + discount * self.value_preds[step + 1]\
        #                 * self.masks[step + 1] - self.value_preds[step]
        #         gae = delta + discount * gae_lambda * self.masks[step + 1] * gae
        #         self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator(self, advantages):
        # Just those indices where there are no nans in the state (alive avatars)
        valid_indices = (~torch.isnan(self.env_state[:-1])).any(-1).any(-1).any(-1).nonzero().flatten()

        # If we sample from a 1D tensor, the `indices` will a list of tensors,
        # which does not go well with torch's `index_select` or tensor[indices]
        indices = BatchSampler(
            SubsetRandomSampler(valid_indices.tolist()),
            self.batch_size,
            drop_last=True)

        for i in indices:
            controller       = self.controller [:-1].reshape(-1)[i]
            env_state        = self.env_state  [:-1].view(-1, *self.env_state.shape[3:])[i]
            rec_state        = self.rec_state  [:-1].view(-1, self.rec_state.size(-1))[i]
            actions          = self.action          .view(-1, self.action.size(-1))[i]
            value_pred       = self.value_pred [:-1].view(-1, 1)[i]
            returns          = self.returns    [:-1].view(-1, 1)[i]
            done             = self.done       [:-1].view(-1, 1)[i]
            old_action_probs = self.action_prob     .view(-1, 1)[i]
            if advantages is not None:
                adv_targ     = advantages           .view(-1, 1)[i]
            else:
                adv_targ = None

            yield controller, env_state, rec_state, actions, value_pred, returns, done, old_action_probs, adv_targ

    # def recurrent_generator(self, advantages, num_mini_batch):
    #     # TODO take into account new `num_avatar` dimension, and ignore ones where rewards is -inf
    #     assert self.num_processes >= num_mini_batch, (
    #         "PPO requires the number of processes ({}) "
    #         "to be greater than or equal to the number of "
    #         "PPO mini batches ({}).".format(self.num_processes, num_mini_batch))
    #     num_envs_per_batch = self.num_processes // num_mini_batch
    #     perm = torch.randperm(self.num_processes)
    #     for start_ind in range(0, self.num_processes, num_envs_per_batch):
    #         obs_batch = []
    #         recurrent_hidden_states_batch = []
    #         actions_batch = []
    #         value_preds_batch = []
    #         return_batch = []
    #         masks_batch = []
    #         old_action_log_probs_batch = []
    #         adv_targ = []
    #
    #         for offset in range(num_envs_per_batch):
    #             ind = perm[start_ind + offset]
    #             obs_batch.append(self.obs[:-1, ind])
    #             recurrent_hidden_states_batch.append(
    #                 self.recurrent_hidden_states[0:1, ind])
    #             actions_batch.append(self.actions[:, ind])
    #             value_preds_batch.append(self.value_preds[:-1, ind])
    #             return_batch.append(self.returns[:-1, ind])
    #             masks_batch.append(self.masks[:-1, ind])
    #             old_action_log_probs_batch.append(
    #                 self.action_log_probs[:, ind])
    #             adv_targ.append(advantages[:, ind])
    #
    #         T, N = self.num_steps, num_envs_per_batch
    #         # These are all tensors of size (T, N, -1)
    #         obs_batch = torch.stack(obs_batch, 1)
    #         actions_batch = torch.stack(actions_batch, 1)
    #         value_preds_batch = torch.stack(value_preds_batch, 1)
    #         return_batch = torch.stack(return_batch, 1)
    #         masks_batch = torch.stack(masks_batch, 1)
    #         old_action_log_probs_batch = torch.stack(
    #             old_action_log_probs_batch, 1)
    #         adv_targ = torch.stack(adv_targ, 1)
    #
    #         # States is just a (N, -1) tensor
    #         recurrent_hidden_states_batch = torch.stack(
    #             recurrent_hidden_states_batch, 1).view(N, -1)
    #
    #         # Flatten the (T, N, ...) tensors to (T * N, ...)
    #         obs_batch = _flatten_helper(T, N, obs_batch)
    #         actions_batch = _flatten_helper(T, N, actions_batch)
    #         value_preds_batch = _flatten_helper(T, N, value_preds_batch)
    #         return_batch = _flatten_helper(T, N, return_batch)
    #         masks_batch = _flatten_helper(T, N, masks_batch)
    #         old_action_log_probs_batch = _flatten_helper(T, N,
    #                 old_action_log_probs_batch)
    #         adv_targ = _flatten_helper(T, N, adv_targ)
    #
    #         yield obs_batch, recurrent_hidden_states_batch, actions_batch, \
    #             value_preds_batch, return_batch, masks_batch, old_action_log_probs_batch, adv_targ
