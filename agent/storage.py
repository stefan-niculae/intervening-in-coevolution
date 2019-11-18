""" Container for RL traces (env state, action, reward, recurrent state) and returns computation """

import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    """ Stores transitions for a single avatar """
    def __init__(self, config, env_state_shape):
        """ Instantiate empty (zero) tensors """
        self.batch_size = config.batch_size
        self.discount = config.discount

        self.num_recurrent_layers = config.num_recurrent_layers
        self.recurrent_layer_size = config.recurrent_layer_size
        self.controller_recurrent = config.num_recurrent_layers > 0  # assumes homogenous controllers

        self.env_states       = torch.empty(config.num_transitions, *env_state_shape, dtype=torch.float32,  requires_grad=False)
        self.actions          = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)  # float32 so it can be filled with nan
        self.action_log_probs = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)
        self.rewards          = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)
        self.dones            = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)  # has to be non-boolean to allow multiplication
        self.returns          = torch.empty(config.num_transitions + 1,               dtype=torch.float32,  requires_grad=False)
        if self.controller_recurrent:
            # This is the way torch expects hidden state: [num_recurrent_layers, batch_size, recurrent_size]
            self.rec_hs       = torch.empty(config.num_recurrent_layers, config.num_transitions, config.recurrent_layer_size, dtype=torch.float32)
            self.rec_cs       = torch.empty(config.num_recurrent_layers, config.num_transitions, config.recurrent_layer_size, dtype=torch.float32)

        self.step = None
        self.last_done = None
        self.reset()

    def insert(self, env_state, action, action_log_prob, reward, done, rec_h, rec_c):
        """
        Place at the current step and increment the step counter

        Args:
            env_state: array-like of shape env_state_shape
            action: int
            action_log_prob: float
            reward: float
            done: bool
            rec_h: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
        """
        self.env_states      [self.step] = torch.tensor(env_state,       dtype=torch.float32)
        self.actions         [self.step] = torch.tensor(action,          dtype=torch.float32)
        self.action_log_probs[self.step] = torch.tensor(action_log_prob, dtype=torch.float32)
        self.rewards         [self.step] = torch.tensor(reward,          dtype=torch.float32)
        self.dones           [self.step] = torch.tensor(done,            dtype=torch.float32)
        if self.controller_recurrent:
            self.rec_hs      [:, self.step] = rec_h.view(self.num_recurrent_layers, self.recurrent_layer_size)
            self.rec_cs      [:, self.step] = rec_c.view(self.num_recurrent_layers, self.recurrent_layer_size)

        self.step += 1

    def reset(self):
        """ Clear out (set to nan so we can avoid if they are used before being filled) the current tensors and reset step """
        self.step = 0
        self.last_done = None

        to_reset = [self.env_states, self.actions, self.action_log_probs, self.rewards, self.dones, self.returns]
        if self.controller_recurrent:
            to_reset += [self.rec_hs, self.rec_cs]
        for tensor in to_reset:
            # Set to NaN to be able to detect if wrong elements are sampled
            tensor[:] = np.nan

    def compute_returns(self):
        """ Fills out self.returns until the last finished episode """
        self.last_done = (self.dones == 1).nonzero().max()
        self.returns[self.last_done + 1] = 0.

        # Accumulate discounted returns
        for step in reversed(range(self.last_done+1)):
            self.returns[step] = self.returns[step + 1] * self.discount * (1 - self.dones[step + 1]) + self.rewards[step]
            # TODO (?) GAE

    def sample_batches(self):
        """
        Sample transitions from all finished episodes
        Must be called after `self.compute_returns`

        Yields:
            env_states:       float tensor of shape [batch_size, *env_state_shape]
            actions:          int   tensor of shape [batch_size,]
            action_log_probs: float tensor of shape [batch_size,]
            returns:          float tensor of shape [batch_size,]
            rec_hs:           float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
                or None if the controller is not recurrent
            rec_cs:           float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
                or None if the controller is not recurrent
        """
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done + 1)),
            self.batch_size,
            drop_last=True)

        for indices in sampler:
            if self.controller_recurrent:
                rec_hs = self.rec_hs[:, indices]
                rec_cs = self.rec_cs[:, indices]
            else:
                rec_hs = None
                rec_cs = None
            yield (
                self.env_states[indices],
                rec_hs,
                rec_cs,
                self.actions[indices],
                self.action_log_probs[indices],
                self.returns[indices],
            )
