""" Container for RL traces (env state, action, reward, recurrent state) and returns computation """

import torch
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from configs.structure import Config


class RolloutsStorage:
    def __init__(self, config: Config, env_state_shape: tuple):
        if config.num_recurrent_layers > 0:
            # This is the way torch expects hidden state: [num_recurrent_layers, batch_size, recurrent_size]
            self.rec_hs = torch.empty(config.num_recurrent_layers, config.num_transitions, config.recurrent_layer_size, dtype=torch.float32)
            self.rec_cs = torch.empty(config.num_recurrent_layers, config.num_transitions, config.recurrent_layer_size, dtype=torch.float32)
        else:
            self.rec_hs = None
            self.rec_cs = None

    def insert(self, env_state, action, action_log_prob, value, reward, done, rec_h, rec_c):
        pass

    def sample_batches(self):
        pass

    def _rec_state_at(self, indices):
        if self.rec_hs is None:
            rec_hs = None
            rec_cs = None
        else:
            rec_hs = self.rec_hs[:, indices]
            rec_cs = self.rec_cs[:, indices]
        return rec_hs, rec_cs

    def compute_returns(self):
        pass

    def reset(self):
        pass


class CyclicStorage(RolloutsStorage):
    def __init__(self, config: Config, env_state_shape: tuple):
        super().__init__(config, env_state_shape)
        self.size = config.memory_size
        self.batch_size = config.batch_size

        self.env_states       = torch.empty(self.size + 1, *env_state_shape, dtype=torch.float32,  requires_grad=False)
        self.actions          = torch.empty(self.size,                       dtype=torch.long,     requires_grad=False)
        self.rewards          = torch.empty(self.size + 1,                   dtype=torch.float32,  requires_grad=False)
        self.dones            = torch.empty(self.size + 1,                   dtype=torch.float32,  requires_grad=False)

        self.multi_step = 1
        self.insertion_index = 0
        self.max_index = 0

    def insert(self, env_state, action, action_log_prob, value, reward, done, rec_h, rec_c):
        self.env_states[self.insertion_index] = torch.tensor(env_state, dtype=torch.float32)
        self.actions   [self.insertion_index] = torch.tensor(action, dtype=torch.long)
        self.rewards   [self.insertion_index] = torch.tensor(reward, dtype=torch.float32)
        self.dones     [self.insertion_index] = torch.tensor(done, dtype=torch.float32)
        if self.rec_hs is not None:
            self.rec_hs[:, self.insertion_index] = rec_h.squeeze(1)
            self.rec_cs[:, self.insertion_index] = rec_c.squeeze(1)

        self.insertion_index += 1
        self.insertion_index %= self.size
        self.max_index = max(self.max_index, self.insertion_index)

    def sample_batches(self):
        # Assure there are enough samples to collect
        assert self.max_index - self.multi_step >= self.batch_size

        # Don't let multistepping cross over the seam
        valid_mask = np.ones(self.max_index - self.multi_step)
        valid_mask[self.insertion_index - self.multi_step : self.insertion_index] = 0

        sampler = BatchSampler(
            SubsetRandomSampler(np.where(valid_mask)[0]),
            self.batch_size,
            drop_last=True)

        # TODO PER https://github.com/Damcy/prioritized-experience-replay

        for indices in sampler:
            indices = torch.LongTensor(indices)
            multi_indices = indices + torch.LongTensor(range(self.multi_step + 1)).unsqueeze(1)

            yield (
                self.env_states[multi_indices],
                *self._rec_state_at(indices),
                self.actions[indices],
                self.rewards[indices],
                self.dones[indices],
            )


class OnPolicyStorage(RolloutsStorage):
    """ Stores transitions for a single avatar """
    def __init__(self, config: Config, env_state_shape: tuple):
        """ Instantiate empty (zero) tensors """
        super().__init__(config, env_state_shape)

        self.batch_size = config.batch_size
        self.discount = config.discount
        self.gae_lambda = config.gae_lambda

        self.num_recurrent_layers = config.num_recurrent_layers
        self.recurrent_layer_size = config.recurrent_layer_size

        self.env_states       = torch.empty(config.num_transitions, *env_state_shape, dtype=torch.float32,  requires_grad=False)
        self.actions          = torch.empty(config.num_transitions,                   dtype=torch.long,     requires_grad=False)
        self.action_log_probs = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)
        if config.gae_lambda > 0:
            self.values       = torch.empty(config.num_transitions + 1,               dtype=torch.float32,  requires_grad=False)
        else:
            self.values = None
        self.rewards          = torch.empty(config.num_transitions,                   dtype=torch.float32,  requires_grad=False)
        self.dones            = torch.empty(config.num_transitions + 1,               dtype=torch.float32,  requires_grad=False)  # has to be non-boolean to allow multiplication
        self.returns          = torch.empty(config.num_transitions + 1,               dtype=torch.float32,  requires_grad=False)

        self.step = None
        self.last_done = None
        self.reset()

    def insert(self, env_state, action, action_log_prob, value, reward, done, rec_h, rec_c):
        """
        Place at the current step and increment the step counter

        Args:
            env_state: array-like of shape env_state_shape
            action: int
            action_log_prob: float
            value: float
            reward: float
            done: bool
            rec_h: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
        """
        self.env_states      [self.step] = torch.tensor(env_state,       dtype=torch.float32)
        self.actions         [self.step] = torch.tensor(action,          dtype=torch.long)
        self.action_log_probs[self.step] = torch.tensor(action_log_prob, dtype=torch.float32)
        if self.values is not None:
            self.values      [self.step] = torch.tensor(value,           dtype=torch.float32)
        self.rewards         [self.step] = torch.tensor(reward,          dtype=torch.float32)
        self.dones           [self.step] = torch.tensor(done,            dtype=torch.float32)
        if self.rec_hs is not None:
            self.rec_hs      [:, self.step] = rec_h.view(self.num_recurrent_layers, self.recurrent_layer_size)
            self.rec_cs      [:, self.step] = rec_c.view(self.num_recurrent_layers, self.recurrent_layer_size)

        self.step += 1

    def reset(self):
        """ Clear out (set to nan so we can avoid if they are used before being filled) the current tensors and reset step """
        self.step = 0
        self.last_done = None

        to_reset = [self.env_states, self.actions, self.action_log_probs, self.values, self.rewards, self.dones, self.returns, self.rec_hs, self.rec_cs]
        for tensor in to_reset:
            if tensor is not None:
                tensor[:] = 0

    def compute_returns(self):
        """ Fills out self.returns until the last finished episode """
        self.last_done = (self.dones == 1).nonzero().max()

        # No GAE
        if self.gae_lambda == 0:
            # Accumulate discounted returns
            self.returns[self.last_done + 1] = 0.
            for t in reversed(range(self.last_done + 1)):
                self.returns[t] = self.returns[t + 1] * self.discount * (1 - self.dones[t]) + self.rewards[t]

        # Use Generalized Advantage Estimation
        if self.gae_lambda > 0:
            self.values[self.last_done + 1] = 0.
            self.dones [self.last_done + 1] = 1.
            gae = 0.
            for t in reversed(range(self.last_done + 1)):
                future_coef = self.discount * (1 - self.dones[t + 1])
                delta = self.rewards[t] + future_coef * self.values[t + 1] - self.values[t]
                gae = delta + future_coef * self.gae_lambda * gae
                self.returns[t] = gae + self.values[t]

    def sample_batches(self):
        """
        Sample transitions from all finished episodes
        Must be called after `self.compute_returns`

        Yields:
            env_states:       float tensor of shape [batch_size, *env_state_shape]
            actions:          long  tensor of shape [batch_size,]
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
            yield (
                self.env_states[indices],
                *self._rec_state_at(indices),
                self.actions[indices],
                self.action_log_probs[indices],
                self.returns[indices],
            )
