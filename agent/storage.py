""" Container for RL traces (env state, action, reward, recurrent state) and returns computation """

import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class RolloutStorage:
    """ Stores transitions """
    def __init__(self, config, env_state_shape):
        """ Instantiate empty (zero) tensors """
        self.num_batches = config.num_batches
        self.discount = config.discount

        self.env_states       = torch.zeros(config.num_transitions, *env_state_shape, dtype=torch.float32)
        self.actions          = torch.zeros(config.num_transitions, dtype=torch.int64)
        self.action_log_probs = torch.zeros(config.num_transitions, dtype=torch.float32)
        self.rewards          = torch.zeros(config.num_transitions, dtype=torch.float32)
        self.dones            = torch.zeros(config.num_transitions, dtype=torch.float32)
        self.returns          = torch.zeros(config.num_transitions + 1, requires_grad=False, dtype=torch.float32)

        self.step = None
        self.last_done = None
        self.reset()

    def insert(self, env_state, action, action_log_prob, reward, done):
        """
        Place at the current step and increment the step counter

        Args:
            env_state: array-like of shape env_state_shape
            action: int
            action_log_prob: float
            reward: float
            done: bool(?)
        """
        self.env_states      [self.step].copy_(torch.tensor(env_state))
        self.actions         [self.step].copy_(torch.tensor(action))
        self.action_log_probs[self.step].copy_(torch.tensor(action_log_prob))
        self.rewards         [self.step].copy_(torch.tensor(reward))
        self.dones           [self.step].copy_(torch.tensor(done))

        self.step += 1

    def reset(self):
        """ Clear out (set to zero) the current tensors and reset step """
        self.step = 0
        self.last_done = None
        for tensor in [self.env_states, self.action_log_probs, self.action_log_probs, self.rewards, self.dones, self.returns]:
            tensor[:] = 0

    def compute_returns(self):
        """ Fills out self.returns until the last finished episode """
        self.last_done = (self.dones == 1).nonzero().max()
        self.returns[self.last_done+1] = 0.

        # Accumulate discounted returns
        for step in reversed(range(self.last_done+1)):
            self.returns[step] = self.returns[step + 1] * self.discount * (1 - self.dones[step]) + self.rewards[step]
            # TODO (?) GAE

    def sample_batches(self):
        """
        Sample transitions from all finished episodes
        Must be called after `self.compute_returns`

        Yields:
            env_states:       float tensor of shape [batch_size, *env_state_shape]
            actions:          int tensor of shape [batch_size,]
            action_log_probs: float tensor of shape [batch_size,]
            returns:          float tensor of shape [batch_size,]
        """
        # Splits all the transitions into `num_batches`,
        # each having approximately config.num_transitions // config.num_batches samples
        sampler = BatchSampler(
            SubsetRandomSampler(range(self.last_done)),
            self.num_batches,
            drop_last=True)

        for indices in sampler:
            yield (
                self.env_states[indices],
                self.actions[indices],
                self.action_log_probs[indices],
                self.returns[indices],
            )
