""" Neural network architectures mapping state to action logits """

import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, MultiBinary

from agent.distributions import Bernoulli, Categorical, DiagGaussian
from agent.models import FCBase, ConvBase


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_controllers, base_kind, base_kwargs=None):
        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if base_kind == 'fc':
            base_class = FCBase
        if base_kind == 'conv':
            base_class = ConvBase

        # Create identical controllers
        self.controllers = [base_class(obs_shape[0], **base_kwargs) for _ in range(num_controllers)]

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            dist_class = Categorical
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            dist_class = DiagGaussian
        elif isinstance(action_space, MultiBinary):
            num_outputs = action_space.shape[0]
            dist_class = Bernoulli
        else:
            raise NotImplementedError

        self.dist = dist_class(self.controllers[0].output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.controllers[0].is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """ Size of rnn_hx. """
        return self.controllers[0].recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def _run_controllers(self, controller_ids, inputs, rnn_hxs_input, masks):
        """ Distribute the load to each controller and then merge them back """
        first_two_dims = None
        batch_size = inputs.size(0)
        if len(inputs.shape) == 5:
            first_two_dims = inputs.shape[:2]
            batch_size = inputs.size(0) * inputs.size(1)
            # [num_processes, num_avatars, C, W, H] to [num_processes * num_avatars, C, W, H]
            inputs         = inputs       .view(-1, *inputs.shape[2:])
            rnn_hxs_input  = rnn_hxs_input.view(-1, *rnn_hxs_input.shape[2:])
            masks          = masks        .view(-1, *masks.shape[2:])
            controller_ids = controller_ids.flatten()

        value          = torch.zeros(batch_size, 1)
        actor_features = torch.zeros(batch_size, self.controllers[0].output_size)
        rnn_hxs_output = torch.zeros(batch_size, 1)
        for id, controller in enumerate(self.controllers):
            controller_mask = (controller_ids == id)
            if sum(controller_mask) == 0:
                continue

            # Act just on the samples meant for this input
            controller_value, \
            controller_actor_features, \
            controller_rnn_hxs \
                = controller(inputs       [controller_mask],
                             rnn_hxs_input[controller_mask],
                             masks        [controller_mask])

            # Place the results in the spots for them
            value         [controller_mask] = controller_value
            actor_features[controller_mask] = controller_actor_features
            rnn_hxs_output[controller_mask] = controller_rnn_hxs

        # Re-form the two dimensional batch shape
        if first_two_dims:
            value          = value.view         (*first_two_dims, *value.shape[1:])
            actor_features = actor_features.view(*first_two_dims, *actor_features.shape[1:])
            rnn_hxs_output = rnn_hxs_output.view(*first_two_dims, *rnn_hxs_output.shape[1:])

        return value, actor_features, rnn_hxs_output

    def act(self, controller_ids, inputs, rnn_hxs, masks, deterministic=False):
        """ Pick action """
        value, actor_features, rnn_hxs = self._run_controllers(controller_ids, inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, controller_ids, inputs, rnn_hxs, masks):
        value, _, _ = self._run_controllers(controller_ids, inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, controller_ids, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self._run_controllers(controller_ids, inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs

