""" Neural network architectures mapping state to action logits """

import numpy as np
import torch
import torch.nn as nn

from agent.distributions import Bernoulli, Categorical, DiagGaussian
from agent.utils import init, Flatten
from gym.spaces import Box, Discrete, MultiBinary


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


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super().__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            # TODO give option for lstm as well
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0)
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class ConvBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super().__init__(recurrent, hidden_size, hidden_size)
        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0),
            nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=1)), nn.ReLU(),
            # init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            # init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32, hidden_size)), nn.ReLU())

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()  # set module in training mode

    def forward(self, inputs, rnn_hxs, masks):
        # NaN are for avatars who are not playing anymore
        inputs[torch.isnan(inputs)] = 0

        x = self.main(inputs)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        value = self.critic_linear(x)

        return value, x, rnn_hxs



class FCBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super().__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(
            m,
            nn.init.orthogonal_,
            lambda x: nn.init.
            constant_(x, 0),
            np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()  # set module in training mode

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
