""" Neural network architectures mapping state to action logits """

import torch
import torch.nn as nn

from configs.structure import Config


def init(module, weight_init, bias_init, gain=1):
    # TODO use initialization
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'lrelu': torch.nn.LeakyReLU,
    'tanh': torch.nn.Tanh
}


# TODO (?) TransformerController
# TODO (?) batch normalization, dropout, etc

class RecurrentController(nn.Module):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__()
        self.hidden_layer_size    = config.hidden_layer_size
        self.recurrent_layer_size = config.recurrent_layer_size
        self.is_recurrent         = config.num_recurrent_layers > 0

        if self.is_recurrent:
            self.actor_critic_inp_dim = config.recurrent_layer_size
            self.recurrent = nn.LSTM(
                input_size=config.hidden_layer_size,  # number of features
                hidden_size=config.recurrent_layer_size,
                num_layers=config.num_recurrent_layers,
                bias=True,
                batch_first=True,  # expects shape [batch_size, num_timesteps, num_features], we always simulate on num_timesteps=1
                dropout=0.,
                bidirectional=False,
            )

            batch_size = 1  # when collecting rollouts, we pick transitions one by one
            self.rec_h0 = torch.randn(config.num_recurrent_layers, batch_size, config.recurrent_layer_size, requires_grad=True)
            self.rec_c0 = torch.randn(config.num_recurrent_layers, batch_size, config.recurrent_layer_size, requires_grad=True)

        else:
            # If the recurrent layer(s) are missing, we input directly from the base
            self.actor_critic_inp_dim = config.hidden_layer_size

    def forward(self, env_states, rec_h_inp, rec_c_inp):
        """
        Compute actor_logits and state_value from the actor and critic, and report the resulting recurrent state (h and c)

        Args:
            env_states:   float tensor of shape [batch_size, *env_state_shape]
            rec_h_inp: float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_c_inp: float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]

        Returns:
            actor_logits: float tensor of shape [batch_size, num_actions]
            state_value:  float tensor of shape [batch_size, 1]
            rec_h_out: float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_c_out: float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
        """
        base_out = self.base(env_states)  # shape: [batch_size, hidden_size]

        if self.is_recurrent:
            batch_size = env_states.size(0)
            rec_out, (rec_h_out, rec_c_out) = self.recurrent(
                # Add an extra dummy timestep dimension as 1
                base_out.view(batch_size, 1, self.hidden_layer_size),
                (rec_h_inp, rec_c_inp)
            )
            # Undo the dummy timestep of 1
            actor_critic_inp = rec_out.view(batch_size, self.recurrent_layer_size)
        else:
            actor_critic_inp = base_out
            rec_h_out = None
            rec_c_out = None

        actor_logits = self.actor (actor_critic_inp)
        state_values = self.critic(actor_critic_inp)

        return actor_logits, state_values, rec_h_out, rec_c_out


class FCController(RecurrentController):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__(config, env_state_shape, num_actions)

        # Inputs are flattened
        num_inputs = 1
        for dim_size in env_state_shape:
            num_inputs *= dim_size

        activation = ACTIVATION_FUNCTIONS[config.activation_function]

        hidden_layers = [
            nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            activation(),
        ] * config.num_hidden_layers
        self.base = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, config.hidden_layer_size),
            activation(),
            *hidden_layers
        )

        self.actor  = nn.Linear(self.actor_critic_inp_dim, num_actions)
        self.critic = nn.Linear(self.actor_critic_inp_dim, 1)

        self.train()  # set module in training mode


class ConvController(RecurrentController):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__(config, env_state_shape, num_actions)

        activation = ACTIVATION_FUNCTIONS[config.activation_function]
        num_channels, width, height = env_state_shape

        hidden_layers = [
            nn.Conv2d(config.hidden_layer_size, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
        ] * config.num_hidden_layers
        self.base = nn.Sequential(
            nn.Conv2d(num_channels, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
            *hidden_layers,
            torch.nn.Flatten(),
        )

        self.actor  = nn.Linear(width * height * self.actor_critic_inp_dim, num_actions)
        self.critic = nn.Linear(self.actor_critic_inp_dim, 1),

        self.train()  # set module in training mode


CONTROLLER_CLASSES = {
    'fc': FCController,
    'conv': ConvController,
}
