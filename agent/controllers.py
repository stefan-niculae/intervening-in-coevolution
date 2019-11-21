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


# TODO (?) Transformer
# TODO (?) batch normalization, residual connection, etc

class RecurrentEncoder(nn.Module):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__()
        self.encoder_layer_size    = config.encoder_layer_size
        self.recurrent_layer_size = config.recurrent_layer_size
        self.is_recurrent         = config.num_recurrent_layers > 0
        self.activation = ACTIVATION_FUNCTIONS[config.activation_function]
        self.encoder = None  # will be set by children

    def _build_recurrent_and_decoders(self, config: Config, num_actions: int, encoder_out_dim: int):
        """ must be called by each child after setting self.encoder """
        if self.is_recurrent:
            actor_critic_inp_dim = config.recurrent_layer_size
            self.recurrent = nn.LSTM(
                input_size=encoder_out_dim,  # number of features
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
            # If the recurrent layer(s) are missing, we input directly from the encoder
            actor_critic_inp_dim = encoder_out_dim

        # Build decoder heads
        self.actor = None
        self.critic = None

        if config.algorithm in ['pg', 'ppo']:
            self.actor = self._make_linear_decoder(config, actor_critic_inp_dim, num_actions)
        if config.algorithm == 'ppo':
            self.critic = self._make_linear_decoder(config, actor_critic_inp_dim, 1)

    def _make_linear_decoder(self, config: Config, input_dim: int, output_dim: int):
        num_layers = config.num_decoder_layers
        hidden_size = config.decoder_layer_size

        if num_layers == 1:
            return nn.Linear(input_dim, output_dim)

        if num_layers == 2:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_size),
                nn.Linear(hidden_size, output_dim),
            )

        hidden_layers = [nn.Linear(hidden_size, hidden_size)] * hidden_size
        return nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            *hidden_layers,
            nn.Linear(hidden_size, output_dim),
        )

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
        encoder_out = self.encoder(env_states)  # shape: [batch_size, hidden_size]

        if self.is_recurrent:
            rec_out, (rec_h_out, rec_c_out) = self.recurrent(
                # Add an extra dummy timestep dimension as 1
                encoder_out.unsqueeze(1),
                (rec_h_inp, rec_c_inp)
            )
            # Undo the dummy timestep of 1
            actor_critic_inp = rec_out.squeeze(1)
        else:
            actor_critic_inp = encoder_out
            rec_h_out = None
            rec_c_out = None

        actor_logits = self.actor (actor_critic_inp)
        state_values = self.critic(actor_critic_inp) if self.critic else None

        return actor_logits, state_values, rec_h_out, rec_c_out


class FCEncoder(RecurrentEncoder):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__(config, env_state_shape, num_actions)

        # Inputs are flattened
        num_inputs = 1
        for dim_size in env_state_shape:
            num_inputs *= dim_size

        hidden_layers = [
            nn.Linear(config.encoder_layer_size, config.encoder_layer_size),
            self.activation(),
        ] * config.num_encoder_layers
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, config.encoder_layer_size),
            self.activation(),
            *hidden_layers
        )

        self._build_recurrent_and_decoders(config, num_actions,
                                           encoder_out_dim=config.encoder_layer_size)


class ConvEncoder(RecurrentEncoder):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__(config, env_state_shape, num_actions)

        num_channels, width, height = env_state_shape

        hidden_layers = [
            nn.Conv2d(config.encoder_layer_size, config.encoder_layer_size, kernel_size=1, stride=1),
            self.activation(),
        ] * config.num_encoder_layers
        self.encoder = nn.Sequential(
            nn.Conv2d(num_channels, config.encoder_layer_size, kernel_size=1, stride=1),
            self.activation(),
            *hidden_layers,
            torch.nn.Flatten(),
        )

        self._build_recurrent_and_decoders(config, num_actions,
                                           encoder_out_dim=width * height * config.encoder_layer_size)


CONTROLLER_CLASSES = {
    'fc': FCEncoder,
    'conv': ConvEncoder,
}
