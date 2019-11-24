""" An controller holds the encoder (fc/conv, may have a recurrent ending) and the decoder heads (actor, critic(s)) """

import torch
import torch.nn as nn

from configs.structure import Config
from agent.utils import copy_weights


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'lrelu': torch.nn.LeakyReLU,
    'tanh': torch.nn.Tanh
}


# TODO (?) Transformer encoder
# TODO (?) batch/layer normalization, residual connection, etc
# TODO initialization: (tanh -> xavier; relu -> he)?

class RecurrentController(nn.Module):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__()

        """ Encoder """
        activation = ACTIVATION_FUNCTIONS[config.activation_function]
        if config.encoder == 'fc':
            encoder_out_dim, self.encoder = _build_linear_encoder(config, env_state_shape, activation)
        if config.encoder == 'conv':
            encoder_out_dim, self.encoder = _build_conv_encoder(config, env_state_shape, activation)

        """ Encoder recurrent component """
        self.is_recurrent = config.num_recurrent_layers > 0
        if self.is_recurrent:
            decoders_inp_dim = config.recurrent_layer_size
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
            decoders_inp_dim = encoder_out_dim

        """ Decoder heads """
        if config.algorithm in ['pg', 'ppo', 'sac']:
            self.actor = _build_linear_decoder(config, decoders_inp_dim, num_actions)
        if config.algorithm in ['ppo', 'sac']:
            self.critic = _build_linear_decoder(config, decoders_inp_dim, 1)
        if config.algorithm in ['sac']:
            # Takes the minimum out of the two critics to combat over-optimism
            self.critic_2        = _build_linear_decoder(config, decoders_inp_dim, 1)

            # Target networks for the two critics to reduce variance
            self.critic_target   = _build_linear_decoder(config, decoders_inp_dim, 1)
            self.critic_2_target = _build_linear_decoder(config, decoders_inp_dim, 1)

            # They both start at the same point
            copy_weights(self.critic,   self.critic_target)
            copy_weights(self.critic_2, self.critic_target_2)

    def encode(self, env_states, rec_h_inp, rec_c_inp):
        """
        Run encoder, with logic for the optional recurrent part

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
            encoder_out = rec_out.squeeze(1)
        else:
            rec_h_out = None
            rec_c_out = None
        # TODO variational
        return encoder_out, rec_h_out, rec_c_out


# TODO unify these builders
def _build_linear_decoder(config: Config, input_dim: int, output_dim: int):
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


def _build_linear_encoder(config: Config, env_state_shape: tuple, activation):
    # Inputs are flattened
    num_inputs = 1
    for dim_size in env_state_shape:
        num_inputs *= dim_size

    hidden_layers = [
        nn.Linear(config.encoder_layer_size, config.encoder_layer_size),
        activation(),
    ] * config.num_encoder_layers

    num_outputs = config.encoder_layer_size
    return num_outputs, nn.Sequential(
        nn.Flatten(),
        nn.Linear(num_inputs, config.encoder_layer_size),
        activation(),
        *hidden_layers
    )


def _build_conv_encoder(config: Config, env_state_shape: tuple, activation):
    num_channels, width, height = env_state_shape

    hidden_layers = [
        nn.Conv2d(config.encoder_layer_size, config.encoder_layer_size, kernel_size=1, stride=1),
        activation(),
    ] * config.num_encoder_layers

    num_outputs = width * height * config.encoder_layer_size
    return num_outputs, nn.Sequential(
        nn.Conv2d(num_channels, config.encoder_layer_size, kernel_size=1, stride=1),
        activation(),
        *hidden_layers,
        torch.nn.Flatten(),
    )
