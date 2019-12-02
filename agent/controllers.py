""" An controller holds the encoder (fc/conv, may have a recurrent ending) and the decoder heads (actor, critic(s)) """

import torch
import torch.nn as nn

from configs.structure import Config
from agent.utils import copy_weights


ACTIVATION_FUNCTIONS = {
    'relu': torch.nn.ReLU,
    'leaky_relu': torch.nn.LeakyReLU,
    'tanh': torch.nn.Tanh
}


# TODO (?) Transformer encoder
# TODO (?) residual connection, etc

class RecurrentController(nn.Module):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__()

        """ Encoder """
        activation = ACTIVATION_FUNCTIONS[config.activation]
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

        self.variational = config.variational
        if self.variational:
            self.latent_means    = nn.Linear(decoders_inp_dim, decoders_inp_dim)
            self.latent_log_vars = nn.Linear(decoders_inp_dim, decoders_inp_dim)

        """ Decoder heads """
        if config.algorithm in ['pg', 'ppo', 'sac']:
            self.actor = _build_linear_decoder(config, decoders_inp_dim, num_actions, final_softmax=config.algorithm == 'sac')
        if config.algorithm in ['ppo']:
            self.critic = _build_linear_decoder(config, decoders_inp_dim, 1)
        if config.algorithm in ['sac']:
            # Takes the minimum out of the two critics to combat over-optimism
            self.critic_1        = _build_linear_decoder(config, decoders_inp_dim, num_actions)
            self.critic_2        = _build_linear_decoder(config, decoders_inp_dim, num_actions)

            # Target networks for the two critics to reduce variance
            self.critic_1_target = _build_linear_decoder(config, decoders_inp_dim, num_actions)
            self.critic_2_target = _build_linear_decoder(config, decoders_inp_dim, num_actions)

            # They both start at the same point
            copy_weights(self.critic_1, self.critic_1_target)
            copy_weights(self.critic_2, self.critic_2_target)

    def encode(self, env_states, rec_h_inp=None, rec_c_inp=None):
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

        if self.variational:
            means    = self.latent_means(encoder_out)
            log_vars = self.latent_log_vars(encoder_out)

            # Reparametrize
            stds = torch.exp(log_vars / 2)
            z = torch.randn_like(means, requires_grad=False)  # random
            encoder_out = z * stds + means
        else:
            means = None
            log_vars = None

        return means, log_vars, encoder_out, rec_h_out, rec_c_out


# TODO unify these builders
def _build_linear_decoder(config: Config, input_dim: int, output_dim: int, final_softmax=False):
    activation = ACTIVATION_FUNCTIONS[config.activation]

    fc1 = nn.Linear(input_dim, 64)
    fc2 = nn.Linear(64, 32)
    fc3 = nn.Linear(32, output_dim)

    _initialize_weights(fc1, config.activation)
    _initialize_weights(fc2, config.activation)
    _initialize_weights(fc3, config.activation)

    layers  = [fc1, activation()]
    if config.batch_norm: layers.append(nn.BatchNorm1d(64))

    layers += [fc2, activation()]
    if config.batch_norm: layers.append(nn.BatchNorm1d(32))

    layers += [fc3, activation()]

    # num_layers = config.num_decoder_layers
    # hidden_size = config.decoder_layer_size

    # in_sizes = [input_dim] + [hidden_size] * num_layers
    # layers = []
    # for i, in_size in enumerate(in_sizes):
    #     if i == len(in_sizes) - 1:
    #         out_size = output_dim
    #     else:
    #         out_size = in_sizes[i + 1]
    #     layers.append(nn.Linear(in_size, out_size))
    #     if config.batch_norm:
    #         layers.append(nn.BatchNorm1d(out_size))

    if config.layer_norm:
        layers.append(nn.LayerNorm([output_dim]))

    if final_softmax:
        layers.append(nn.Softmax(dim=1))

    return nn.Sequential(*layers)


def _initialize_weights(layer, activation_name: str):
    if activation_name == 'tanh':
        initializer = nn.init.xavier_normal_
    elif activation_name in ['relu', 'leaky_relu']:
        initializer = nn.init.xavier_uniform_
    gain = nn.init.calculate_gain(activation_name)
    initializer(layer.weight, gain=gain)


def _build_linear_encoder(config: Config, env_state_shape: tuple, activation):
    # Inputs are flattened
    num_inputs = 1
    for dim_size in env_state_shape:
        num_inputs *= dim_size

    layers = [torch.nn.Flatten()]
    if config.layer_norm:
        layers.append(nn.LayerNorm([num_inputs]))

    layer_sizes = [num_inputs] + [config.encoder_layer_size] * config.num_encoder_layers
    for layer_size in layer_sizes:
        layer = nn.Linear(layer_size, config.encoder_layer_size, bias=True)
        _initialize_weights(layer, config.activation)
        layers.append(layer)

        if config.batch_norm:  # TODO activation before batch norm?
            layers.append(nn.BatchNorm1d(config.encoder_layer_size))
        layers.append(activation())

    if config.layer_norm:
        layers.append(nn.LayerNorm([config.encoder_layer_size]))

    num_outputs = config.encoder_layer_size
    return num_outputs, nn.Sequential(*layers)


def _build_conv_encoder(config: Config, env_state_shape: tuple, activation):
    num_channels, width, height = env_state_shape

    conv1 = nn.Conv2d(num_channels, 8, kernel_size=1)
    conv2 = nn.Conv2d(8, 16, kernel_size=3)
    conv3 = nn.Conv2d(16, 16, kernel_size=3)

    _initialize_weights(conv1, config.activation)
    _initialize_weights(conv2, config.activation)
    _initialize_weights(conv3, config.activation)

    layers  = [conv1, activation()]
    if config.batch_norm: layers.append(nn.BatchNorm2d(8))

    layers += [conv2, activation()]
    if config.batch_norm: layers.append(nn.BatchNorm2d(16))

    layers += [conv3, activation()]
    if config.batch_norm: layers.append(nn.BatchNorm2d(16))

    num_outputs = 16 * 5 * 5

    # num_outputs = width * height * config.encoder_layer_size
    # num_padding = config.conv_kernel_size // 2
    #
    # layers = []
    # layer_sizes = [num_channels] + [config.encoder_layer_size] * config.num_encoder_layers
    # for layer_size in layer_sizes:
    #     layer = nn.Conv2d(layer_size, config.encoder_layer_size,
    #                       bias=True,
    #                       kernel_size=config.conv_kernel_size,
    #                       stride=1,
    #                       padding=num_padding, padding_mode='zeros')
    #     _initialize_weights(layer, config.activation)
    #     layers.append(layer)
    #
    #     if config.batch_norm:  # TODO activation before batch norm?
    #         layers.append(nn.BatchNorm2d(config.encoder_layer_size))
    #     layers.append(activation())

    layers.append(torch.nn.Flatten())
    if config.layer_norm:
        layers.append(nn.LayerNorm([num_outputs]))

    return num_outputs, nn.Sequential(*layers)
