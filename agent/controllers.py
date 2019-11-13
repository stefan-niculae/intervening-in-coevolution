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


# TODO use config.num_hidden_layers
# TODO use functional api
# TODO (?) common base from which actor and critic branch
# TODO (?) recurrent layer at the end
# TODO (?) TransformerController

class FCController(nn.Module):
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__()

        # Inputs are flattened
        num_inputs = 1
        for dim_size in env_state_shape:
            num_inputs *= dim_size

        activation = ACTIVATION_FUNCTIONS[config.activation_function]

        self.actor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, num_actions),
            # nn.Linear(num_inputs, config.hidden_layer_size),
            # activation(),
            # nn.Linear(config.hidden_layer_size, config.hidden_layer_size),
            # activation(),
            # nn.Linear(config.hidden_layer_size, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_inputs, 1),
        )

        self.train()  # set module in training mode


class ConvController(nn.Module):
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__()

        activation = ACTIVATION_FUNCTIONS[config.activation_function]
        num_channels, width, height = env_state_shape

        self.actor = nn.Sequential(
            nn.Conv2d(num_channels, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
            nn.Conv2d(config.hidden_layer_size, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
            torch.nn.Flatten(),
            nn.Linear(width * height * config.hidden_layer_size, num_actions),
        )

        self.critic = nn.Sequential(
            nn.Conv2d(num_channels, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
            nn.Conv2d(config.hidden_layer_size, config.hidden_layer_size, kernel_size=1, stride=1),
            activation(),
            torch.nn.Flatten(),
            nn.Linear(config.hidden_layer_size, 1),
        )

        self.train()  # set module in training mode


CONTROLLER_CLASSES = {
    'fc': FCController,
    'conv': ConvController,
}
