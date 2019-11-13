import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """ Environment """
    # Check environment/scenarios.py for more
    scenario: str = '4x4-thief-treasure'

    """ Policy """
    # Check agent/policies.py
    algorithm: str = 'pg'  # pg | ppo

    # Update clip param
    ppo_clip: float = .2

    # Critic
    critic_coef: float = .5

    # Encourage "exploration"
    entropy_coef: float = .01

    """ Controller """
    # Controller architecture — check agent/controllers.py
    controller: str = 'conv'  # fc | conv

    num_hidden_layers: int = 2

    hidden_layer_size: int = 32

    activation_function: str = 'relu'  # lrelu | relu | tanh

    """ Running """
    # Random seed
    seed: int = 0

    # Number of model updates
    num_updates: int = 100

    # Gather this many transitions before running a model update
    num_transitions: int = 1000

    # In how many batches to split the transitions
    num_batches: int = 5

    # How many times to iterate over all transitions
    num_epochs: int = 5

    """ Steering """
    # Discount future rewards (gamma)
    discount: float = .99

    """ Optimizer """
    # Learning rate
    lr: float = 7e-4

    # How often to decay the learning rate
    lr_decay_interval: int = 10

    # How much to decay the learning rate
    lr_decay_factor: float = .25

    # Adam optimizer parameter
    adam_epsilon: float = 1e-5

    # Max norm of the gradients
    max_grad_norm: float = .5

    """ Logging """
    # After how many updates to update the progress plot
    log_interval: int = 10

    # After how many updates to save the model
    save_interval: int = 10

    # After how many updates to evaluate (run deterministically) and save a video
    eval_interval: int = 5


def read_config(config_path: str) -> Config:
    with open(config_path) as f:
        dict_obj = json.load(f)
    return Config(**dict_obj)


def save_config(config: Config, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
