import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """ Environment """
    # Check environment/scenarios.py for more
    scenario: str = '4x4-thief-treasure'

    # Number of steps after which the guardians win
    # in each step all alive avatars move once
    time_limit: int = 30

    # Whether avatars can chose to do nothing for one timestep
    allow_noop: bool = False

    """ Policy """
    # Check agent/policies.py
    algorithm: str = 'pg'  # pg | ppo

    # Update clip param
    ppo_clip: float = .2

    # Critic
    critic_coef: float = .5

    # Encourage "exploration"
    entropy_coef: float = .01
    entropy_coef_decay_interval: int = 10
    entropy_coef_decay_factor: float = .1

    # Force exploration
    exploration_proba: float = .9
    exploration_proba_decay_interval: int = 10
    exploration_proba_decay_factor: float = .1

    """ Controller """
    # Controller architecture — check agent/controllers.py
    controller: str = 'conv'  # fc | conv

    num_hidden_layers: int = 2

    hidden_layer_size: int = 32

    activation_function: str = 'relu'  # lrelu | relu | tanh

    num_recurrent_layers: int = 1

    recurrent_layer_size: int = 16

    """ Running """
    # Random seed
    seed: int = 0

    num_demonstrative_updates: int = 10

    num_random_updates: int = 3

    # Number of model updates
    num_unguided_updates: int = 100

    # Gather this many transitions before running a model update
    num_transitions: int = 1000

    # How large the batches of transitions should be
    batch_size: int = 5

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
    log_interval: int = 10  # set to 0 for no logging

    # After how many updates to save the model
    save_interval: int = 10  # set to 0 for no saving

    # After how many updates to evaluate (run deterministically) and save a video
    eval_interval: int = 5  # set to 0 for no evaluation


def read_config(config_path: str) -> Config:
    with open(config_path) as f:
        dict_obj = json.load(f)
    return Config(**dict_obj)


def save_config(config: Config, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
