import json
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # Check scenarios.py for more
    scenario: str = '4x4-thief-treasure'

    """ Policy """
    # Only one implemented currently
    algorithm: str = 'PPO'

    # Update clip param
    ppo_clip: float = .2

    # Only one implemented currently
    controller: str = 'conv'

    """ Hardware """
    # Random seed
    seed: int = 0

    # CPU processes
    num_processes: int = 4

    # True not tested
    cuda: bool = False

    """ Running """
    # Number of model updates
    num_updates: int = 100

    # Gather this many transitions before running a model update
    num_transitions: int = 1000

    # Sample this many times per model update (ppo_epoch)
    num_batches: int = 5

    # Sample this many per batch
    batch_size: int = 256

    """ Steering """
    # Discount future rewards (gamma)
    discount: float = .99

    # Critic
    critic_coef: float = .5

    # Encourage "exploration"
    entropy_coef: float = .01

    """ Optimizer """
    # Learning rate
    lr: float = 7e-4

    # How often to decay the learning rate
    lr_decay_interval: int = 10

    # How much to decay the learning rate
    lr_decay_factor: float = .25

    # Adam optimizer parameter
    adam_epsilon: float = 1e-5

    # Max norm o fthe gradients
    max_grad_norm: float = .5

    """ Checkpointing """
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
