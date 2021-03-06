import json
from dataclasses import dataclass, asdict
from typing import List


@dataclass
class Config:
    """ Environment """
    # Check environment/scenarios.py for more
    scenario: str = '4x4,1v1'

    # Number of steps after which the guardians win: in each step all alive avatars move once
    time_limit: int = 50

    # How many treasure collections make the environment end
    # zero to never end the episode upon treasure collection, -1 to end the episode when all treasures have been collected
    treasure_collection_limit: int = -1

    # Whether avatars in each team can chose to do nothing for one timestep
    allow_noop: List[bool] = (False, False)

    # Whether avatars in each team can move in eight directions, diagonally
    allow_diagonals: List[bool] = (False, False)

    # Whether avatars going past the right edge will end up on the left edge (and all other edges)
    allow_wraparound: List[bool] = (False, False)

    # How the environment outputs states
    state_representation: str = 'coordinates'  # grid | coordinates

    # When an avatar reaches their target (treasure/thief),
    # whether only the avatar (False) or the entire team (True) should get the reward
    teamwide_rewards: List[bool] = (False, False)

    """ Policy """
    # Check agent/policies.py
    algorithm: str = 'pg'  # pg | ppo | sac

    # PPO Update clip param
    ppo_clip: float = .2

    # PPO Critic coefficient
    ppo_critic_coef: float = .5

    # SAC target smoothing coefficient
    sac_tau: float = 5e-3

    # SAC temperature: determines the relative importance of the entropy term against the reward
    sac_alpha: float = .2

    # Whether to automatically tune entropy
    sac_dynamic_entropy: bool = False

    """ Intervening """
    # Encourage exploration: by optimizing for diversity in action distributions
    entropy_coef_milestones:   List[int]   = (  0,)
    entropy_coef_values:       List[float] = (.01,)

    # Force exploration: sample actions at random
    uniform_proba_milestones:  List[int]   = ( 0,)
    uniform_proba_values:      List[float] = ( 0,)

    # Don't consult the model in sampling actions, instead follow pre-scripted behavior
    scripted_proba_milestones: List[int]   = (  0,)
    scripted_proba_values:     List[float] = (  0,)

    # Learning rate
    lr_milestones:             List[int]   = (   0,)
    lr_values:                 List[float] = (.001,)

    # Inverse sampling
    inverse_proba_milestones:  List[int] = (0,)
    inverse_proba_values:      List[int] = (0.,)

    variational_constraint_milestones: List[int]   = (0,)
    variational_constraint_values:     List[float] = (0,)

    latent_constraint_milestones: List[int]   = (0,)
    latent_constraint_values:     List[float] = (0,)

    # Winrate for one side to kick in intervention tactics (1.0 to disable)
    winrate_threshold: float = 1

    first_no_adjustment: int = 0  # how many iterations to ignore adjusting (zero to disable)

    adjust_lr_to: float = None
    adjust_mi_to: float = None
    adjust_uniform_to: float = None
    adjust_scripted_to: float = None

    """ Encoder """
    # Encoder architecture — check agent/controllers.py
    encoder: str = 'fc'  # fc | conv

    # Whether to treat encoder output as mean and variance and sample from it
    variational: bool = False

    activation: str = 'relu'  # leaky_relu | relu | tanh

    num_encoder_layers: int = 2
    encoder_layer_size: int = 32

    num_recurrent_layers: int = 0
    recurrent_layer_size: int = 16

    num_decoder_layers: int = 1
    decoder_layer_size: int = 32

    batch_norm: bool = False

    layer_norm: bool = False

    conv_kernel_size: int = 1

    """ Running """
    # Random seed
    seed: int = 0

    max_run_time: float = 999  # in hours, set to 9999 to disable

    # Number of model updates
    num_iterations: int = 60

    # Gather this many transitions for each iteration
    num_transitions: int = 2000

    # Remember this many
    memory_size: int = None

    # How large the batches of transitions should be
    batch_size: int = 256

    # How many times to iterate over all transitions
    num_epochs: int = 5

    """ Steering """
    # Discount future rewards (gamma)
    discount: float = .99

    multi_step: int = 1

    """ Optimizer """
    # Adam optimizer parameter
    adam_epsilon: float = 1e-5

    # Max norm of the gradients
    max_grad_norm: float = 5

    # Generalized Advantage Estimation lambda (0 to disable)
    gae_lambda: float = 0

    """ Logging """
    """ Set to 0 to disable, if non-zero, will also do it on the last iteration """
    # Every how many iterations to update the progress plot
    log_interval: int = 1

    # Every how many iterations to save the model
    save_interval: int = 999

    # Every how many iterations to film a rollout
    eval_interval: int = 5

    # Every how many iterations to compare against other models
    compare_interval: int = 0
    comparison_models_dir: str = 'tuning/comparison_models'

    # How many matches to play against external opponents or themselves (at the end of training)
    comparison_num_episodes: int = 5

    # Whether to just visualize the scripted behavior, no training
    viz_scripted_mode: bool = False


def read_config(config_path: str) -> Config:
    with open(config_path) as f:
        raw_dict = json.load(f)

    fields = Config.__dict__
    parsed_dict = {}
    for k, v in raw_dict.items():
        # Example: lr=.1 -> lr_values=[.1], lr_milestones=[0]
        if k not in fields and k + '_values' in fields:
            parsed_dict[k + '_values'] = [v]
            parsed_dict[k + '_milestones'] = [0]
        else:
            parsed_dict[k] = v

    config = Config(**parsed_dict)

    if config.gae_lambda > 0:
        assert config.algorithm == 'ppo',  'GAE defined only for PPO'

    if config.state_representation == 'coordinates':
        assert config.encoder != 'conv', 'Convolution cannot be used on a list of coordinates'

    if config.multi_step > 1 or config.memory_size:
        assert config.algorithm == 'sac'

    assert config.conv_kernel_size % 2 == 1, 'Kernel size must be an odd number'

    return config


def save_config(config: Config, save_path: str):
    with open(save_path, 'w') as f:
        json.dump(asdict(config), f, indent=4)
