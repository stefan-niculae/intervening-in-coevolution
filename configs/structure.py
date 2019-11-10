from dataclasses import dataclass


@dataclass
class Config:
    # Check scenarios.py for more
    scenario = '4x4-thief-treasure'

    """ Policy """
    # Only one implemented currently
    algorithm = 'PPO'

    # Update clip param
    ppo_clip = .2

    # Only one implemented currently
    controller = 'conv'

    """ Hardware """
    # Random seed
    seed = 0

    # CPU processes
    num_processes = 4

    # True not tested
    cuda = False

    """ Running """
    # Number of model updates
    num_updates = 100

    # Gather this many transitions before running a model update
    num_transitions = 1000

    # Sample this many times per model update (ppo_epoch)
    num_batches = 5

    # Sample this many per batch
    batch_size = 256

    """ Steering """
    # Discount future rewards (gamma)
    discount = .99

    # Critic
    critic_coef = .5

    # Encourage "exploration"
    entropy_coef = .01

    """ Optimizer """
    # Learning rate
    lr = 7e-4

    # How often to decay the learning rate
    lr_decay_interval = 10

    # How much to decay the learning rate
    lr_decay_factor = .25

    # Adam optimizer parameter
    adam_epsilon = 1e-5

    # Max norm o fthe gradients
    max_grad_norm = .5

    """ Checkpointing """
    # After how many updates to update the progress plot
    log_interval = 10

    # After how many updates to save the model
    save_interval = 10

    # After how many updates to evaluate (run deterministically) and save a video
    eval_interval = 5
