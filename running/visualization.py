import numpy as np

from typing import List
from torch.utils.tensorboard import SummaryWriter

from agent.policies import Policy

TEAM_NAMES = [
    'Thieves',
    'Guardians'
]


def log_layers(team_policies: List[Policy], writer: SummaryWriter, update_number: int):
    """ Adds histograms of all parameters and their gradients """
    for team_name, policy in zip(TEAM_NAMES, team_policies):
        for layer_name, layer_params in policy.controller.named_parameters():
            writer.add_histogram(f'weights/{team_name}/{layer_name}', layer_params, update_number)
            if layer_params.grad is not None:
                writer.add_histogram(f'grads/policy/{layer_name}', layer_params.grad, update_number)


def log_descriptive_statistics(array: np.array, prefix: str, writer, update_number: int,  axis=0):
    """ Min, max, std avg """
    for op in ['min', 'max', 'mean', 'std']:
        writer.add_scalar(
            prefix + (op if op != 'mean' else 'avg'),
            getattr(array, op)(axis=axis),
            update_number
        )


def log_scalars(total_reward: np.array, steps_alive: np.array, writer: SummaryWriter, update_number: int):
    """
    Args:
        total_reward: float array-like of shape [num_episodes, num_avatars]
        steps_alive: int array-like of shape [num_episodes, num_avatars]
    """
    total_reward = np.asarray(total_reward)
    steps_alive  = np.asarray(steps_alive)

    num_avatars = total_reward.shape[1]

    for name, array in [('total-episode-reward', total_reward), ('episode-steps-alive', steps_alive)]:
        for avatar_id in range(num_avatars):
            log_descriptive_statistics(array[:, avatar_id], f'{name}/avatar-{avatar_id}/', writer, update_number)

