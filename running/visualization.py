import numpy as np

from typing import List
from torch.utils.tensorboard import SummaryWriter

from agent.policies import Policy

TEAM_NAMES = [
    'Thieves',
    'Guardians'
]

DESCRIPTIVE_STATS = ['mean', 'max']  # any combination of ['min', 'mean', 'std', 'max']


def log_layers(team_policies: List[Policy], writer: SummaryWriter, update_number: int):
    """ Adds histograms of all parameters and their gradients """
    for team_name, policy in zip(TEAM_NAMES, team_policies):
        for layer_name, layer_params in policy.controller.named_parameters():
            writer.add_histogram(f'weights/{team_name}/{layer_name}', layer_params, update_number)
            if layer_params.grad is not None:
                writer.add_histogram(f'grads/policy/{layer_name}', layer_params.grad, update_number)


def log_descriptive_statistics(array: np.array, prefix: str, writer, update_number: int,  axis=0):
    """ Min, max, std avg """
    for op in DESCRIPTIVE_STATS:
        writer.add_scalar(
            prefix + (op if op != 'mean' else 'avg'),
            getattr(array, op)(axis=axis),
            update_number
        )


def log_scalars(training_history: (np.array, np.array, [dict]), writer: SummaryWriter, update_number: int):
    total_reward, steps_alive, team_losses_history = training_history
    total_reward = np.asarray(total_reward)
    steps_alive  = np.asarray(steps_alive)

    num_avatars = total_reward.shape[1]

    for name, array in [('total-episode-reward', total_reward), ('episode-steps-alive', steps_alive)]:
        for avatar_id in range(num_avatars):
            log_descriptive_statistics(array[:, avatar_id], f'{name}/avatar-{avatar_id}/', writer, update_number)

    for team, tlh in enumerate(team_losses_history):
        # From [{actor: a1, critic: c1}, {actor: a2, critic: c2}]
        # to {actor: [a1, a2], critic: [c1, c2]}
        transposed = {
            loss_name: np.zeros(len(tlh))
            for loss_name in tlh[0]
        }
        for i, losses in enumerate(tlh):
            for name, value in losses.items():
                transposed[name][i] = value

        for loss_name, values in transposed.items():
            log_descriptive_statistics(values, f'loss/{TEAM_NAMES[team]}/{loss_name}/', writer, update_number)
