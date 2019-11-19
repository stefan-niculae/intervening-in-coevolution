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


def log_scalars(training_history: (np.array, np.array, np.array, [str], [dict], [dict]), writer: SummaryWriter, update_number: int):
    (
        avatar_total_reward,
        avatar_steps_alive,
        avatar_first_probas,
        episode_end_reasons,
        team_losses_history,
        scheduling_statuses,
    ) = training_history

    # Scheduling values, per team
    for team, status in enumerate(scheduling_statuses):
        for var, value in status.items():
            writer.add_scalar(f'scheduling/{TEAM_NAMES[team]}/{var}', value, update_number)

    # Episode reward, descriptive stats per avatar
    num_avatars = avatar_total_reward.shape[1]
    for name, array in [('total-episode-reward', avatar_total_reward), ('episode-steps-alive', avatar_steps_alive)]:
        for avatar_id in range(num_avatars):
            log_descriptive_statistics(array[:, avatar_id], f'{name}/avatar-{avatar_id}/', writer, update_number)

    # End of episode reasons, percentage per iteration
    num_episodes = len(episode_end_reasons)
    for reason in set(episode_end_reasons):
        percentage_of_episodes = episode_end_reasons.count(reason) / num_episodes
        writer.add_scalar(f'end-reason-per/{reason}', percentage_of_episodes, update_number)

    # Probabilities of actions in the first env state, histogram per avatar
    for avatar_id, probas in enumerate(avatar_first_probas.mean(axis=0)):
        writer.add_histogram(f'first-env-state-action-probas/avatar-{avatar_id}', probas, update_number)

    # Losses, descriptive stats and histogram per team
    for team, tlh in enumerate(team_losses_history):
        # Random policies don't have losses to log
        if not tlh:
            continue

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
            writer.add_histogram(f'loss/{TEAM_NAMES[team]}/{loss_name}', values, update_number)
