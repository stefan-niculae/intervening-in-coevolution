from typing import List
from torch.utils.tensorboard import SummaryWriter

from agent.policies import Policy

TEAM_NAMES = [
    'Thieves',
    'Guardians'
]


def write_logs(team_policies: List[Policy], writer: SummaryWriter, update_number: int):
    log_layers(team_policies, writer, update_number)
    # log_scalars(config, envs, rollouts, episode_number, writer, update_number)


def log_layers(team_policies: List[Policy], writer: SummaryWriter, update_number: int):
    """ Adds histograms of all parameters and their gradients """
    for team_name, policy in zip(TEAM_NAMES, team_policies):
        for layer_name, layer_params in policy.controller.named_parameters():
            writer.add_histogram(f'weights/{team_name}/{layer_name}', layer_params, update_number)
            if layer_params.grad is not None:
                writer.add_histogram(f'grads/policy/{layer_name}', layer_params.grad, update_number)


# def log_scalars(config, envs, rollouts, episode_number, writer, update_number):
#     rewards = rollouts.reward.numpy()
#     teams = rollouts.controller[0, 0]  # assumes all envs have the same team orders
#
#     rows = []
#     for step in range(config.num_transitions - 1):
#         for env_id in range(envs.num_envs):
#             for avatar_id in range(envs.num_avatars):
#                 row = (
#                     env_id,
#                     avatar_id,
#                     episode_number[step, env_id],
#                     rewards[step, env_id, avatar_id, 0],
#                     teams[avatar_id],
#                 )
#                 rows.append(row)
#
#     df = pd.DataFrame(rows, columns=['env_id', 'avatar_id', 'episode', 'reward', 'team'])
#     grouped = df.groupby(['env_id', 'episode', 'team']).reward.sum().groupby('team')
#
#     TEAM_NAMES = {
#         0: 'thieves',
#         1: 'guardians',
#     }
#
#     avg = grouped.mean()
#     std = grouped.std()
#     min = grouped.min()
#     max = grouped.max()
#     for team_id in set(teams):
#         writer.add_scalar(f'reward/avg/{TEAM_NAMES[team_id]}', avg[team_id], update_number)
#         writer.add_scalar(f'reward/std/{TEAM_NAMES[team_id]}', std[team_id], update_number)
#         writer.add_scalar(f'reward/min/{TEAM_NAMES[team_id]}', min[team_id], update_number)
#         writer.add_scalar(f'reward/max/{TEAM_NAMES[team_id]}', max[team_id], update_number)


