import pandas as pd


def log_weights(policy, writer, update_number):
    """ Adds histograms of all parameters and their gradients """
    for name, param in policy.named_parameters():
        writer.add_histogram(f'weights/policy/{name}', param, update_number)
        if param.grad is not None:
            writer.add_histogram(f'grads/policy/{name}', param.grad, update_number)
    for controller_id, controller in enumerate(policy.controllers):
        for name, param in controller.named_parameters():
            writer.add_histogram(f'weights/controller-{controller_id}/{name}', param, update_number)
            if param.grad is not None:
                writer.add_histogram(f'grads/controller-{controller_id}/{name}', param.grad, update_number)


def log_scalars(config, envs, rollouts, episode_number, writer, update_number):
    rewards = rollouts.reward.numpy()
    teams = rollouts.controller[0, 0]  # assumes all envs have the same team orders

    rows = []
    for step in range(config.num_transitions - 1):
        for env_id in range(envs.num_envs):
            for avatar_id in range(envs.num_avatars):
                row = (
                    env_id,
                    avatar_id,
                    episode_number[step, env_id],
                    rewards[step, env_id, avatar_id, 0],
                    teams[avatar_id],
                )
                rows.append(row)

    df = pd.DataFrame(rows, columns=['env_id', 'avatar_id', 'episode', 'reward', 'team'])
    grouped = df.groupby(['env_id', 'episode', 'team']).reward.sum().groupby('team')

    TEAM_NAMES = {
        0: 'thieves',
        1: 'guardians',
    }

    avg = grouped.mean()
    std = grouped.std()
    min = grouped.min()
    max = grouped.max()
    for team_id in set(teams):
        writer.add_scalar(f'reward/avg/{TEAM_NAMES[team_id]}', avg[team_id], update_number)
        writer.add_scalar(f'reward/std/{TEAM_NAMES[team_id]}', std[team_id], update_number)
        writer.add_scalar(f'reward/min/{TEAM_NAMES[team_id]}', min[team_id], update_number)
        writer.add_scalar(f'reward/max/{TEAM_NAMES[team_id]}', max[team_id], update_number)


def write_logs(config, envs, policy, rollouts, episode_number, writer, update_number):
    log_scalars(config, envs, rollouts, episode_number, writer, update_number)
    log_weights(policy, writer, update_number)
