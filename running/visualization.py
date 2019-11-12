import pandas as pd


def write_summary(config, envs, rollouts, episode_number, writer, update_number):
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
    df.to_csv('df.csv', index=False)
    team_rewards = df.groupby(['env_id', 'episode', 'team']).reward.sum().groupby('team').mean()

    TEAM_NAMES = {
        0: 'thieves',
        1: 'guardians',
    }

    for team_id in set(teams):
        print('writing', f'reward/{TEAM_NAMES[team_id]}', 'reward:', team_rewards[team_id], 'step number:', update_number)
        writer.add_scalar(f'reward/{TEAM_NAMES[team_id]}', team_rewards[team_id], update_number)
