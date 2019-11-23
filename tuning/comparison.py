""" Measure the performance of one team against all other teams """

from os import listdir
from pathlib import Path
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from environment.thieves_guardians_env import TGEnv, THIEF, GUARDIAN, WINNING_REASONS
from intervening.scheduling import SAMPLE
from running.evaluation import simulate_episode


def read_models(models_dir='comparison_models'):
    models_dir = Path(models_dir)
    model_files = listdir(models_dir)
    all_policies = [torch.load(models_dir / filename) for filename in model_files]
    return all_policies, model_files


def play_all_pairs(env: TGEnv, all_policies, model_names: [str], num_episodes=10) -> pd.DataFrame:
    thief_policies, guardian_policies = zip(*all_policies)

    n = len(thief_policies)
    rows = []
    for i in range(n):
        for j in range(i, n):
            policies = [
                thief_policies[i],
                guardian_policies[j]
            ]
            for episode_number in range(num_episodes):
                _, _, cumulative_rewards_history, _, end_reason = simulate_episode(env, policies, SAMPLE)

                total_rewards = cumulative_rewards_history[-1]
                thieves_reward   = total_rewards[env.team_masks[THIEF]].sum()
                guardians_reward = total_rewards[env.team_masks[GUARDIAN]].sum()

                for (t, g) in [(i, j), (j, i)]:
                    rows.append((
                        model_names[t],
                        model_names[g],
                        episode_number,
                        thieves_reward,
                        guardians_reward,
                        end_reason == WINNING_REASONS[THIEF],
                        end_reason == WINNING_REASONS[GUARDIAN],
                        end_reason == WINNING_REASONS[None],
                    ))

    return pd.DataFrame(rows, columns=[
        'thieves_model',
        'guardians_model',
        'episode_number',
        'thieves_reward',
        'guardians_reward',
        'thieves_won',
        'guardians_won',
        'tied'
    ], dtype=float)


def compute_team_results(all_results, team: str, measure: str, normalize=False):
    opponents = 'thieves'
    if team == 'thieves':
        opponents = 'guardians'

    team_results = all_results.pivot_table(
        columns=f'{team}_model',
        index=[f'{opponents}_model', 'episode_number'],
        values=f'{team}_{measure}',
    )

    if normalize:
        # Scale by the maximum episode per opponent
        team_results /= team_results.groupby(
            f'{opponents}_model', level=0).max().max(axis=1)

    return team_results


def plot_team_aggregates(all_results: pd.DataFrame, team: str) -> pd.DataFrame.style:
    team_rewards = compute_team_results(all_results, team, 'reward', normalize=True)
    avg_rewards = team_rewards.mean(axis=0)
    winrate     = compute_team_results(all_results, team, 'won').mean(axis=0)  # times won/times played; the average

    team_aggregates = pd.DataFrame([avg_rewards, winrate], index=['avg_rewards', 'winrate']).T\
        .sort_values(by='avg_rewards',ascending=False)\
        .style.bar(color='lightblue').set_precision(2)
    return team_aggregates, team_rewards, avg_rewards.index


def plot_boxplot(team_results: pd.DataFrame, models_order: [str], team: str, figsize=(5, 12)):
    # Sort by average
    team_results = team_results.T.reindex(models_order).T

    plt.figure(figsize=figsize)
    #sns.swarmplot(data=team_results, orient='h', color='.25')
    sns.boxplot(data=team_results, orient='h', color='C0')
    plt.xlabel('Total reward per episode')
    plt.ylabel('Model')
    plt.title(f'{team.title()} Performance')
