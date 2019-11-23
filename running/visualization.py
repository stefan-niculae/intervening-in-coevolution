from time import time
from dataclasses import asdict
import numpy as np

from typing import List
from torch.utils.tensorboard import SummaryWriter

from agent.policies import Policy
from environment.thieves_guardians_env import TEAM_NAMES, TGEnv
from configs.structure import Config


DESCRIPTIVE_STATS = ['mean', 'max', 'std']  # any combination of ['min', 'mean', 'std', 'max']


def log_layers(team_policies: List[Policy], writer: SummaryWriter, update_number: int):
    """ Adds histograms of all parameters and their gradients """
    for team_name, policy in zip(TEAM_NAMES, team_policies):
        for layer_name, layer_params in policy.controller.named_parameters():
            writer.add_histogram(f'weights/{team_name}s/{layer_name}', layer_params, update_number)
            if layer_params.grad is not None:
                writer.add_histogram(f'grads/policy/{layer_name}', layer_params.grad, update_number)


def log_descriptive_statistics(prefix: str, array: np.array, update_number: int, writer, axis=0):
    """ Min, max, std avg """
    for op in DESCRIPTIVE_STATS:
        writer.add_scalar(
            prefix + (op if op != 'mean' else 'avg'),
            getattr(array, op)(axis=axis),
            update_number
        )


def log_scalars(training_history: (np.array, np.array, np.array, [str], [dict], [dict]), writer: SummaryWriter, update_number: int, env: TGEnv):
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
            team = env.id2team[avatar_id]
            log_descriptive_statistics(f'{name}/{TEAM_NAMES[team]}-{avatar_id}/', array[:, avatar_id], update_number, writer)

    # Aggregates of rewards and episode length per team
    for team_name, team_mask in zip(TEAM_NAMES, env.team_masks):
        log_descriptive_statistics(f'total-episode-reward/{team_name}s-sum/', avatar_total_reward[:, team_mask].sum(axis=1),
                                   update_number, writer)
        log_descriptive_statistics(f'episode-steps-alive/{team_name}s-average/', avatar_steps_alive[:, team_mask].mean(axis=1),
                                   update_number, writer)

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
            log_descriptive_statistics(f'loss/{TEAM_NAMES[team]}/{loss_name}/', values, update_number, writer)
            writer.add_histogram(f'loss/{TEAM_NAMES[team]}/{loss_name}', values, update_number)


def log_comparisons(won_statuses: [[bool], [bool]], team_rewards: [[float], [float]], writer: SummaryWriter, update_number: int):
    for won, rew, team_name in zip(won_statuses, team_rewards, TEAM_NAMES):
        writer.add_scalar         (f'external_comparisons/{team_name}/winrate',  won.mean(), update_number)
        log_descriptive_statistics(f'external_comparisons/{team_name}/rewards/', rew,        update_number, writer)
        writer.add_histogram      (f'external_comparisons/{team_name}/rewards',  rew,        update_number)


def log_hyperparams_and_metrics(config: Config,
                                selves_wons: [[bool], [bool]], selves_rewards: [[float], [float]],
                                others_wons: [[bool], [bool]], others_rewards: [[float], [float]],
                                writer: SummaryWriter, start_time: float):

    metrics = {
        'run_time': time() - start_time
    }
    for team_idx, team_name in enumerate(TEAM_NAMES):
        metrics    [f'comparison/{team_name}s-winrate-against_training_opponents'] = selves_wons[team_idx].mean()
        if others_wons is not None:
            metrics[f'comparison/{team_name}s-winrate-against_external_opponents'] = others_wons[team_idx].mean()

        for op in DESCRIPTIVE_STATS:
            metrics[f'comparison/{team_name}s-reward_{op}-against_training_opponents'] = getattr(selves_rewards, op)()
            if others_rewards is not None:
                metrics[f'comparison/{team_name}s-winrate-against_external_opponents'] = getattr(others_rewards, op)()

    # Tensorboard only accepts int, float, str, bool, or torch.Tensor
    # So we turn lists into their string representation, rightfully treating them as categorical values (if more than one item)
    hyperparams = asdict(config)
    for k, v in hyperparams.items():
        if type(v) not in [int, float, str, bool]:
            if len(v) == 1:
                hyperparams[k] = v[0]
            else:
                hyperparams[k] = str(v)

    writer.add_hparams(hyperparams, metrics)
