""" Routing avatars to storages and policies  """

from typing import List
import numpy as np

from configs.structure import Config
from environment.thieves_guardians_env import TGEnv
from agent.policies import POLICY_CLASSES, Policy
from agent.utils import softmax
from agent.storage import RolloutsStorage, OnPolicyStorage, CyclicStorage
from running.utils import EpisodeAccumulator
from intervening.scheduling import SCRIPTED


def instantiate(config: Config) -> (TGEnv, List[Policy], List[RolloutsStorage]):
    """ Instantiate the environment, agents and storages """
    env = TGEnv(config)

    # Each avatar has its own storage (because they do not all die at the same time)
    storage_class = CyclicStorage if config.algorithm == 'sac' else OnPolicyStorage
    avatar_storages = [
        storage_class(config, env.state_shape)
        for _ in range(env.num_avatars)
    ]

    # Each team has its own copy of the policy
    policy_class = POLICY_CLASSES[config.algorithm]
    team_policies = [
        policy_class(config, env.state_shape, env.num_actions[team])
        for team in range(env.num_teams)
    ]

    return env, team_policies, avatar_storages


def _get_initial_recurrent_state(avatar_policies):
    """
    Gets rec_h0 and rec_c0 if the policy has a controller

    Args:
        avatar_policies: list of Policy objects, length env.num_avatars

    Returns:
        rec_hs: list of float tensor of shape [num_recurrent_layers, 1, recurrent_size], length env.num_avatars
        rec_cs: list of float tensor of shape [num_recurrent_layers, 1, recurrent_size], length env.num_avatars
            or None if the policy is not
    """
    rec_hs = [None] * len(avatar_policies)
    rec_cs = [None] * len(avatar_policies)
    for i, policy in enumerate(avatar_policies):
        if policy.controller.is_recurrent:
            rec_hs[i] = policy.controller.rec_h0
            rec_cs[i] = policy.controller.rec_c0
    return rec_hs, rec_cs


def perform_update(config, env: TGEnv, team_policies: List[Policy], avatar_storages: List[RolloutsStorage]):
    """ Collects rollouts and updates """

    # Used to log
    total_rewards     = EpisodeAccumulator(env.num_avatars)
    steps_alive       = EpisodeAccumulator(env.num_avatars)
    first_step_probas = EpisodeAccumulator(env.num_avatars, max(env.num_actions))
    end_reasons = []

    # Will be filled in for each avatar when stepping the environment individually
    actions          = [0] * env.num_avatars
    action_log_probs = [0] * env.num_avatars
    values           = [0] * env.num_avatars

    avatar_policies = [
        team_policies[env.id2team[avatar_id]]
        for avatar_id in range(env.num_avatars)
    ]

    # Always start with a fresh env
    env_states = env.reset()  # shape: [num_avatars, *env_state_shape]
    rec_hs, rec_cs = _get_initial_recurrent_state(avatar_policies)
    next_rec_hs, next_rec_cs = rec_hs, rec_cs
    first_episode_step = True

    # Set to training mode
    for policy in team_policies:
        policy.controller.train()

    # Collect rollouts
    # TODO (?): collect in parallel
    for step in range(config.num_transitions):
        # Alive at the beginning of step
        avatar_alive = env.avatar_alive.copy()

        # Run each alive avatar individually
        for avatar_id in range(env.num_avatars):
            if avatar_alive[avatar_id]:
                # Chose action based on the policy
                team = env.id2team[avatar_id]
                policy = team_policies[team]

                action_source = policy.scheduler.pick_action_source()
                if action_source == SCRIPTED:
                    scripted_action = env.scripted_action(avatar_id)
                else:
                    scripted_action = None

                (
                    actions[avatar_id],
                    action_log_probs[avatar_id],
                    actor_logits,
                    values[avatar_id],
                    next_rec_hs[avatar_id],
                    next_rec_cs[avatar_id],
                ) = policy.pick_action_and_info(
                    env_states[avatar_id],
                    rec_hs[avatar_id],
                    rec_cs[avatar_id],
                    sampling_method=action_source,
                    externally_chosen_action=scripted_action,
                )

                if first_episode_step:
                    probas = softmax(actor_logits.detach().numpy().flatten())
                    first_step_probas.current[avatar_id, :len(probas)] = probas

        # Step the environment with one action for each avatar
        next_env_states, rewards, dones, info = env.step(actions)

        # Insert transitions for alive avatars
        for avatar_id in range(env.num_avatars):
            if avatar_alive[avatar_id]:
                storage = avatar_storages[avatar_id]
                storage.insert(
                    env_states[avatar_id],
                    actions[avatar_id],
                    action_log_probs[avatar_id],
                    values[avatar_id],
                    rewards[avatar_id],
                    dones[avatar_id],
                    rec_hs[avatar_id],
                    rec_cs[avatar_id],
                )

        total_rewards.current += rewards
        steps_alive.current   += avatar_alive

        # Episode is done
        if all(dones):
            env_states = env.reset()
            rec_hs, rec_cs = _get_initial_recurrent_state(avatar_policies)

            total_rewards    .episode_over()
            steps_alive      .episode_over()
            first_step_probas.episode_over()
            end_reasons.append(info['end_reason'])
            first_episode_step = True

        # The states were not immediately overwritten because we store the state that was used to generate (env_states)
        # the action for the current time-step, not the one we arrive in (next_env_states)
        else:
            env_states = next_env_states
            rec_hs = next_rec_hs
            rec_cs = next_rec_cs

            first_episode_step = False

    # Compute returns for all storages
    for storage in avatar_storages:
        storage.compute_returns()

    # Report progress
    avatar_rewards = total_rewards.final_history(drop_last=True)
    avg_team_rewards = np.array([
        avatar_rewards[:, mask].sum()  / sum(mask)  # take average per avatar
        for mask in env.team_masks
    ])  # shape [num_teams,] holds the average reward of all thieves thieves got and the average of all guardians
    relative_team_rewards = avg_team_rewards / avg_team_rewards.sum()
    for measure, policy in zip(relative_team_rewards, team_policies):
        policy.scheduler.end_iteration_report(measure)

    # Update policies
    losses_history = [[] for _ in range(env.num_teams)]
    for epoch in range(config.num_epochs):
        for avatar_id in range(env.num_avatars):
            team = env.id2team[avatar_id]
            policy = team_policies[team]
            storage = avatar_storages[avatar_id]

            for batch in storage.sample_batches():
                # A batch contains multidimensional env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns
                losses = policy.update(*batch)
                losses_history[team].append(losses)

    # Prepare storages for the next update
    for storage in avatar_storages:
        storage.reset()

    scheduling_statuses = [policy.sync_scheduled_values() for policy in team_policies]

    # Ignore last episode since it's most likely unfinished
    return (
        total_rewards.final_history(drop_last=True),
        steps_alive.final_history(drop_last=True),
        first_step_probas.final_history(drop_last=False),
        end_reasons,
        losses_history,
        scheduling_statuses,
    )
