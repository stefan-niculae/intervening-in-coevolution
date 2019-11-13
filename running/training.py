""" Routing avatars to storages and policies  """

from typing import List
import numpy as np

from configs.structure import Config
from environment.thieves_guardians_env import TGEnv
from agent.policies import POLICY_CLASSES, Policy
from agent.storage import RolloutStorage


def instantiate(config: Config) -> (TGEnv, List[Policy], List[RolloutStorage]):
    """ Instantiate the environment, agents and storages """
    env = TGEnv(config.scenario)

    # Each avatar has its own storage (because they do not all die at the same time)
    avatar_storages = [
        RolloutStorage(config, env.state_shape)
        for _ in range(env.num_avatars)
    ]

    # Each team has its own policy
    policy_class = POLICY_CLASSES[config.algorithm]
    team_policies = [
        policy_class(config, env.state_shape, env.num_actions)
        for _ in range(env.num_teams)
    ]

    return env, team_policies, avatar_storages


def perform_update(config, env: TGEnv, team_policies: List[Policy], avatar_storages: List[RolloutStorage]):
    """ Collects rollouts and updates """

    # Always start with a fresh env
    env_states = env.reset()
    dones = [False] * env.num_avatars

    actions          = [0] * env.num_avatars
    action_log_probs = [0] * env.num_avatars

    episode_rewards = np.zeros(env.num_avatars)

    # Collect rollouts
    # TODO (?): collect in parallel
    for step in range(config.num_transitions):
        if all(dones):
            env_states = env.reset()
            print('total rewards (per avatar) this episode', episode_rewards)

        # Alive at the beginning of step
        avatar_alive = env.avatar_alive.copy()

        # Run each alive avatar individually
        for avatar_id in range(env.num_avatars):
            if avatar_alive[avatar_id]:
                # Chose action based on the policy
                team = env.id2team[avatar_id]
                policy = team_policies[team]
                actions[avatar_id], action_log_probs[avatar_id] = policy.pick_action(env_states[avatar_id])

        # Step the environment with one action for each avatar
        env_states, rewards, dones, infos = env.step(actions)

        episode_rewards += rewards

        # Insert transitions for alive avatars
        for avatar_id in range(env.num_avatars):
            if avatar_alive[avatar_id]:
                storage = avatar_storages[avatar_id]
                storage.insert(
                    env_states[avatar_id],
                    actions[avatar_id],
                    action_log_probs[avatar_id],
                    rewards[avatar_id],
                    dones[avatar_id],
                )

    # Compute returns for all storages
    for storage in avatar_storages:
        storage.compute_returns()

    # Update policies
    for epoch in range(config.num_epochs):
        for avatar_id in range(env.num_avatars):
            team = env.id2team[avatar_id]
            policy = team_policies[team]
            storage = avatar_storages[avatar_id]

            for batch in storage.sample_batches():
                policy.update(*batch)

    # Prepare storages for the next update
    for storage in avatar_storages:
        storage.reset()
