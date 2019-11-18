import numpy as np

from typing import List
from copy import copy

from environment.thieves_guardians_env import TGEnv, ACTION_IDX2SYMBOL, DEAD
from agent.policies import Policy
from running.training import _get_initial_recurrent_state


def evaluate(env: TGEnv, team_policies: List[Policy], deterministic: bool):
    map_history     = []
    pos2id_history  = []
    rewards_history = []
    actions_history = []

    avatar_policies = [
        team_policies[env.id2team[avatar_id]]
        for avatar_id in range(env.num_avatars)
    ]

    env_states = env.reset()  # shape: [num_avatars, *env_state_shape]
    rec_hs, rec_cs = _get_initial_recurrent_state(avatar_policies)
    dones = [False] * env.num_avatars
    cumulative_reward = np.zeros(env.num_avatars)

    actions = [0] * env.num_avatars
    action_log_probs = [0] * env.num_avatars

    while not all(dones):
        map_history.append(env._map.copy())
        pos2id_history.append(copy(env._pos2id))
        rewards_history.append(cumulative_reward.copy())

        # Alive at the beginning of step
        avatar_alive = env.avatar_alive.copy()

        # Run each alive avatar individually
        for avatar_id in range(env.num_avatars):
            if avatar_alive[avatar_id]:
                # Chose action based on the policy
                team = env.id2team[avatar_id]
                policy = team_policies[team]

                (
                    actions[avatar_id],
                    action_log_probs[avatar_id],
                    _,
                    rec_hs[avatar_id],
                    rec_cs[avatar_id],
                ) = policy.pick_action(
                    env_states[avatar_id],
                    rec_hs[avatar_id],
                    rec_cs[avatar_id],
                    explore_proba=0,
                    deterministic=deterministic,
                )
            else:
                actions[avatar_id] = DEAD

        # Step the environment with one action for each avatar
        env_states, rewards, dones, infos = env.step(actions)
        cumulative_reward += rewards
        actions_history.append([ACTION_IDX2SYMBOL[a] for a in actions])

    # Add final state as well
    map_history    .append(env._map.copy())
    pos2id_history .append(copy(env._pos2id))
    rewards_history.append(cumulative_reward.copy())
    actions_history.append([ACTION_IDX2SYMBOL[DEAD]] * env.num_avatars)

    return map_history, pos2id_history, rewards_history, actions_history
