import os
from typing import List
from copy import copy
import torch
import numpy as np

from environment.thieves_guardians_env import TGEnv, ACTION_IDX2SYMBOL, DEAD, THIEF, GUARDIAN
from agent.policies import Policy
from running.training import _get_initial_recurrent_state
from intervening.scheduling import SCRIPTED, DETERMINISTIC, SAMPLE
from environment.visualization import create_animation


def simulate_episode(env: TGEnv, team_policies: List[Policy], sampling_method):
    """
    sampling_method: either one int or one for each team
    """
    if type(sampling_method) is int:
        sampling_method = [sampling_method] * env.num_teams

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

    # Set to evaluation mode
    for policy in team_policies:
        policy.controller.eval()

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

                if sampling_method[team] == SCRIPTED:
                    scripted_action = env.scripted_action(avatar_id)
                else:
                    scripted_action = None

                with torch.no_grad():
                    (
                        actions[avatar_id],
                        action_log_probs[avatar_id],
                        _,
                        _,
                        rec_hs[avatar_id],
                        rec_cs[avatar_id],
                    ) = policy.pick_action_and_info(
                        env_states[avatar_id],
                        rec_hs[avatar_id],
                        rec_cs[avatar_id],
                        sampling_method=sampling_method[team],
                        externally_chosen_action=scripted_action,
                    )
            else:
                actions[avatar_id] = DEAD

        # Step the environment with one action for each avatar
        env_states, rewards, dones, infos = env.step(actions)
        cumulative_reward += rewards
        actions_history.append([ACTION_IDX2SYMBOL[env._interpret_action(a, env.id2team[i])]
                                for i, a in enumerate(actions)])

    # Add final state as well
    map_history    .append(env._map.copy())
    pos2id_history .append(copy(env._pos2id))
    rewards_history.append(cumulative_reward.copy())
    actions_history.append([ACTION_IDX2SYMBOL[DEAD]] * env.num_avatars)

    return map_history, pos2id_history, rewards_history, actions_history, infos['end_reason']


def record_match(videos_dir: str, env: TGEnv, all_policies: List, all_names: List[str],
                 selected_thief_name: str, selected_guard_name: str,
                 deterministic=False, n_sampling=2):

    os.makedirs(videos_dir, exist_ok=True)

    if selected_thief_name == 'scripted':
        thief_idx = 0
    else:
        thief_idx = all_names.index(selected_thief_name)

    if selected_guard_name == 'scripted':
        guard_idx = 0
    else:
        guard_idx = all_names.index(selected_guard_name)

    match_name = f'T({selected_thief_name[:3]}) vs G({selected_guard_name[:3]})'

    selected_policies = [
        all_policies[thief_idx][THIEF],
        all_policies[guard_idx][GUARDIAN]
    ]

    def _run_and_save(sampling_method: int, name_suffix: str):
        sampling_method = [sampling_method] * 2
        if selected_thief_name == 'scripted':
            sampling_method[0] = SCRIPTED
        if selected_guard_name == 'scripted':
            sampling_method[1] = SCRIPTED

        [*env_history, _] = simulate_episode(env, selected_policies, sampling_method)
        video_path = f'{videos_dir}/{match_name} {name_suffix}.gif'
        create_animation(env_history, video_path)
        print('Saved', video_path)

    if deterministic:
        _run_and_save(DETERMINISTIC, 'deterministic')
    for i in range(n_sampling):
        _run_and_save(SAMPLE, f'sample {i}')
