from typing import List
from copy import copy

from environment.thieves_guardians_env import TGEnv
from agent.policies import Policy


def evaluate(env: TGEnv, team_policies: List[Policy]):
    maps    = []
    pos2ids = []

    env_states = env.reset()
    dones = [False] * env.num_avatars

    actions = [0] * env.num_avatars
    action_log_probs = [0] * env.num_avatars

    while not all(dones):
        maps.append(env._map.copy())
        pos2ids.append(copy(env._pos2id))

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

    maps.append(env._map.copy())
    pos2ids.append(copy(env._pos2id))

    return maps, pos2ids
