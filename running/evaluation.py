from copy import copy
import torch

from environment.ThievesGuardiansEnv import ThievesGuardiansEnv


def evaluate(config, policy):
    env = ThievesGuardiansEnv(
        config.scenario,
        env_id=0
    )
    env.seed(0)

    # Initialize environment
    observation = env.reset()

    maps = [env._map.copy()]
    pos2ids = [copy(env._pos2id)]

    while True:
        controller_ids = env._controller
        env_state = torch.tensor(observation, dtype=torch.float32)
        rec_state = torch.zeros(env.num_avatars, policy.recurrent_hidden_state_size)
        individual_done = torch.tensor([env._avatar_alive]).transpose(0, 1)
        _, action, _, _ = policy.pick_action(
            controller_ids,
            env_state,
            rec_state,
            individual_done,
            # TODO force deterministic
        )
        action = action.numpy().flatten()
        observation, _, all_done, _ = env.step(action)

        maps.append(env._map.copy())
        pos2ids.append(copy(env._pos2id))

        if all_done:
            break

    return maps, pos2ids
