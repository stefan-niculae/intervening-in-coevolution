""" Decoupled steps (initialize and update) of the learning process  """

import numpy as np
import torch

from environment.parallelization import make_vec_envs, VecEnv
from agent.policies import Policy
from agent.algorithms import PPO
from agent.storage import RolloutStorage


def instantiate(config, device):
    """ Instantiate the environment, policy, agent, storage """
    envs = make_vec_envs(config, device)

    policy = Policy(
        envs.observation_space.shape,
        envs.action_space,
        num_controllers=2,  # TODO don't hardcode this
        controller_kind=config.controller,
        controller_kwargs={'is_recurrent': False})
    policy.to(device)

    if config.algorithm != 'PPO':
        raise NotImplemented
    agent = PPO(policy, config)

    rollouts = RolloutStorage(config,
        envs.num_avatars, envs.observation_space.shape, envs.action_space,
        policy.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.env_state[0].copy_(obs)

    lr_decay = torch.optim.lr_scheduler.StepLR(
        agent.optimizer, config.lr_decay_interval, config.lr_decay_factor)

    return envs, policy, agent, rollouts, lr_decay


def perform_update(config, envs: VecEnv, policy: Policy, agent: PPO, rollouts: RolloutStorage):
    """ Runs the agent on the env and updates the model """

    episode_number_history = np.zeros((config.num_transitions - 1, config.num_processes))
    current_episode_number = np.zeros(config.num_processes)

    for step in range(config.num_transitions - 1):
        # Sample actions
        with torch.no_grad():
            value_pred, action, action_prob, rec_state = policy.pick_action(
                rollouts.controller[step],
                rollouts.env_state[step],
                rollouts.rec_state[step],
                rollouts.done[step])

        # Simulate the environment
        env_state, reward, all_done, infos = envs.step(action)
        # if(len(reward.shape)==2):
        #     reward = reward[]
        reward = torch.transpose(reward, 1, 2)

        # Gather extra return values from all processes
        individual_done = torch.FloatTensor([i['individual_done'] for i in infos])
        individual_done.unsqueeze_(-1)
        controller = np.array([i['controller'] for i in infos])

        rollouts.insert(env_state, rec_state, action, action_prob, value_pred, reward, individual_done, controller)

        episode_number_history[step] = current_episode_number.copy()
        # When one of the environments is done, increment its run number
        current_episode_number += all_done

    # Estimate value of env state we arrived in
    with torch.no_grad():
        next_value = policy.get_value(
            rollouts.controller[-1],
            rollouts.env_state[-1],
            rollouts.rec_state[-1],
            rollouts.done[-1]).detach()
    rollouts.compute_returns(next_value)

    value_loss, action_loss, dist_entropy = agent.update(rollouts)

    return value_loss, action_loss, dist_entropy, episode_number_history

