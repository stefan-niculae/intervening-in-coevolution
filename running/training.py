import torch

from environment.vec_env import make_vec_envs
from agent.policies import Policy
from agent.algorithms import PPO, A2C_ACKTR
from agent.storage import RolloutStorage


def instantiate(args, device):
    """ Instantiate the environment, agent, storage """
    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space,
        base_kind=args.policy_base,
        base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic.to(device)

    if args.algo == 'a2c':
        agent = A2C_ACKTR(
            actor_critic,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.optim_eps,
            alpha=args.rmsprop_alpha,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'ppo':
        agent = PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.lr,
            eps=args.optim_eps,
            max_grad_norm=args.max_grad_norm)
    elif args.algo == 'acktr':
        agent = A2C_ACKTR(
            actor_critic, args.value_loss_coef, args.entropy_coef, acktr=True)

    rollouts = RolloutStorage(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)

    return envs, actor_critic, agent, rollouts


def perform_update(args, envs, actor_critic, agent, rollouts, update_number, n_updates, episode_rewards):
    """ Runs the agent on the env and updates the model """
    if args.use_linear_lr_decay:
        # decrease learning rate linearly
        update_linear_schedule(agent.optimizer, update_number, n_updates,
            agent.optimizer.lr if args.algo == "acktr" else args.lr)

    for step in range(args.num_steps):
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(rollouts.obs[step],
                rollouts.recurrent_hidden_states[step], rollouts.masks[step])

        # Obser reward and next obs
        obs, reward, done, infos = envs.step(action)

        for info in infos:
            if 'episode' in info.keys():
                episode_rewards.append(info['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
        bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
        rollouts.insert(obs, recurrent_hidden_states, action, action_log_prob, value, reward, masks, bad_masks)

    with torch.no_grad():
        next_value = actor_critic.get_value(rollouts.obs[-1], rollouts.recurrent_hidden_states[-1], rollouts.masks[-1]).detach()

    rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits)

    value_loss, action_loss, dist_entropy = agent.update(rollouts)

    rollouts.after_update()

    return value_loss, action_loss, dist_entropy


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
