import torch
import torch.nn as nn
import torch.optim as optim


class PPO:
    def __init__(self, policy, config, use_clipped_value_loss=True):
        self.policy = policy  # actor critic

        self.num_batches = config.num_batches
        self.batch_size = config.batch_size
        self.clip_param = config.ppo_clip
        self.critic_coef = config.critic_coef
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # TODO is this right? a single optimizer?
        self.optimizer = optim.Adam(
            sum([list(c.parameters()) for c in self.policy.controllers], []),
            lr=config.lr,
            eps=config.adam_epsilon,
        )

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_pred[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)

        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.num_batches):
            if self.policy.is_recurrent:
                data_generator = rollouts.recurrent_generator   (advantages)
            else:
                data_generator = rollouts.feed_forward_generator(advantages)

            for sample in data_generator:
                controller_id, env_state, rec_state, action, value_pred, \
                    return_batch, done, old_action_prob, adv_targ = sample

                # Reshape to do in a single forward pass for all steps
                values, action_log_prob, dist_entropy, _ = self.policy.evaluate_actions(
                    controller_id, env_state, rec_state, done, action)

                ratio = torch.exp(action_log_prob - old_action_prob)
                surr1 = ratio * adv_targ
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
                action_loss = -torch.min(surr1, surr2).mean()

                if self.use_clipped_value_loss:
                    value_pred_clipped = value_pred + \
                                         (values - value_pred).clamp(-self.clip_param, self.clip_param)
                    value_losses = (values - return_batch).pow(2)
                    value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
                    value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = 0.5 * (return_batch - values).pow(2).mean()

                self.optimizer.zero_grad()
                (value_loss * self.critic_coef + action_loss -
                 dist_entropy * self.entropy_coef).backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(),
                                         self.max_grad_norm)
                self.optimizer.step()

                value_loss_epoch   += value_loss.item()
                action_loss_epoch  += action_loss.item()
                dist_entropy_epoch += dist_entropy.item()

        num_updates = self.num_batches * self.batch_size

        value_loss_epoch   /= num_updates
        action_loss_epoch  /= num_updates
        dist_entropy_epoch /= num_updates

        return value_loss_epoch, action_loss_epoch, dist_entropy_epoch
