""" Maps env state to action and provides rules on how to update inner controller """

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from configs.structure import Config
from agent.controllers import CONTROLLER_CLASSES


class Policy:
    """ Abstract class """
    def __init__(self, config: Config, env_state_shape, num_actions):
        pass

    def pick_action(self, env_state, rec_h, rec_c, deterministic=False):
        """
        Args:
            env_state:     float array of shape env_state_shape
            rec_h:         float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c:         float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            deterministic: bool -- when True, it always picks the most probable action
        """
        return None, None

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns):
        """
        Args:
            env_states:           float tensor of shape [batch_size, *env_state_shape]
            rec_hs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_cs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            actions:              int tensor of shape  [batch_size,]
            old_action_log_probs: float tensor of shape [batch_size,]
            returns:              float tensor of shape [batch_size,]

        """
        pass


class RandomPolicy(Policy):
    def __init__(self, config: Config, env_state_shape, num_actions):
        self.num_actions = num_actions

    def pick_action(self, *args):
        action = np.random.randint(self.num_actions)
        log_prob = 0.
        return action, log_prob


class LearningPolicy(Policy):
    def __init__(self, config: Config, env_state_shape, num_actions):
        controller_class = CONTROLLER_CLASSES[config.controller]
        self.controller = controller_class(config, env_state_shape, num_actions)
        self.entropy_coef = config.entropy_coef
        self.max_grad_norm = config.max_grad_norm
        self.all_parameters = list(self.controller.parameters())

    def _create_optimizer(self, config: Config):
        self.optimizer = torch.optim.Adam(self.all_parameters, lr=config.lr)
        self.lr_decay = torch.optim.lr_scheduler.StepLR(self.optimizer, config.lr_decay_interval, config.lr_decay_factor)

    def pick_action(self, env_state, rec_h, rec_c, deterministic=False):
        """
        In the given env_state, pick a single action either by sampling or most probable one (when deterministic=True)
        Called during rollouts collection to generate the next action, one by one

        Returns:
            action: int
            action_log_prob: float
            actor_logits: float tensor of shape [num_actions,]
            rec_h: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
        """
        env_state = torch.tensor(env_state, dtype=torch.float32)

        # actor_logits: float tensor of shape [batch_size, num_actions]
        actor_logits, _, h, c = self.controller.forward(
            # Set dummy batch sizes of 1
            env_state.view(1, *env_state.shape),
            rec_h,
            rec_c,
        )

        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [1, num_actions]
        if deterministic:  # pick most probable
            action = action_distributions.probs.argmax()
        else:
            action = action_distributions.sample()
        action_log_probs = action_distributions.log_prob(action)

        return (
            action.item(),
            action_log_probs.item(),
            actor_logits,
            h,
            c,
        )

    def _evaluate_actions(self, env_states, rec_hs, rec_cs, actions):
        """
        See how likely these actions (using the current model) in the given env_states
        Called when updating, on batches of transitions

        Args:
            env_states: float tensor of shape [batch_size, *env_state_shape]
            rec_hs:     float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_cs:     float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            actions:    int tensor of shape  [batch_size,]

        Returns:
            action_log_probs: float tensor of shape [batch_size,]
            entropy:          float tensor of shape [batch_size,]
            values:           float tensor of shape [batch_size,]
        """
        # actor_logits: float tensor of shape [batch_size, num_actions]
        # values: float tensor of shape [batch_size,1]
        actor_logits, values, _, _ = self.controller.forward(env_states, rec_hs, rec_cs)
        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [batch_size, num_actions]

        action_log_probs = action_distributions.log_prob(actions)  # float tensor of shape [batch_size,]
        entropy = action_distributions.entropy()  # float tensor of shape [batch_size,]

        return action_log_probs, entropy, values

    def _optimize(self, loss):
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)  # retain_graph=True so that backward passes thorugh the same hidden recurrent states work
        nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)
        self.optimizer.step()
        self.lr_decay.step(None)


class PG(LearningPolicy):
    """ Policy Gradient - single actor """
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__(config, env_state_shape, num_actions)
        self._create_optimizer(config)

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns) -> {str: float}:
        """
        Increase the probability of actions that give high returns

        Returns:
            {name : loss}
        """
        action_log_probs, entropy, _ = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

        policy_loss = -(action_log_probs * returns).mean()
        entropy_loss = -entropy.mean()
        loss = policy_loss + self.entropy_coef * entropy_loss

        self._optimize(loss)
        return {
            'actor': policy_loss.item(),
            'entropy': entropy_loss.item(),
        }


class PPO(PG):
    """ Proximal Policy Optimization — actor and critic with max bound on update """
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__(config, env_state_shape, num_actions)
        self._create_optimizer(config)
        self.clip_param = config.ppo_clip
        self.critic_coef = config.critic_coef

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns) -> {str: float}:
        """
        Increase the probability of actions that give high advantages
        and move predicted values towards observed returns

        Returns:
            {name : loss}
        """
        action_log_probs, entropy, values = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

        advantage = returns - values.detach()

        ratio = torch.exp(action_log_probs - old_action_log_probs.detach())
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

        actor_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -entropy.mean()
        critic_loss = ((returns - values) ** 2).mean()

        loss = (actor_loss +
                self.critic_coef * critic_loss +
                self.entropy_coef * entropy_loss)

        self._optimize(loss)
        return {
            'actor': actor_loss.item(),
            'critic': actor_loss.item(),
            'entropy': entropy_loss.item(),
        }


POLICY_CLASSES = {
    'pg': PG,
    'ppo': PPO,
    'random': RandomPolicy,
}


# TODO (?) PPO advantage target
# TODO (?) PPO clipped value loss
# def old_update():
#     advantages = returns[:-1] - value_pred[:-1]
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
#
#     for samples in batches:
#         env_state, action, value_pred, return_batch, done, old_action_prob, adv_targ = sample
#
#         values, action_log_prob, dist_entropy, _ = self.policy.evaluate_actions(env_state, done, action)
#
#         ratio = torch.exp(action_log_prob - old_action_prob)
#         surr1 = ratio * adv_targ
#         surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ
#         action_loss = -torch.min(surr1, surr2).mean()
#
#         if use_clipped_value_loss:
#             value_pred_clipped = value_pred + \
#                                  (values - value_pred).clamp(-self.clip_param, self.clip_param)
#             value_losses = (values - return_batch).pow(2)
#             value_losses_clipped = (value_pred_clipped - return_batch).pow(2)
#             value_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()

