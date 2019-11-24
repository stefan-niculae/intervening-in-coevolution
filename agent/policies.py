""" Maps env state to action and provides rules on how to update inner controller """

from abc import ABC
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from agent.utils import softmax
from configs.structure import Config
from agent.controllers import RecurrentController
from intervening.scheduling import Scheduler, SAMPLE, UNIFORM, INVERSE, DETERMINISTIC


class Policy(ABC):
    """ Abstract class """
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        # Scheduler setup
        # TODO refactor externally?
        self.scheduler = Scheduler(config)

        # Controller setup
        self.controller = RecurrentController(config, env_state_shape, num_actions)

        # Optimizer setup
        self.max_grad_norm = config.max_grad_norm
        self.recurrent_controller = self.controller.is_recurrent
        self.optimizers = {
            component_name: torch.optim.Adam(
                params,
                lr=self.scheduler.get_current_lr(),
                eps=config.adam_epsilon)
            for component_name, params in self._components_params().items()
        }

        # For uniformly random action picking
        self.num_actions = num_actions

    def _components_params(self) -> dict:
        """ One optimizer will be created for each, by default all optimizers """
        return {'all': self.controller.parameters()}

    def _optimize(self, loss: torch.scalar_tensor, component_name='all'):
        optimizer = self.optimizers[component_name]
        optimizer.zero_grad()

        # Retain the graph so that multiple backward passes can be done through the same hidden recurrent states
        loss.backward(retain_graph=self.recurrent_controller)

        # Clip gradients
        params = optimizer.param_groups[0]['params']  # by default pytorch makes one group
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        optimizer.step()

    def pick_action(self, env_state, rec_h, rec_c, sampling_method: int, externally_chosen_action:int = None):
        """
        In the given env_state, pick a single action.
        Called during rollouts collection to generate the next action, one by one

        Args:
            env_state:     float array of shape env_state_shape
            rec_h:         float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c:         float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]

        Returns:
            action: int
            action_log_prob: float
            actor_logits: float tensor of shape [num_actions,]
            value: float
            rec_h: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
            rec_c: float tensor of shape [num_recurrent_layers, 1, recurrent_layer_size]
        """
        env_state = torch.tensor(env_state, dtype=torch.float32)

        # actor_logits: float tensor of shape [batch_size, num_actions]
        encoder_out, h, c = self.controller.encode(
            # Set dummy batch sizes of 1
            env_state.view(1, *env_state.shape),
            rec_h,
            rec_c,
        )

        actor_logits = self.controller.actor(encoder_out)
        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [1, num_actions]

        # Take the action provided
        if externally_chosen_action is not None:
            # TODO refactor externally?
            action = externally_chosen_action

        else:
            # Sample according to the learned distribution
            if sampling_method == SAMPLE:
                action = action_distributions.sample()

            # Pick most probable action
            elif sampling_method == DETERMINISTIC:
                action = action_distributions.probs.argmax()

            # Explore uniformly
            elif sampling_method == UNIFORM:
                action = np.random.choice(self.num_actions)

            # Sample the opposite probabilities
            elif sampling_method == INVERSE:
                p = 1 / softmax(actor_logits.numpy())
                action = np.random.choice(self.num_actions, p=p / sum(p))

        if type(action) is int:
            action = torch.LongTensor([action])

        action_log_prob = action_distributions.log_prob(action)

        return (
            action.item(),
            action_log_prob.item(),
            actor_logits,
            self._compute_state_value(encoder_out),
            h,
            c,
        )

    def _compute_state_value(self, encoder_out) -> Optional[float]:
        return None

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns):
        """
        Args:
            env_states:           float tensor of shape [batch_size, *env_state_shape]
            rec_hs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_cs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            actions:              int   tensor of shape [batch_size,]
            old_action_log_probs: float tensor of shape [batch_size,]
            returns:              float tensor of shape [batch_size,]

        """
        pass

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
            encoder_out: float tensor of shape [batch_size, m]  -- so it's not recomputed again for values
            action_log_probs: float tensor of shape [batch_size,]
            entropy:          float tensor of shape [batch_size,]
        """
        encoder_out, _, _ = self.controller.encode(env_states, rec_hs, rec_cs)
        actor_logits = self.controller.actor(encoder_out)
        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [batch_size, num_actions]

        action_log_probs = action_distributions.log_prob(actions)  # float tensor of shape [batch_size,]
        entropy = action_distributions.entropy()  # float tensor of shape [batch_size,]

        return encoder_out, action_log_probs, entropy

    def after_iteration(self) -> dict:
        self.scheduler.current_update += 1
        # TODO input noise/layers dropout params

        # Set learning rate to the optimizer for each component
        for optimizer in self.optimizers.values():
            optimizer.param_groups[0]['lr'] = self.scheduler.get_current_lr()  # by default pytorch makes one group

        return self.scheduler.current_values


class PG(Policy):
    """ Policy Gradient - single actor """
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__(config, env_state_shape, num_actions)
        self.entropy_coef = self.scheduler.get_current_entropy_coef()

    def after_iteration(self) -> dict:
        self.entropy_coef = self.scheduler.get_current_entropy_coef()
        return super().after_iteration()

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns) -> {str: float}:
        """
        Increase the probability of actions that give high returns

        Returns:
            {name : loss}
        """
        _, action_log_probs, entropy = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

        actor_loss = -(action_log_probs * returns).mean()
        entropy_loss = -entropy.mean()
        loss = actor_loss + self.entropy_coef * entropy_loss

        self._optimize(loss)
        return {
            'actor':     actor_loss.item(),
            'entropy': entropy_loss.item(),
        }


class PPO(PG):
    """ Proximal Policy Optimization — actor and critic with max bound on update """
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__(config, env_state_shape, num_actions)
        self.clip_param = config.ppo_clip
        self.critic_coef = config.ppo_critic_coef

    def _compute_state_value(self, encoder_out) -> Optional[float]:
        return self.controller.critic(encoder_out)

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns) -> {str: float}:
        """
        Increase the probability of actions that give high advantages
        and move predicted values towards observed returns

        Returns:
            {name : loss}
        """
        encoder_out, action_log_probs, entropy = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

        values = self._compute_state_value(encoder_out)
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
            'actor':     actor_loss.item(),
            'critic':   critic_loss.item(),
            'entropy': entropy_loss.item(),
        }


class SAC(Policy):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        super().__init__(config, env_state_shape, num_actions)

        self.tau = config.sac_tau
        self.dynamic_alpha = config.sac_dynamic_entropy
        if self.dynamic_alpha:
            self.target_entropy = -np.log((1. / num_actions)) * .98  # set maximum entropy as the target
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = config.sac_alpha

    def _components_params(self) -> dict:
        component2params = {
            'actor':    self.controller.actor   .parameters(),
            'critic':   self.controller.critic  .parameters(),
            'critic_2': self.controller.critic_2.parameters(),
        }
        if self.dynamic_alpha:
            component2params['alpha'] = [self.log_alpha]
        return component2params

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns):
        env_next_states = env_states
        # Calculate critic loss
        # Ordinary Q-learning loss plus the additional entropy term
        qf1_next_target = self.controller.critic_target(env_next_states)
        qf2_next_target = self.controller.critic_2_target(env_next_states)
        min_qf_next_target = action_probabilities * (
                    torch.min(qf1_next_target, qf2_next_target) - self.alpha * log_action_probabilities)
        min_qf_next_target = min_qf_next_target.mean(dim=1).unsqueeze(-1)
        next_q_value = reward_batch + (1.0 - mask_batch) * self.hyperparameters["discount_rate"] * (min_qf_next_target)


POLICY_CLASSES = {
    'pg':  PG,
    'ppo': PPO,
    'sac': SAC,
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

