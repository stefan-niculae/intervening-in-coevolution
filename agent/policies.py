""" Maps env state to action and provides rules on how to update inner controller """

from abc import ABC
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from agent.utils import softmax, copy_weights, kl_divergence
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
        if config.variational:
            self.variational_coef = self.scheduler.get_current_variational_coef()

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

    def _optimize(self, loss: torch.scalar_tensor, component_name='all', retain_graph=False):
        optimizer = self.optimizers[component_name]
        optimizer.zero_grad()

        # Retain the graph so that multiple backward passes can be done through the same hidden recurrent states
        loss.backward(retain_graph=retain_graph or self.recurrent_controller)

        # Clip gradients
        params = optimizer.param_groups[0]['params']  # by default pytorch makes one group
        nn.utils.clip_grad_norm_(params, self.max_grad_norm)

        optimizer.step()

    def _action_distributions(self, encoder_out):
        actor_logits = self.controller.actor(encoder_out)
        return actor_logits, Categorical(logits=actor_logits)  # float tensor of shape [1, num_actions]

    def _variational_loss(self, means, log_vars) -> (float, dict):
        if self.controller.variational:
            l = self.variational_coef * kl_divergence(means, log_vars)
            return l, {'variational': l}
        else:
            return 0, {}

    def pick_action_and_info(self, env_state, rec_h, rec_c, sampling_method: int, externally_chosen_action:int = None):
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
        _, _, encoder_out, h, c = self.controller.encode(
            # Set dummy batch sizes of 1
            env_state.view(1, *env_state.shape),
            rec_h,
            rec_c,
        )

        actor_logits, action_distributions = self._action_distributions(encoder_out)

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

    def update(self, *args):
        return {}

    def after_iteration(self) -> dict:
        self.scheduler.current_update += 1
        # TODO sync up input noise/layers dropout params

        if self.controller.variational:
            self.variational_coef = self.scheduler.get_current_entropy_coef()

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
        latent_means, latent_log_vars, encoder_out, _, _ = self.controller.encode(env_states, rec_hs, rec_cs)
        actor_logits = self.controller.actor(encoder_out)
        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [batch_size, num_actions]

        action_log_probs = action_distributions.log_prob(actions)  # float tensor of shape [batch_size,]
        entropy = action_distributions.entropy()  # float tensor of shape [batch_size,]

        return latent_means, latent_log_vars, encoder_out, action_log_probs, entropy

    def update(self, env_states, rec_hs, rec_cs, actions, old_action_log_probs, returns) -> {str: float}:
        """
        Increase the probability of actions that give high returns

        Args
            env_states:           float tensor of shape [batch_size, *env_state_shape]
            rec_hs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_cs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            actions:              int   tensor of shape [batch_size,]
            old_action_log_probs: float tensor of shape [batch_size,]
            returns:              float tensor of shape [batch_size,]
        """
        latent_means, latent_log_vars, _, action_log_probs, entropy = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

        actor_loss = -(action_log_probs * returns).mean()
        entropy_loss = -entropy.mean()
        loss = (actor_loss +
                entropy_loss * self.entropy_coef)

        variational_loss, extra_losses = self._variational_loss(latent_means, latent_log_vars)
        loss += variational_loss

        self._optimize(loss)
        return {
            'actor': actor_loss.item(),
            'entropy': entropy_loss.item(),
            **extra_losses,
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

        Args
            env_states:           float tensor of shape [batch_size, *env_state_shape]
            rec_hs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            rec_cs:               float tensor of shape [num_recurrent_layers, batch_size, recurrent_layer_size]
            actions:              int   tensor of shape [batch_size,]
            old_action_log_probs: float tensor of shape [batch_size,]
            returns:              float tensor of shape [batch_size,]
        """
        latent_means, latent_log_vars, encoder_out, action_log_probs, entropy = self._evaluate_actions(env_states, rec_hs, rec_cs, actions)

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

        variational_loss, extra_losses = self._variational_loss(latent_means, latent_log_vars)
        loss += variational_loss

        self._optimize(loss)
        return {
            'actor':     actor_loss.item(),
            'critic':   critic_loss.item(),
            'entropy': entropy_loss.item(),
            **extra_losses,
        }


class SAC(Policy):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        self.dynamic_alpha = config.sac_dynamic_entropy
        super().__init__(config, env_state_shape, num_actions)

        self.discount = config.discount
        self.tau = config.sac_tau
        if self.dynamic_alpha:
            self.target_entropy = -np.log((1. / num_actions)) * .98  # set maximum entropy as the target
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha = config.sac_alpha

    def _components_params(self) -> dict:
        component2params = {
            'actor':    self.controller.actor   .parameters(),
            'critic_1': self.controller.critic_1.parameters(),
            'critic_2': self.controller.critic_2.parameters(),
        }
        if self.dynamic_alpha:
            component2params['alpha'] = [self.log_alpha]
        return component2params

    def _action_distributions(self, encoder_out):
        action_probas = self.controller.actor(encoder_out)
        return action_probas, Categorical(action_probas)  # note: no logits

    def _pick_batch_actions_and_infos(self, env_states, rec_hs_inp, rec_cs_inp):
        latent_means, latent_log_vars, encoder_out, rec_hs_out, rec_cs_out = self.controller.encode(env_states, rec_hs_inp, rec_cs_inp)

        action_probs, action_distributions = self._action_distributions(encoder_out)
        action_log_probs = torch.log(action_probs + 1e-10)

        actions = action_distributions.sample()

        return encoder_out, rec_hs_out, rec_cs_inp, actions, action_probs, action_log_probs

    def _estimate_future_value(self, env_states, rec_hs, rec_cs):
        """ Observed reward + discounted estimated future value, using the target networks """
        with torch.no_grad():
            # TODO multistep
            encoder_out, rec_hs, rec_cs, actions, action_probs, action_log_probs = self._pick_batch_actions_and_infos(
                env_states, rec_hs, rec_cs)

            action_values_1 = self.controller.critic_1_target(encoder_out)
            action_values_2 = self.controller.critic_2_target(encoder_out)
            action_values_min = torch.min(action_values_1, action_values_2)
            all_action_values = action_probs * (action_values_min - self.alpha * action_log_probs)
            avg_values = all_action_values.mean(dim=1)

        return avg_values

    def update(self, env_states, rec_hs, rec_cs, actions, rewards, dones):
        """
        Args:
            env_states: [2, batch_size, *env_state_shape]
                env_states[0] = current_env_state
                env_states[1] = next_env_state
            rec_hs: [batch_size, rec_size]
            rec_cs: [batch_size, rec_size]
            actions: int [batch_size,]
            dones: float [batch_size]
            rewards: [batch_size]
        Returns:

        """
        # Critic
        latent_means, latent_log_vars, encoder_out, _, _ = self.controller.encode(env_states[0], rec_hs, rec_cs)
        all_actions_values_1 = self.controller.critic_1(encoder_out)
        all_actions_values_2 = self.controller.critic_2(encoder_out)
        all_actions_values_min = torch.min(all_actions_values_1, all_actions_values_2)

        chosen_actions_value_1 = all_actions_values_1.take(actions)
        chosen_actions_value_2 = all_actions_values_2.take(actions)

        future_values = rewards + (1 - dones) * self.discount * self._estimate_future_value(env_states[1], rec_hs, rec_cs)
        critic_1_loss = F.mse_loss(chosen_actions_value_1, future_values)
        critic_2_loss = F.mse_loss(chosen_actions_value_2, future_values)

        # Actor
        _, _, _, _, action_probs, action_log_probs = self._pick_batch_actions_and_infos(env_states[0], rec_hs, rec_cs)
        inside_term = self.alpha * action_log_probs - all_actions_values_min
        actor_loss = (action_probs * inside_term).mean()

        # Perform optimization steps
        self._optimize(critic_1_loss, 'critic_1', retain_graph=True)
        self._optimize(critic_2_loss, 'critic_2', retain_graph=True)
        self._optimize(actor_loss,    'actor')

        # Perform soft updates
        copy_weights(self.controller.critic_1, self.controller.critic_1_target, self.tau)
        copy_weights(self.controller.critic_2, self.controller.critic_2_target, self.tau)

        losses = {
            'actor':    actor_loss.item(),
            'critic_1': critic_1_loss.item(),
            'critic_2': critic_2_loss.item(),
        }

        if self.dynamic_alpha:
            s = torch.sum(action_log_probs * action_probs, dim=1)
            alpha_loss = -(self.log_alpha * (s + self.target_entropy).detach()).mean()
            self._optimize(alpha_loss, 'alpha')
            losses['alpha'] = alpha_loss

        return losses


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

