""" Maps env state to action and provides rules on how to update inner controller """

from abc import ABC
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

from configs.structure import Config
from agent.controllers import CONTROLLER_CLASSES
from intervening.scheduling import Scheduler, SAMPLE, UNIFORM, INVERSE, DETERMINISTIC


def softmax(x: np.array) -> np.array:
    x = np.exp(x - max(x))
    return x / (sum(x) + 1e-9)


class Policy(ABC):
    """ Abstract class """
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        # Scheduler setup
        self.scheduler = Scheduler(config)

        # Controller setup
        self.controller_class = CONTROLLER_CLASSES[config.encoder]
        self.controller = None  # will be set by children

        self.num_actions = num_actions
        self.entropy_coef = self.scheduler.entropy_coef

        # Optimizer setup
        self.max_grad_norm = config.max_grad_norm

    def _build_optimizer(self):
        """ must be set by each child after setting self.controller """
        self.recurrent_controller = self.controller.is_recurrent
        self.all_parameters = list(self.controller.parameters())
        self.optimizer = torch.optim.Adam(self.all_parameters, lr=1)  # will be updated by the lr_scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda _: self.scheduler.lr
        )

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
        actor_logits, value, h, c = self.controller.forward(
            # Set dummy batch sizes of 1
            env_state.view(1, *env_state.shape),
            rec_h,
            rec_c,
        )

        action_distributions = Categorical(logits=actor_logits)  # float tensor of shape [1, num_actions]

        # Take the action provided
        if externally_chosen_action is not None:
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
            None if value is None else value.item(),
            h,
            c,
        )

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
        # Retain the graph so that multiple backward passes can be done through the same hidden recurrent states
        loss.backward(retain_graph=self.recurrent_controller)
        nn.utils.clip_grad_norm_(self.all_parameters, self.max_grad_norm)
        self.optimizer.step()

    def after_iteration(self) -> dict:
        self.scheduler.current_update += 1
        self.lr_scheduler.step(None)
        self.entropy_coef = self.scheduler.entropy_coef
        return self.scheduler.current_values


class PG(Policy):
    """ Policy Gradient - single actor """
    def __init__(self, config: Config, env_state_shape, num_actions):
        super().__init__(config, env_state_shape, num_actions)

        self.controller = self.controller_class(config, env_state_shape, num_actions)
        self._build_optimizer()

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

        self.controller = self.controller_class(config, env_state_shape, num_actions)
        self._build_optimizer()

        self.clip_param = config.ppo_clip
        self.critic_coef = config.ppo_critic_coef

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
            'critic': critic_loss.item(),
            'entropy': entropy_loss.item(),
        }


class SAC(Policy):
    def __init__(self, config: Config, env_state_shape: tuple, num_actions: int):
        self.tau = config.sac_tau
        self.alpha = config.sac_alpha
        self.dynamic_alpha = config.sac_dynamic_alpha
        super().__init__(config, env_state_shape, num_actions)



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

