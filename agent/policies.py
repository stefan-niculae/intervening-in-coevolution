
import torch
import torch.nn as nn
from gym.spaces import Box, Discrete, MultiBinary

from agent.distributions import Bernoulli, Categorical, DiagGaussian
from agent.controllers import FCController, ConvController


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, num_controllers: int, controller_kind, controller_kwargs=None):
        super().__init__()

        if controller_kind == 'fc':
            controller_class = FCController
        if controller_kind == 'conv':
            controller_class = ConvController

        if controller_kwargs is None:
            controller_kwargs = {}
        # Create identical controllers
        self.controllers = [controller_class(obs_shape[0], **controller_kwargs) for _ in range(num_controllers)]

        if isinstance(action_space, Discrete):
            num_outputs = action_space.n
            dist_class = Categorical
        elif isinstance(action_space, Box):
            num_outputs = action_space.shape[0]
            dist_class = DiagGaussian
        elif isinstance(action_space, MultiBinary):
            num_outputs = action_space.shape[0]
            dist_class = Bernoulli
        else:
            raise NotImplementedError

        self.dist = dist_class(self.controllers[0].output_size, num_outputs)

    @property
    def is_recurrent(self):
        return self.controllers[0].is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """ Size of rnn_hx. """
        return self.controllers[0].recurrent_hidden_state_size

    def forward(self, env_state, rec_state, done):
        raise NotImplementedError

    def _run_controllers(self, controller_ids, env_state, rec_state_input, done):
        """
        Distribute the load to each controller and then merge them back

        :param controller_ids: numpy array of shape [batch_size,]
        :param env_state:  tensor of shape [batch_size, C, W, H]
        :param rec_state_input: tensor of shape [batch_size, env.rec_state_size]
        :param done: tensor of shape [batch_size, 1]

        `batch_size` can also be two dimensional and will be flattened (just the first two dims)

        """
        first_two_dims = None
        batch_size = env_state.size(0)
        if len(env_state.shape) == 5:
            first_two_dims = env_state.shape[:2]
            batch_size       = env_state.size(0) * env_state.size(1)
            # [num_processes, num_avatars, C, W, H] to [num_processes * num_avatars, C, W, H]
            env_state        = env_state      .view(-1, *env_state.shape[2:])
            rec_state_input  = rec_state_input.view(-1, *rec_state_input.shape[2:])
            done             = done           .view(-1, *done.shape[2:])
            controller_ids   = controller_ids.flatten()

        value            = torch.zeros(batch_size, 1)
        actor_features   = torch.zeros(batch_size, self.controllers[0].output_size)
        rec_state_output = torch.zeros(batch_size, 1)
        for id, controller in enumerate(self.controllers):
            controller_mask = (controller_ids == id)
            if sum(controller_mask) == 0:
                continue

            # Act just on the samples meant for this input
            controller_value, \
            controller_actor_features, \
            controller_rec_state \
                = controller(env_state      [controller_mask],
                             rec_state_input[controller_mask],
                             done           [controller_mask])

            # Place the results in the spots for them
            value           [controller_mask] = controller_value
            actor_features  [controller_mask] = controller_actor_features
            rec_state_output[controller_mask] = controller_rec_state

        # Re-form the two dimensional batch shape
        if first_two_dims:
            value            = value           .view(*first_two_dims, *value.shape[1:])
            actor_features   = actor_features  .view(*first_two_dims, *actor_features.shape[1:])
            rec_state_output = rec_state_output.view(*first_two_dims, *rec_state_output.shape[1:])

        return value, actor_features, rec_state_output

    def pick_action(self, controller_ids, env_state, rec_state, done, deterministic=False):
        value, actor_features, rec_state = self._run_controllers(controller_ids, env_state, rec_state, done)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rec_state

    def get_value(self, controller_ids, env_state, rec_state, done):
        value, _, _ = self._run_controllers(controller_ids, env_state, rec_state, done)
        return value

    def evaluate_actions(self, controller_ids, env_state, rec_state, done, action):
        value, actor_features, rec_state = self._run_controllers(controller_ids, env_state, rec_state, done)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rec_state

