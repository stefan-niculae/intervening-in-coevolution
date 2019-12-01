from typing import List
import numpy as np

from configs.structure import Config

SAMPLE        = 0  # according to policy
UNIFORM       = 1  # disregard policy, pick at random
SCRIPTED      = 2  # externally picked action
INVERSE       = 3  # sampling according to the exact opposite policy
DETERMINISTIC = 4  # most probable action in the policy

action_source_names = {
    SAMPLE: 'sample',
    UNIFORM: 'uniform',
    SCRIPTED: 'scripted',
    INVERSE: 'inverse',
    DETERMINISTIC: 'deterministic',
}


def make_schedule(milestones: List[int], values: List[float], num_iterations: int) -> np.array:
    """
    Change every `milestones` with the respective `values`

    >>> make_schedule(milestones=[0, 3, 8], values=[1, .5, .2], num_iterations=10)
    [1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
    """
    iteration2value = [None] * num_iterations
    change_point_idx = 0

    for i in range(num_iterations):
        if change_point_idx < len(milestones) - 1 and i == milestones[change_point_idx + 1]:
            change_point_idx += 1
        iteration2value[i] = values[change_point_idx]

    return np.array(iteration2value)


class Scheduler:
    def __init__(self, config: Config):
        iters = config.num_iterations + 1
        self.lrs               = make_schedule(config.lr_milestones,             config.lr_values,             iters)
        self.entropy_coefs     = make_schedule(config.entropy_coef_milestones,   config.entropy_coef_values,   iters)
        self.uniform_probas    = make_schedule(config.uniform_proba_milestones, config.uniform_proba_values, iters)
        self.scripted_probas   = make_schedule(config.scripted_proba_milestones, config.scripted_proba_values, iters)
        self.inverse_probas    = make_schedule(config.inverse_proba_milestones, config.inverse_proba_values, iters)
        self.variational_coefs = make_schedule(config.variational_constraint_milestones, config.variational_constraint_values, iters)
        self.latent_mi_coefs   = make_schedule(config.latent_constraint_milestones, config.latent_constraint_values, iters)

        self.sample_probas = 1 - self.uniform_probas - self.scripted_probas - self.inverse_probas
        self.sample_probas = np.maximum(0, self.sample_probas)

        self.current_iteration = 0
        self.progress_history = []
        self.winrate_threshold = config.winrate_threshold
        self.first_no_adjustment = config.first_no_adjustment

        self.lr_adjust_value = config.adjust_lr_to
        self.latent_mi_coef_adjust_value = config.adjust_mi_to
        self.uniform_proba_adjust_value = config.adjust_uniform_to
        self.scripted_proba_adjust_value = config.adjust_scripted_to

    def end_iteration_report(self, winrate: float):
        self.progress_history.append(winrate)
        self.current_iteration += 1

        if self.first_no_adjustment != 0 and self.current_iteration < self.first_no_adjustment:
            return

        def _adjust_if_enabled(adjust_value, values_list):
            if adjust_value is not None:
                values_list[self.current_iteration] = adjust_value

        # Process adjustments
        if winrate > self.winrate_threshold:
            _adjust_if_enabled(self.lr_adjust_value, self.lrs)
            _adjust_if_enabled(self.latent_mi_coef_adjust_value, self.latent_mi_coefs)
            _adjust_if_enabled(self.uniform_proba_adjust_value, self.uniform_probas)
        if winrate < 1 - self.winrate_threshold:
            _adjust_if_enabled(self.scripted_proba_adjust_value, self.scripted_probas)

    @property
    def current_lr(self) -> float:
        return self.lrs[self.current_iteration]

    @property
    def current_entropy_coef(self) -> float:
        return self.entropy_coefs[self.current_iteration]

    @property
    def current_variational_coef(self) -> float:
        return self.variational_coefs[self.current_iteration]

    @property
    def current_latent_mi_coef(self) -> float:
        return self.latent_mi_coefs[self.current_iteration]

    def _normalized_action_source_probas(self) -> np.array:
        p = np.array([
            self.sample_probas  [self.current_iteration],
            self.uniform_probas [self.current_iteration],
            self.scripted_probas[self.current_iteration],
            self.inverse_probas [self.current_iteration],
        ])
        return p / sum(p)

    def pick_action_source(self) -> int:
        return np.random.choice(4, p=self._normalized_action_source_probas())

    @property
    def current_values(self) -> dict:
        """ For logging """
        probas = {
            action_source_names[i] + '_proba': p
            for i, p in enumerate(self._normalized_action_source_probas())
        }
        return {
            'lr':               self.current_lr,
            'entropy_coef':     self.current_entropy_coef,
            'variational_coef': self.current_variational_coef,
            'latent_mi_coef':   self.current_latent_mi_coef,
            **probas
        }


if __name__ == '__main__':
    import doctest
    doctest.testmod()

