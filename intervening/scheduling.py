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
        self.lrs            = make_schedule(config.lr_milestones,             config.lr_values,             iters)
        self.entropy_coefs  = make_schedule(config.entropy_coef_milestones,   config.entropy_coef_values,   iters)
        self.uniform_proba  = make_schedule(config.uniform_proba_milestones,  config.uniform_proba_values,  iters)
        self.scripted_proba = make_schedule(config.scripted_proba_milestones, config.scripted_proba_values, iters)
        self.inverse_proba  = make_schedule(config.inverse_proba_milestones,  config.inverse_proba_values,  iters)

        self.sample_proba = 1 - self.uniform_proba - self.scripted_proba - self.inverse_proba
        self.sample_proba = np.maximum(0, self.sample_proba)

        self.current_update = 0

    @property
    def lr(self) -> float:
        return self.lrs[self.current_update]

    @property
    def entropy_coef(self) -> float:
        return self.entropy_coefs[self.current_update]

    def _normalized_action_source_probas(self) -> np.array:
        p = np.array([
            self.sample_proba  [self.current_update],
            self.uniform_proba [self.current_update],
            self.scripted_proba[self.current_update],
            self.inverse_proba [self.current_update],
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
            'lr':            self.lr,
            'entropy_coef':  self.entropy_coef,
            **probas
        }


if __name__ == '__main__':
    import doctest
    doctest.testmod()

