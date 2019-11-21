import unittest

from agent.storage import RolloutStorage
from configs.structure import Config


class TestReturns(unittest.TestCase):
    def _make_storage(self, reward_seq, done_seq, values_seq=None, gae_lambda=0., discount=.5, reset_before_insert=False):
        config = Config()
        config.num_transitions = len(reward_seq)
        config.gae_lambda = gae_lambda
        config.discount = discount
        storage = RolloutStorage(config, env_state_shape=(1,))

        if reset_before_insert:
            storage.reset()

        if values_seq is None:
            values_seq = [0.] * len(reward_seq)

        dummy_action = 0
        dummy_action_log_prob = 0.
        dummy_env_state = [0]

        for r, d, v in zip(reward_seq, done_seq, values_seq):
            storage.insert(
                dummy_env_state,
                dummy_action,
                dummy_action_log_prob,
                v,
                r,
                d,
                None,
                None,
            )

        return storage

    def test_one_episode(self):
        storage = self._make_storage(
            reward_seq=[0, 0, 1],
            done_seq=[False, False, True],
        )
        storage.compute_returns()

        self.assertEqual(
            2,
            storage.last_done,
        )
        self.assertEqual(
            [.25, .5, 1],
            list(storage.returns.numpy())[:-1],
        )

    def test_two_episodes(self):
        storage = self._make_storage(
            reward_seq=[0, 0, 1, 0, 1],
            done_seq=[False, False, True, False, True],
        )
        storage.compute_returns()

        self.assertEqual(
            4,
            storage.last_done,
        )
        self.assertEqual(
            [.25, .5, 1, .5, 1],
            list(storage.returns.numpy())[:-1],
        )

    def test_reset_one_episode(self):
        storage = self._make_storage(
            reward_seq=[0, 0, 1],
            done_seq=[False, False, True],
            reset_before_insert=True,
        )
        storage.compute_returns()

        self.assertEqual(
            2,
            storage.last_done,
        )
        self.assertEqual(
            [.25, .5, 1],
            list(storage.returns.numpy())[:-1],
        )

    def test_reset_two_episodes(self):
        storage = self._make_storage(
            reward_seq=[0, 0, 1, 0, 1],
            done_seq=[False, False, True, False, True],
            reset_before_insert=True,
        )
        storage.compute_returns()

        self.assertEqual(
            4,
            storage.last_done,
        )
        self.assertEqual(
            [.25, .5, 1, .5, 1],
            list(storage.returns.numpy())[:-1],
        )

    def test_gae(self):
        storage = self._make_storage(
            reward_seq=[0, 0, 0, 1],
            done_seq=[False, False, False, True],
            # returns .125 .25 .5 1
            values_seq=[.125, .6, 2, 2],
            gae_lambda=.1,
        )
        storage.compute_returns()

        self.assertEqual(
            [.25, .5, 1.],
            list(storage.returns.numpy())[:-1],
        )


if __name__ == '__main__':
    unittest.main()
