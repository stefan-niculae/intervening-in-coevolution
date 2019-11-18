import unittest

from agent.storage import RolloutStorage
from configs.structure import Config


class TestReturns(unittest.TestCase):
    def _make_storage(self, reward_seq, done_seq, reset_before_insert=False):
        config = Config()
        config.num_transitions = len(reward_seq)
        config.discount = .5
        storage = RolloutStorage(config, env_state_shape=(1,))

        if reset_before_insert:
            storage.reset()

        for r, d in zip(reward_seq, done_seq):
            storage.insert(
                storage.env_states[0],
                storage.actions[0],
                storage.action_log_probs[0],
                r,
                d,
                storage.rec_hs[:, 0],
                storage.rec_cs[:, 0],
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



if __name__ == '__main__':
    unittest.main()
