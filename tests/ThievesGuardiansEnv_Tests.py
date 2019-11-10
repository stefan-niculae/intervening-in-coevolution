import unittest
import sys
sys.path.insert(1, 'environment')
from ThievesGuardiansEnv import *


class MovementTests(unittest.TestCase):
  def setUp(self):
    """Create simple environment to test thief and guardian movements."""
    self._env = ThievesGuardiansEnv('4x4-test-movements', env_id=0)
    self._init_thief_pos = self._env._id2pos[0]
    self._init_guard_pos = self._env._id2pos[1]

  def basic_movement(self, moves:[int], expected_delta:[int]):
    # Take a step
    self._env.step(moves)

    # Get the new thief and guardian positions
    curr_thief_pos = self._env._id2pos[0]
    curr_guard_pos = self._env._id2pos[1]

    # Check the new positions against expected delta + old position
    self.assertEqual(curr_thief_pos, tuple(self._init_thief_pos + expected_delta))
    self.assertEqual(curr_guard_pos, tuple(self._init_guard_pos + expected_delta))

  def test_move_right(self):
    """Test moving right for thief and guardian."""
    self.basic_movement([RIGHT, RIGHT], action_idx2delta[RIGHT])
  
  def test_move_down(self):
    """Test moving down for thief and guardian."""
    self.basic_movement([DOWN, DOWN], action_idx2delta[DOWN])

  def test_move_up(self):
    """Test moving up for thief and guardian."""
    self.basic_movement([UP, UP], action_idx2delta[UP])

  def test_move_left(self):
    """Test moving left for thief and guardian."""
    self.basic_movement([LEFT, LEFT], action_idx2delta[LEFT])


class RewardsTests(unittest.TestCase):
  def setUp(self):
    """Create simple environment to test rewards."""
    self._env = ThievesGuardiansEnv('4x4-test-rewards', env_id=0)
  
  def test_reward_guard_catch_thief(self):
    """Test reward for guard catching thief."""
    _, reward, _, _ = self._env.step([NOOP, UP])

    self.assertEqual(tuple(reward), REWARDS['killed'])

  def test_reward_thief_runs_into_guard(self):
    """Test reward for thief running into a guard, stupidly."""
    _, reward, _, _ = self._env.step([DOWN, NOOP])

    self.assertEqual(tuple(reward), REWARDS['killed'])
  
  def test_reward_thief_finds_treasure(self):
    """Test reward for thief getting that sweet, sweet treasure."""
    _, reward, _, _ = self._env.step([UP, NOOP])
    
    self.assertEqual(tuple(reward), REWARDS['treasure'])
  
  def test_reward_out_of_time(self):
    """Test rewards for when environment time is spent."""
    # Perform no movements until the max number of steps
    done = False
    while not done:
      _, reward, done, _ = self._env.step([NOOP, NOOP])
    
    self.assertEqual(tuple(reward), REWARDS['out_of_time'])
  

if __name__ == '__main__':
  unittest.main()
