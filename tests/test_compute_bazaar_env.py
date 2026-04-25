import unittest

from compute_bazaar_env import ComputeBazaarEnv


class TestComputeBazaarEnv(unittest.TestCase):
    def test_reset_returns_expected_observation_shape(self):
        env = ComputeBazaarEnv(seed=7)
        obs, info = env.reset(options={"difficulty": "easy"})
        self.assertIn("conversation_history", obs)
        self.assertIn("private_utility", obs)
        self.assertIn("remaining_compute_pool", obs)
        self.assertIn("last_proposal", obs)
        self.assertIn("last_opponent_response", obs)
        self.assertIn("last_opponent_utility", obs)
        self.assertEqual(len(obs["private_utility"]), 3)
        self.assertEqual(info["difficulty"], "easy")

    def test_sampled_utilities_are_normalized_and_non_negative(self):
        env = ComputeBazaarEnv(seed=7)
        env.reset(options={"difficulty": "hard"})
        for vec in env.utilities.values():
            self.assertAlmostEqual(float(sum(vec)), 1.0, places=6)
            self.assertTrue(all(value >= 0.0 for value in vec))

    def test_parse_allocation_and_successful_step(self):
        env = ComputeBazaarEnv(seed=7)
        env.reset(options={"difficulty": "easy"})
        action = (
            "propose learner: gpu 34 cpu 33 memory 33; "
            "opponent_1: gpu 33 cpu 34 memory 33; "
            "opponent_2: gpu 33 cpu 33 memory 34"
        )
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIsInstance(obs, dict)
        self.assertIsInstance(reward, float)
        self.assertFalse(truncated)
        self.assertIn("rounds_used", info)
        self.assertIsInstance(terminated, bool)

    def test_repeated_proposal_is_penalized(self):
        env = ComputeBazaarEnv(seed=11)
        env.reset(options={"difficulty": "hard"})
        proposal = (
            "PROPOSE: learner: gpu 34 cpu 33 memory 33; "
            "opponent_1: gpu 33 cpu 34 memory 33; "
            "opponent_2: gpu 33 cpu 33 memory 34"
        )
        env.step(proposal)
        _, reward, terminated, _, info = env.step(proposal)
        self.assertFalse(terminated)
        self.assertLessEqual(reward, -3.0)
        self.assertEqual(info.get("success"), False)


if __name__ == "__main__":
    unittest.main()
