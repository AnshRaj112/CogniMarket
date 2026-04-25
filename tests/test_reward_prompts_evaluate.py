"""Tests for reward.py, prompts.py, and evaluate.py."""

import unittest

from compute_bazaar_env import ComputeBazaarEnv
from evaluate import baseline_policy, evaluate, run_episode
from prompts import build_learner_hint, build_opponent_prompt, build_oversight_prompt
from reward import (
    NO_DEAL_PENALTY,
    FAST_DEAL_BONUS,
    ROUND_PENALTY,
    SPARSE_BONUS,
    SUCCESS_UTILITY_THRESHOLD,
    calculate_reward,
    calculate_utility,
)


class TestCalculateUtility(unittest.TestCase):
    def test_full_gpu_allocation_with_gpu_preference(self):
        # If all 100 GPU units go to learner and utility weight is entirely GPU,
        # the result should equal MAX_UTILITY_SCALE (15).
        utility = calculate_utility(
            allocation={"gpu": 100, "cpu": 0, "memory": 0},
            utility_vector=[1.0, 0.0, 0.0],
        )
        self.assertAlmostEqual(utility, 15.0, places=5)

    def test_zero_allocation_returns_zero(self):
        utility = calculate_utility({}, [0.5, 0.3, 0.2])
        self.assertEqual(utility, 0.0)

    def test_equal_split_with_uniform_weights(self):
        # 1/3 allocation of each resource, 1/3 weight each → 5.0
        utility = calculate_utility(
            allocation={"gpu": 33.33, "cpu": 33.33, "memory": 33.33},
            utility_vector=[1 / 3, 1 / 3, 1 / 3],
        )
        self.assertAlmostEqual(utility, 5.0, delta=0.1)


class TestCalculateReward(unittest.TestCase):
    def test_successful_fast_deal_grants_all_bonuses(self):
        reward, breakdown = calculate_reward(
            allocation={"gpu": 100, "cpu": 0, "memory": 0},
            utility_vector=[1.0, 0.0, 0.0],
            rounds_used=2,
            deal_closed=True,
        )
        # Utility = 15, rounds_penalty = -0.8 * 2 = -1.6, efficiency = 4, sparse = 10
        self.assertEqual(breakdown["efficiency_bonus"], FAST_DEAL_BONUS)
        self.assertEqual(breakdown["sparse_bonus"], SPARSE_BONUS)
        self.assertEqual(breakdown["no_deal_penalty"], 0.0)
        self.assertAlmostEqual(breakdown["round_penalty"], ROUND_PENALTY * 2, places=5)

    def test_no_deal_applies_penalty(self):
        reward, breakdown = calculate_reward(
            allocation=None,
            utility_vector=[0.5, 0.3, 0.2],
            rounds_used=12,
            deal_closed=False,
        )
        self.assertEqual(breakdown["no_deal_penalty"], NO_DEAL_PENALTY)
        self.assertEqual(breakdown["utility_reward"], 0.0)

    def test_suboptimal_deal_has_no_no_deal_penalty(self):
        # Any closed deal should avoid the global no-deal penalty, even if utility is low.
        _, breakdown = calculate_reward(
            allocation={"gpu": 5, "cpu": 0, "memory": 0},
            utility_vector=[1.0, 0.0, 0.0],
            rounds_used=3,
            deal_closed=True,
        )
        self.assertEqual(breakdown["no_deal_penalty"], 0.0)

    def test_oversight_bonus_applied_when_accurate(self):
        _, breakdown = calculate_reward(
            allocation={"gpu": 60, "cpu": 20, "memory": 20},
            utility_vector=[1.0, 0.0, 0.0],
            rounds_used=4,
            deal_closed=True,
            oversight_accurate=True,
        )
        self.assertGreater(breakdown["oversight_bonus"], 0.0)

    def test_reward_total_equals_sum_of_breakdown(self):
        reward, breakdown = calculate_reward(
            allocation={"gpu": 40, "cpu": 30, "memory": 30},
            utility_vector=[0.6, 0.3, 0.1],
            rounds_used=5,
            deal_closed=True,
        )
        self.assertAlmostEqual(reward, sum(breakdown.values()), places=8)


class TestPrompts(unittest.TestCase):
    def test_opponent_prompt_contains_agent_id(self):
        prompt = build_opponent_prompt(
            agent_id="opponent_1",
            utility_vector=[0.5, 0.3, 0.2],
            conversation_history=["learner: hi"],
            remaining_pool={"gpu": 100.0, "cpu": 100.0, "memory": 100.0},
        )
        self.assertIn("opponent_1", prompt)
        self.assertIn("GPU", prompt)

    def test_oversight_prompt_includes_proposal(self):
        proposal = {
            "learner": {"gpu": 34.0, "cpu": 33.0, "memory": 33.0},
            "opponent_1": {"gpu": 33.0, "cpu": 34.0, "memory": 33.0},
            "opponent_2": {"gpu": 33.0, "cpu": 33.0, "memory": 34.0},
        }
        prompt = build_oversight_prompt(
            conversation_history=["learner: propose …"],
            current_proposal=proposal,
        )
        self.assertIn("learner", prompt)
        self.assertIn("feasible", prompt.lower())

    def test_learner_hint_contains_utility_and_rounds(self):
        hint = build_learner_hint(
            utility_vector=[0.5, 0.3, 0.2],
            conversation_history=[],
            remaining_pool={"gpu": 100.0, "cpu": 100.0, "memory": 100.0},
            rounds_remaining=8,
        )
        self.assertIn("rounds_left=8", hint)
        self.assertIn("GPU=0.500", hint)


class TestEvaluateLoop(unittest.TestCase):
    def test_run_episode_returns_expected_keys(self):
        env = ComputeBazaarEnv(seed=1)
        metrics = run_episode(env, difficulty="easy", seed=1)
        for key in ("total_reward", "utility_achieved", "rounds_used", "success"):
            self.assertIn(key, metrics)

    def test_evaluate_runs_without_error(self):
        # Smoke test: just make sure it completes without raising.
        evaluate(episodes=3, difficulty="easy", seed=7)

    def test_baseline_policy_returns_equal_split_on_round1(self):
        env = ComputeBazaarEnv(seed=5)
        obs, _ = env.reset()
        action = baseline_policy(obs, round_num=1)
        # learner is index 0 and receives the 1-unit remainder → 34
        self.assertIn("learner: gpu 34", action)


if __name__ == "__main__":
    unittest.main()
