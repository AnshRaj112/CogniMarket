"""Evaluate ComputeBazaar with composable OpenEnv-style rubrics.

Usage:
    python evaluate_openenv_rubrics.py --episodes 20 --difficulty hard --policy strategic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Callable, Dict, List

from compute_bazaar_env import ComputeBazaarEnv
from evaluate import baseline_policy, strategic_baseline_policy
from openenv_rubrics import ComputeBazaarRubric, EpisodeTrace, summarize_breakdowns


def _run_episode_trace(
    env: ComputeBazaarEnv,
    policy: Callable[[Dict[str, object], int], str],
    difficulty: str,
    seed: int | None,
) -> EpisodeTrace:
    obs, _ = env.reset(seed=seed, options={"difficulty": difficulty})
    done = False
    truncated = False
    round_num = 0

    rewards: List[float] = []
    infos: List[Dict[str, object]] = []
    actions: List[str] = []

    while not done and not truncated:
        round_num += 1
        action = policy(obs, round_num)
        obs, reward, done, truncated, info = env.step(action)
        rewards.append(float(reward))
        infos.append(dict(info))
        actions.append(action)

    return EpisodeTrace(rewards=rewards, infos=infos, actions=actions)


def evaluate_rubrics(
    episodes: int,
    difficulty: str,
    max_rounds: int,
    seed: int | None,
    policy_name: str,
) -> Dict[str, object]:
    env = ComputeBazaarEnv(max_rounds=max_rounds, seed=seed)
    policy = strategic_baseline_policy if policy_name == "strategic" else baseline_policy

    rubric = ComputeBazaarRubric()
    breakdowns: List[Dict[str, float]] = []
    rewards: List[float] = []
    successes: List[bool] = []

    for ep in range(1, episodes + 1):
        ep_seed = None if seed is None else seed + ep
        trace = _run_episode_trace(env, policy=policy, difficulty=difficulty, seed=ep_seed)
        breakdown = rubric.score_with_breakdown(trace)
        breakdowns.append(breakdown)
        rewards.append(sum(trace.rewards))
        success = bool(trace.infos[-1].get("success", False)) if trace.infos else False
        successes.append(success)

    avg = summarize_breakdowns(breakdowns)
    pass_flags = {
        "rich_informative_signal": avg.get("dense_signal", 0.0) >= 0.75,
        "captures_hard_to_measure_proxy": avg.get("hard_to_measure_proxy", 0.0) >= 0.60,
        "composable_rubric_used": True,
        "hard_to_game": avg.get("anti_gaming", 0.0) >= 0.70,
    }

    return {
        "config": {
            "episodes": episodes,
            "difficulty": difficulty,
            "max_rounds": max_rounds,
            "seed": seed,
            "policy": policy_name,
        },
        "episode_avg_reward": mean(rewards) if rewards else 0.0,
        "success_rate": (sum(1 for s in successes if s) / len(successes)) if successes else 0.0,
        "rubric_scores": avg,
        "checks": pass_flags,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate composable OpenEnv-style rubrics.")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--difficulty", choices=["easy", "hard"], default="hard")
    parser.add_argument("--max-rounds", type=int, default=12, dest="max_rounds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--policy", choices=["baseline", "strategic"], default="strategic")
    parser.add_argument("--output-json", default="rubric_report.json", dest="output_json")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = evaluate_rubrics(
        episodes=args.episodes,
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        seed=args.seed,
        policy_name=args.policy,
    )
    out_path = Path(args.output_json)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nSaved rubric report to: {out_path.resolve()}")
