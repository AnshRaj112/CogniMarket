"""Run end-to-end training evidence evaluation and write publishable artifacts.

This script is designed to satisfy the "show real learning" bar:
- compare a baseline policy vs untrained model vs trained model
- evaluate all series on identical seeds
- write JSON + Markdown artifacts for README / writeups

Outputs (under --checkpoint-dir):
- training_progress.json    # baseline/pre/post metrics for plotting
- evidence_summary.md       # paste-ready README section
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from compute_bazaar_env import ComputeBazaarEnv, build_agent_ids
from evaluate import (
    baseline_policy,
    load_model_policy,
    run_episode,
    run_rule_baseline_metrics,
    strategic_baseline_policy,
)


Policy = Callable[[Dict[str, Any], int], str]


def random_policy(obs: Dict[str, Any], round_num: int) -> str:
    """Simple untrained baseline: random valid propose / accept-no action."""
    agent_ids = obs.get("agent_ids", build_agent_ids(2))
    total_pool = int(float(obs.get("total_pool", 100.0)))
    rng = random.Random(round_num + total_pool)

    if round_num > 1 and rng.random() < 0.20:
        return "ACCEPT: NO"

    # Random but valid allocation per resource, normalized to total_pool.
    parts_by_agent = {a: {"gpu": 0, "cpu": 0, "memory": 0} for a in agent_ids}
    for rk in ("gpu", "cpu", "memory"):
        cuts = sorted([0, total_pool] + [rng.randint(0, total_pool) for _ in range(len(agent_ids) - 1)])
        shares = [cuts[i + 1] - cuts[i] for i in range(len(agent_ids))]
        for i, aid in enumerate(agent_ids):
            parts_by_agent[aid][rk] = shares[i]

    chunks = []
    for aid in agent_ids:
        a = parts_by_agent[aid]
        chunks.append(f"{aid}: gpu {a['gpu']} cpu {a['cpu']} memory {a['memory']}")
    return "PROPOSE: " + "; ".join(chunks)


def evaluate_policy_metrics(
    policy: Policy,
    *,
    policy_name: str,
    num_episodes: int,
    difficulty: str,
    max_rounds: int,
    seed: int,
    num_opponents: int,
) -> Dict[str, Any]:
    """Run fixed-seed multi-episode eval and return aggregate metrics."""
    env = ComputeBazaarEnv(
        max_rounds=max_rounds,
        seed=seed,
        agent_ids=build_agent_ids(num_opponents),
    )
    results: List[Dict[str, Any]] = []
    for ep in range(1, num_episodes + 1):
        ep_seed = seed + ep
        results.append(run_episode(env, difficulty=difficulty, seed=ep_seed, policy=policy))

    rewards = [float(r["total_reward"]) for r in results]
    utilities = [float(r["utility_achieved"]) for r in results]
    rounds = [float(r["rounds_used"]) for r in results]
    successes = [bool(r["success"]) for r in results]

    return {
        "policy": policy_name,
        "num_episodes": num_episodes,
        "difficulty": difficulty,
        "seed": seed,
        "success_rate": sum(1 for s in successes if s) / max(len(successes), 1),
        "avg_reward": statistics.mean(rewards) if rewards else 0.0,
        "reward_std": statistics.stdev(rewards) if len(rewards) > 1 else 0.0,
        "avg_utility": statistics.mean(utilities) if utilities else 0.0,
        "utility_std": statistics.stdev(utilities) if len(utilities) > 1 else 0.0,
        "avg_rounds": statistics.mean(rounds) if rounds else 0.0,
    }


def _fmt_metrics_row(name: str, row: Dict[str, Any]) -> str:
    return (
        f"| {name} | {row['success_rate'] * 100:.1f}% | "
        f"{row['avg_reward']:.3f} ± {row.get('reward_std', 0.0):.3f} | "
        f"{row['avg_utility']:.3f} ± {row.get('utility_std', 0.0):.3f} | "
        f"{row['avg_rounds']:.2f} |"
    )


def build_markdown_summary(
    *,
    baseline: Dict[str, Any],
    pre: Dict[str, Any],
    post: Dict[str, Any],
    episodes: int,
    difficulty: str,
    max_rounds: int,
    seed: int,
) -> str:
    delta_reward = post["avg_reward"] - pre["avg_reward"]
    delta_success = (post["success_rate"] - pre["success_rate"]) * 100.0
    delta_utility = post["avg_utility"] - pre["avg_utility"]

    lines = [
        "## Training Evidence (Environment-Connected, End-to-End)",
        "",
        "All evaluations use the same environment and same per-episode seeds.",
        "",
        "### Setup",
        f"- Difficulty: `{difficulty}`",
        f"- Max rounds: `{max_rounds}`",
        f"- Episodes per series: `{episodes}`",
        f"- Base seed: `{seed}` (episode seeds = `seed + ep`)",
        "",
        "### Quantitative Comparison",
        "| Series | Success Rate | Avg Reward | Avg Utility | Avg Rounds |",
        "|---|---:|---:|---:|---:|",
        _fmt_metrics_row("Rule baseline", baseline),
        _fmt_metrics_row("Untrained model", pre),
        _fmt_metrics_row("Trained model", post),
        "",
        "### Improvement (Trained - Untrained)",
        f"- Reward delta: `{delta_reward:+.3f}`",
        f"- Success-rate delta: `{delta_success:+.1f} pp`",
        f"- Utility delta: `{delta_utility:+.3f}`",
        "",
        "### Curves",
        "Generate training curves (and bar panel from `training_progress.json`) with:",
        "```bash",
        "python plot_training_rewards.py --checkpoint-dir ./checkpoints --smooth 10",
        "```",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate end-to-end training evidence artifacts.")
    parser.add_argument("--checkpoint-dir", default="./checkpoints", help="Checkpoint path (also output directory).")
    parser.add_argument("--base-model", default=None, help="Base model name for loading LoRA adapter checkpoints.")
    parser.add_argument("--episodes", type=int, default=200, help="Evaluation episodes per series.")
    parser.add_argument("--difficulty", choices=["easy", "hard"], default="hard", help="Evaluation difficulty.")
    parser.add_argument("--max-rounds", type=int, default=12, dest="max_rounds", help="Max rounds per episode.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--num-opponents", type=int, default=2, dest="num_opponents", help="Number of opponents.")
    parser.add_argument(
        "--pre-policy",
        choices=["random", "strategic", "baseline"],
        default="random",
        dest="pre_policy",
        help="What to use as untrained baseline when no --base-model is provided.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    baseline = run_rule_baseline_metrics(
        num_episodes=args.episodes,
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        seed=args.seed,
        num_opponents=args.num_opponents,
    )
    baseline["num_episodes"] = args.episodes
    baseline["difficulty"] = args.difficulty
    baseline["seed"] = args.seed
    baseline.setdefault("reward_std", 0.0)
    baseline.setdefault("utility_std", 0.0)

    if args.base_model:
        pre_policy = load_model_policy(
            checkpoint_dir=args.base_model,
            difficulty=args.difficulty,
            max_rounds=args.max_rounds,
            base_model=None,
        )
        pre_name = f"untrained_{Path(args.base_model).name.replace('/', '_')}"
    else:
        if args.pre_policy == "strategic":
            pre_policy = strategic_baseline_policy
            pre_name = "strategic_policy_proxy"
        elif args.pre_policy == "baseline":
            pre_policy = baseline_policy
            pre_name = "baseline_equal_split_proxy"
        else:
            pre_policy = random_policy
            pre_name = "random_policy"

    pre = evaluate_policy_metrics(
        pre_policy,
        policy_name=pre_name,
        num_episodes=args.episodes,
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        seed=args.seed,
        num_opponents=args.num_opponents,
    )

    post_policy = load_model_policy(
        checkpoint_dir=str(checkpoint_dir),
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        base_model=args.base_model,
    )
    post = evaluate_policy_metrics(
        post_policy,
        policy_name="trained_model",
        num_episodes=args.episodes,
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        seed=args.seed,
        num_opponents=args.num_opponents,
    )

    progress = {"baseline": baseline, "pre": pre, "post": post}
    progress_path = checkpoint_dir / "training_progress.json"
    progress_path.write_text(json.dumps(progress, indent=2), encoding="utf-8")

    summary_md = build_markdown_summary(
        baseline=baseline,
        pre=pre,
        post=post,
        episodes=args.episodes,
        difficulty=args.difficulty,
        max_rounds=args.max_rounds,
        seed=args.seed,
    )
    summary_path = checkpoint_dir / "evidence_summary.md"
    summary_path.write_text(summary_md, encoding="utf-8")

    print(f"Wrote {progress_path.resolve()}")
    print(f"Wrote {summary_path.resolve()}")
    print("\nQuick next step:")
    print(f"python plot_training_rewards.py --checkpoint-dir {checkpoint_dir} --smooth 10")


if __name__ == "__main__":
    main()
