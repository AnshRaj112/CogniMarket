"""Minimal evaluation loop for Compute Allocation Bazaar.

Runs N episodes (default 10) with a simple rule-based learner that always
proposes an equal three-way split, then prints per-episode and aggregate
metrics.

Usage:
    python evaluate.py                    # 10 episodes, hard difficulty
    python evaluate.py --episodes 20 --difficulty easy
    python evaluate.py --episodes 5 --seed 42
"""

from __future__ import annotations

import argparse
import re
import statistics
from typing import Any, Callable, Dict, List, Optional

from compute_bazaar_env import (
    ComputeBazaarEnv,
    build_agent_ids,
    clean_action,
    proposal_has_all_agents,
    validate_and_fix_proposal,
)


# ---------------------------------------------------------------------------
# Simple baseline policy (rule-based equal split)
# ---------------------------------------------------------------------------

def _build_equal_split_action(
    total_pool: float = 100.0,
    agent_ids: Optional[List[str]] = None,
) -> str:
    """Build an equal-split proposal string derived from the given pool size.

    Each agent receives ``floor(total_pool / 3)`` units per resource; the
    first ``total_pool % 3`` agents each receive one extra unit to ensure
    the pool is fully allocated.
    """
    agents = tuple(agent_ids or ["learner", "opponent_1", "opponent_2"])
    per_agent = int(total_pool) // len(agents)
    remainder = int(total_pool) % len(agents)
    parts = []
    for i, agent in enumerate(agents):
        share = per_agent + (1 if i < remainder else 0)
        parts.append(f"{agent}: gpu {share} cpu {share} memory {share}")
    return "PROPOSE: " + "; ".join(parts)


_EQUAL_SPLIT_ACTION = _build_equal_split_action()


def baseline_policy(obs: Dict[str, Any], round_num: int) -> str:
    """Rule-based baseline: propose equal split on round 1, accept otherwise.

    Args:
        obs: Current observation from the environment.
        round_num: 1-indexed round counter.

    Returns:
        An action string.
    """
    agent_ids = obs.get("agent_ids", ["learner", "opponent_1", "opponent_2"])
    total_pool = float(obs.get("total_pool", 100.0))
    equal_split_action = _build_equal_split_action(total_pool=total_pool, agent_ids=agent_ids)
    if round_num == 1:
        return equal_split_action
    # If the deal exists in the history (any opponent said "accept"), mirror-accept.
    history = obs.get("conversation_history", [])
    if any("accept" in turn for turn in history if not turn.startswith("learner")):
        return "ACCEPT: YES"
    # Threshold-aware concession for multi-opponent scenarios.
    # In easy mode, opponents typically need >=4 utility; allocating 30% each gives
    # 4.5 utility regardless of utility vector shape (0.30 * 15).
    opponents = [a for a in agent_ids if a != "learner"]
    if not opponents:
        return equal_split_action
    pool_i = int(total_pool)
    target_opp = max(0, min(pool_i, int(round(0.30 * pool_i))))
    total_needed = target_opp * len(opponents)
    if total_needed > pool_i:
        target_opp = pool_i // len(opponents)
        total_needed = target_opp * len(opponents)
    learner_share = max(0, pool_i - total_needed)

    parts = [f"learner: gpu {learner_share} cpu {learner_share} memory {learner_share}"]
    for opp in opponents:
        parts.append(f"{opp}: gpu {target_opp} cpu {target_opp} memory {target_opp}")
    return "PROPOSE: " + "; ".join(parts)


# ---------------------------------------------------------------------------
# Strategic baseline policy (diverse proposals for cold-start dataset)
# ---------------------------------------------------------------------------

import random as _random

_strategic_rng = _random.Random(42)


def _build_biased_proposal(
    learner_utility: List[float],
    learner_share_pct: float = 0.28,
    total_pool: float = 100.0,
) -> str:
    """Build a proposal that gives MORE to opponents, biased by learner utility.

    The learner keeps ``learner_share_pct`` of the pool on average but
    concentrates its share on its highest-valued resource. The remaining
    resources are split generously between opponents with some randomness.

    Args:
        learner_utility: Learner's private utility weights [gpu, cpu, memory].
        learner_share_pct: How much of the pool the learner keeps (0.0-1.0).
        total_pool: Total pool per resource (100).

    Returns:
        A PROPOSE: action string.
    """
    resources = ["gpu", "cpu", "memory"]
    pool = int(total_pool)

    # Learner keeps more of what it values, less of what it doesn't
    learner_alloc = {}
    for i, res in enumerate(resources):
        weight = learner_utility[i]
        # Bias: take more of high-value resources, less of low-value
        share = int(pool * learner_share_pct * (0.5 + weight))
        share = max(5, min(share, pool - 10))  # Clamp to [5, pool-10]
        learner_alloc[res] = share

    # Split remainder between opponents with some noise
    opp1_alloc = {}
    opp2_alloc = {}
    for res in resources:
        remaining = pool - learner_alloc[res]
        # Add randomness to opponent split (40-60% to each)
        split_ratio = _strategic_rng.uniform(0.40, 0.60)
        opp1_share = int(remaining * split_ratio)
        opp2_share = remaining - opp1_share
        opp1_alloc[res] = opp1_share
        opp2_alloc[res] = opp2_share

    parts = [
        f"learner: gpu {learner_alloc['gpu']} cpu {learner_alloc['cpu']} memory {learner_alloc['memory']}",
        f"opponent_1: gpu {opp1_alloc['gpu']} cpu {opp1_alloc['cpu']} memory {opp1_alloc['memory']}",
        f"opponent_2: gpu {opp2_alloc['gpu']} cpu {opp2_alloc['cpu']} memory {opp2_alloc['memory']}",
    ]
    return "PROPOSE: " + "; ".join(parts)


def _build_concession_proposal(
    learner_utility: List[float],
    round_num: int,
    history: List[str],
    total_pool: float = 100.0,
) -> str:
    """Build progressively more generous proposals as rounds increase.

    Each rejected round, the learner gives up more resources to opponents.
    The concession is fastest on the learner's least-valued resource.

    Args:
        learner_utility: Learner's private utility weights.
        round_num: Current round (1-indexed).
        history: Conversation history.
        total_pool: Total pool per resource.

    Returns:
        A PROPOSE: action string.
    """
    resources = ["gpu", "cpu", "memory"]
    pool = int(total_pool)

    # Concession rate increases with rounds
    concession = min(0.08 * round_num, 0.40)  # Max 40% concession
    learner_base_pct = max(0.15, 0.35 - concession)

    # Prioritize keeping high-value resources
    sorted_res = sorted(range(3), key=lambda i: learner_utility[i], reverse=True)

    learner_alloc = {}
    for rank, idx in enumerate(sorted_res):
        res = resources[idx]
        if rank == 0:
            # Keep most of best resource
            share = int(pool * (learner_base_pct + 0.10))
        elif rank == 1:
            share = int(pool * learner_base_pct)
        else:
            # Give away most of worst resource
            share = int(pool * max(0.10, learner_base_pct - 0.10))
        learner_alloc[res] = max(5, min(share, pool - 10))

    # Split remainder with randomness
    opp1_alloc = {}
    opp2_alloc = {}
    for res in resources:
        remaining = pool - learner_alloc[res]
        split = _strategic_rng.uniform(0.42, 0.58)
        opp1_alloc[res] = int(remaining * split)
        opp2_alloc[res] = remaining - opp1_alloc[res]

    parts = [
        f"learner: gpu {learner_alloc['gpu']} cpu {learner_alloc['cpu']} memory {learner_alloc['memory']}",
        f"opponent_1: gpu {opp1_alloc['gpu']} cpu {opp1_alloc['cpu']} memory {opp1_alloc['memory']}",
        f"opponent_2: gpu {opp2_alloc['gpu']} cpu {opp2_alloc['cpu']} memory {opp2_alloc['memory']}",
    ]
    return "PROPOSE: " + "; ".join(parts)


def strategic_baseline_policy(obs: Dict[str, Any], round_num: int) -> str:
    """Diverse strategic baseline that breaks equal-split bias.

    - Round 1: Biased proposal favoring opponents (learner keeps ~28%)
    - Rounds 2+:
      - If opponents accepted previous, ACCEPT: YES
      - Otherwise, concede progressively (give more to opponents each round)
    - Varies allocations across resources based on learner utility weights

    Args:
        obs: Current observation from the environment.
        round_num: 1-indexed round counter.

    Returns:
        An action string.
    """
    history = obs.get("conversation_history", [])
    utility = obs.get("private_utility", [0.33, 0.33, 0.34])

    if round_num == 1:
        return _build_biased_proposal(utility, learner_share_pct=0.28)

    # If any opponent accepted, accept too
    if any("accept" in turn.lower() for turn in history if not turn.startswith("learner")):
        return "ACCEPT: YES"

    # Progressive concession
    return _build_concession_proposal(utility, round_num, history)


def _build_prompt_from_obs(
    obs: Dict[str, Any], round_num: int, difficulty: str, max_rounds: int
) -> str:
    """Build a learner prompt from environment observation."""
    from prompts import build_learner_hint  # local import for robustness

    rounds_remaining = max_rounds - round_num
    return build_learner_hint(
        utility_vector=obs["private_utility"],
        conversation_history=obs["conversation_history"],
        remaining_pool=obs["remaining_compute_pool"],
        rounds_remaining=rounds_remaining,
        difficulty=difficulty,
        agent_ids=obs.get("agent_ids", ["learner", "opponent_1", "opponent_2"]),
    )


def _shift_toward_opponents(action: str, shift: int = 2) -> str:
    """Concede a small amount from learner to each opponent per resource."""
    if not action.lower().startswith("propose:"):
        return action
    pattern = re.compile(
        r"(learner|opponent_1|opponent_2)\s*:\s*gpu\s*(\d+(?:\.\d+)?)\s*cpu\s*(\d+(?:\.\d+)?)\s*memory\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(action)
    if len(matches) < 3:
        return action

    alloc: Dict[str, Dict[str, int]] = {}
    for agent, gpu, cpu, memory in matches[:3]:
        alloc[agent.lower()] = {
            "gpu": int(float(gpu)),
            "cpu": int(float(cpu)),
            "memory": int(float(memory)),
        }

    if any(a not in alloc for a in ("learner", "opponent_1", "opponent_2")):
        return action

    for res in ("gpu", "cpu", "memory"):
        take = min(2 * shift, alloc["learner"][res])
        if take <= 0:
            continue
        alloc["learner"][res] -= take
        alloc["opponent_1"][res] += take // 2
        alloc["opponent_2"][res] += take - (take // 2)

    adapted = (
        "PROPOSE: "
        f"learner: gpu {alloc['learner']['gpu']} cpu {alloc['learner']['cpu']} memory {alloc['learner']['memory']}; "
        f"opponent_1: gpu {alloc['opponent_1']['gpu']} cpu {alloc['opponent_1']['cpu']} memory {alloc['opponent_1']['memory']}; "
        f"opponent_2: gpu {alloc['opponent_2']['gpu']} cpu {alloc['opponent_2']['cpu']} memory {alloc['opponent_2']['memory']}"
    )
    return validate_and_fix_proposal(adapted)


def _recent_opponent_status(history: List[str]) -> Dict[str, str]:
    """Return latest acceptance status for each opponent from history."""
    status = {"opponent_1": "unknown", "opponent_2": "unknown"}
    for turn in reversed(history):
        lower = turn.lower()
        if lower.startswith("opponent_1:") and status["opponent_1"] == "unknown":
            if "accept" in lower and "reject" not in lower:
                status["opponent_1"] = "accept"
            elif "reject" in lower:
                status["opponent_1"] = "reject"
        elif lower.startswith("opponent_2:") and status["opponent_2"] == "unknown":
            if "accept" in lower and "reject" not in lower:
                status["opponent_2"] = "accept"
            elif "reject" in lower:
                status["opponent_2"] = "reject"
        if status["opponent_1"] != "unknown" and status["opponent_2"] != "unknown":
            break
    return status


def _concede_to_rejecting_opponent(action: str, rejecting_opponent: str, shift: int = 3) -> str:
    """Increase rejecting opponent allocation by taking from learner and other opponent."""
    if not action.lower().startswith("propose:"):
        return action
    pattern = re.compile(
        r"(learner|opponent_1|opponent_2)\s*:\s*gpu\s*(\d+(?:\.\d+)?)\s*cpu\s*(\d+(?:\.\d+)?)\s*memory\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(action)
    if len(matches) < 3:
        return action

    alloc: Dict[str, Dict[str, int]] = {}
    for agent, gpu, cpu, memory in matches[:3]:
        alloc[agent.lower()] = {
            "gpu": int(float(gpu)),
            "cpu": int(float(cpu)),
            "memory": int(float(memory)),
        }
    if any(a not in alloc for a in ("learner", "opponent_1", "opponent_2")):
        return action

    other = "opponent_2" if rejecting_opponent == "opponent_1" else "opponent_1"
    for res in ("gpu", "cpu", "memory"):
        take_from_learner = min(shift, alloc["learner"][res])
        take_from_other = min(max(1, shift // 2), alloc[other][res])
        gain = take_from_learner + take_from_other
        alloc["learner"][res] -= take_from_learner
        alloc[other][res] -= take_from_other
        alloc[rejecting_opponent][res] += gain

    adapted = (
        "PROPOSE: "
        f"learner: gpu {alloc['learner']['gpu']} cpu {alloc['learner']['cpu']} memory {alloc['learner']['memory']}; "
        f"opponent_1: gpu {alloc['opponent_1']['gpu']} cpu {alloc['opponent_1']['cpu']} memory {alloc['opponent_1']['memory']}; "
        f"opponent_2: gpu {alloc['opponent_2']['gpu']} cpu {alloc['opponent_2']['cpu']} memory {alloc['opponent_2']['memory']}"
    )
    return validate_and_fix_proposal(adapted)


def make_model_policy(model, tokenizer, difficulty: str, max_rounds: int) -> Callable[[Dict[str, Any], int], str]:
    """Create a policy callable backed by a trained language model."""
    import torch  # type: ignore
    last_raw_output: Optional[str] = None

    def _policy(obs: Dict[str, Any], round_num: int) -> str:
        nonlocal last_raw_output
        agent_ids = tuple(obs.get("agent_ids", ["learner", "opponent_1", "opponent_2"]))
        resource_keys = tuple(obs.get("resource_keys", ["gpu", "cpu", "memory"]))
        total_pool = int(float(obs.get("total_pool", 100.0)))

        def _generate_once() -> str:
            prompt_local = _build_prompt_from_obs(obs, round_num, difficulty, max_rounds)
            inputs_local = tokenizer(prompt_local, return_tensors="pt").to(model.device)
            with torch.no_grad():
                outputs_local = model.generate(
                    **inputs_local,
                    max_new_tokens=64,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id,
                )
            raw_local = tokenizer.decode(
                outputs_local[0][inputs_local["input_ids"].shape[1]:],
                skip_special_tokens=True,
            ).strip()
            print(f"MODEL OUTPUT [R{round_num}]: {repr(raw_local[:220])}")
            return raw_local

        prompt = _build_prompt_from_obs(obs, round_num, difficulty, max_rounds)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw_text = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        ).strip()
        print(f"MODEL OUTPUT [R{round_num}]: {repr(raw_text[:220])}")
        if last_raw_output is not None and raw_text == last_raw_output:
            print(f"WARNING [R{round_num}]: model output repeated exactly from previous step.")
        last_raw_output = raw_text

        action = validate_and_fix_proposal(
            clean_action(raw_text),
            total_pool=total_pool,
            agent_ids=agent_ids,
            resource_keys=resource_keys,
        )

        # Strict validator: malformed proposals are regenerated before env.step().
        if action.lower().startswith("propose:") and not proposal_has_all_agents(
            action, agent_ids=agent_ids, resource_keys=resource_keys
        ):
            regenerated = False
            for attempt in range(2):
                regen_raw = _generate_once()
                candidate = validate_and_fix_proposal(
                    clean_action(regen_raw),
                    total_pool=total_pool,
                    agent_ids=agent_ids,
                    resource_keys=resource_keys,
                )
                if proposal_has_all_agents(
                    candidate, agent_ids=agent_ids, resource_keys=resource_keys
                ):
                    action = candidate
                    regenerated = True
                    print(f"REGENERATED VALID PROPOSAL [R{round_num}] on attempt {attempt + 1}")
                    break
            if not regenerated:
                # Make fallback explicit and deterministic; env will also penalize fallback use.
                action = validate_and_fix_proposal(
                    _build_equal_split_action(total_pool=float(total_pool), agent_ids=list(agent_ids)),
                    total_pool=total_pool,
                    agent_ids=agent_ids,
                    resource_keys=resource_keys,
                )
                print(f"FORCED FALLBACK PROPOSAL [R{round_num}] -> {action}")
        history = obs.get("conversation_history", [])
        opp_status = _recent_opponent_status(history)
        last_response = str(obs.get("last_opponent_response", "")).lower()
        last_proposal = str(obs.get("last_proposal", ""))
        threshold = 4.0 if difficulty == "easy" else 5.0
        all_accepted = opp_status["opponent_1"] == "accept" and opp_status["opponent_2"] == "accept"
        opp_util_ok = float(obs.get("last_opponent_utility", 0.0)) >= threshold
        if action.upper().startswith("ACCEPT: YES") and not (all_accepted or opp_util_ok):
            # Invalid accept is blocked: must propose a better deal.
            fallback = last_proposal if isinstance(last_proposal, str) else ""
            if fallback:
                action = validate_and_fix_proposal(f"PROPOSE: {fallback}")
            else:
                action = _build_equal_split_action()
            print(f"BLOCKED INVALID ACCEPT [R{round_num}] -> {action}")

        repeated = action.lower().startswith("propose:") and last_proposal and action.replace("PROPOSE: ", "").strip() == last_proposal.strip()
        if last_response in {"rejected", "partial_accept", "repeated_proposal"} or repeated:
            rejecting = None
            if opp_status["opponent_1"] == "reject":
                rejecting = "opponent_1"
            elif opp_status["opponent_2"] == "reject":
                rejecting = "opponent_2"
            if rejecting:
                action = _concede_to_rejecting_opponent(action, rejecting, shift=2 + min(round_num // 2, 3))
                print(f"ADAPTED ACTION [R{round_num}] targeted to {rejecting}: {action}")
            else:
                action = _shift_toward_opponents(action, shift=1 + min(round_num // 2, 3))
                print(f"ADAPTED ACTION [R{round_num}] after rejection/repetition: {action}")
        return action

    return _policy


def load_model_policy(
    checkpoint_dir: str,
    difficulty: str,
    max_rounds: int,
    base_model: Optional[str] = None,
) -> Callable[[Dict[str, Any], int], str]:
    """Load a trained model checkpoint and return evaluation policy."""
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Unsloth is required for model evaluation. Install with: "
            "pip install \"unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git\""
        ) from exc

    model_name = checkpoint_dir if base_model is None else base_model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=1024,
        load_in_4bit=True,
        dtype=None,
    )
    if base_model is not None:
        model.load_adapter(checkpoint_dir)
    model = FastLanguageModel.for_inference(model)
    return make_model_policy(model, tokenizer, difficulty, max_rounds)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(
    env: ComputeBazaarEnv,
    difficulty: str = "hard",
    seed: int | None = None,
    policy: Optional[Callable[[Dict[str, Any], int], str]] = None,
) -> Dict[str, Any]:
    """Run a single episode and return final metrics.

    Args:
        env: An instantiated ComputeBazaarEnv.
        difficulty: Curriculum difficulty ("easy" or "hard").
        seed: Optional per-episode seed for reproducibility.
        policy: Optional callable ``(obs, round_num) -> action_str``. Defaults
            to :func:`baseline_policy` when not provided.

    Returns:
        Metrics dict with keys: total_reward, utility_achieved, rounds_used,
        success, efficiency_bonus, sparse_bonus, oversight_queries, terminated.
    """
    _policy = policy if policy is not None else baseline_policy
    obs, _ = env.reset(seed=seed, options={"difficulty": difficulty})
    total_reward = 0.0
    terminated = truncated = False
    info: Dict[str, Any] = {}

    round_num = 0
    while not terminated and not truncated:
        round_num += 1
        action = _policy(obs, round_num)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

    return {
        "total_reward": round(total_reward, 4),
        "utility_achieved": info.get("utility_achieved", 0.0),
        "rounds_used": info.get("rounds_used", round_num),
        "success": info.get("success", False),
        "efficiency_bonus": info.get("efficiency_bonus", 0.0),
        "sparse_bonus": info.get("sparse_bonus", 0.0),
        "oversight_queries": info.get("oversight_queries", 0),
        "terminated": terminated,
    }


def run_rule_baseline_metrics(
    num_episodes: int = 10,
    difficulty: str = "hard",
    max_rounds: int = 12,
    seed: int = 42,
    num_opponents: int = 2,
) -> Dict[str, Any]:
    """Aggregate env metrics for :func:`baseline_policy` (fixed equal-split rule).

    Uses the same per-episode seeds as model eval in ``train.py``
    (``seed + ep`` for ``ep`` in ``1 .. num_episodes``) so bars are comparable.

    Returns:
        Dict with ``success_rate``, ``avg_reward``, ``avg_utility``, ``avg_rounds``,
        plus ``policy`` (short name string) for logging and plots.
    """
    env = ComputeBazaarEnv(
        max_rounds=max_rounds,
        seed=seed,
        agent_ids=build_agent_ids(num_opponents),
    )
    results: List[Dict[str, Any]] = []
    for ep in range(1, num_episodes + 1):
        metrics = run_episode(
            env, difficulty=difficulty, seed=seed + ep, policy=baseline_policy
        )
        results.append(metrics)
    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_reward": statistics.mean(r["total_reward"] for r in results),
        "avg_utility": statistics.mean(r["utility_achieved"] for r in results),
        "avg_rounds": statistics.mean(r["rounds_used"] for r in results),
        "policy": "baseline_equal_split",
    }


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def evaluate(
    episodes: int = 10,
    difficulty: str = "hard",
    max_rounds: int = 12,
    seed: int | None = None,
    policy: Optional[Callable[[Dict[str, Any], int], str]] = None,
    policy_name: str = "strategic",
    num_opponents: int = 2,
) -> None:
    """Run the full evaluation loop and print per-episode and aggregate stats.

    Args:
        episodes: Number of episodes to run.
        difficulty: Curriculum difficulty ("easy" or "hard").
        max_rounds: Maximum rounds per episode.
        seed: Optional base seed; per-episode seeds are derived from it.
    """
    env = ComputeBazaarEnv(
        max_rounds=max_rounds,
        seed=seed,
        agent_ids=build_agent_ids(num_opponents),
    )

    results: List[Dict[str, Any]] = []

    selected_policy = policy if policy is not None else strategic_baseline_policy

    print(f"\n{'=' * 60}")
    print(f"  Compute Allocation Bazaar — Evaluation ({episodes} episodes)")
    print(
        f"  Policy: {policy_name} | Difficulty: {difficulty} | "
        f"max_rounds: {max_rounds} | opponents: {num_opponents}"
    )
    print(f"{'=' * 60}\n")
    print(f"{'Ep':>3}  {'Reward':>8}  {'Utility':>8}  {'Rounds':>6}  {'Success':>7}  {'EfficBonus':>10}  {'SparseBonus':>11}")
    print(f"{'-' * 60}")

    for ep in range(1, episodes + 1):
        ep_seed = None if seed is None else seed + ep
        metrics = run_episode(env, difficulty=difficulty, seed=ep_seed, policy=selected_policy)
        results.append(metrics)
        success_str = "Y" if metrics["success"] else "N"
        print(
            f"{ep:>3}  "
            f"{metrics['total_reward']:>8.3f}  "
            f"{metrics['utility_achieved']:>8.4f}  "
            f"{metrics['rounds_used']:>6}  "
            f"{success_str:>7}  "
            f"{metrics['efficiency_bonus']:>10.1f}  "
            f"{metrics['sparse_bonus']:>11.1f}"
        )

    # ---------- aggregate stats ----------
    rewards = [r["total_reward"] for r in results]
    utilities = [r["utility_achieved"] for r in results]
    rounds_list = [r["rounds_used"] for r in results]
    success_rate = sum(1 for r in results if r["success"]) / episodes

    print(f"\n{'=' * 60}")
    print("  Aggregate Metrics")
    print(f"{'=' * 60}")
    print(f"  Success rate:          {success_rate * 100:.1f}%  ({sum(r['success'] for r in results)}/{episodes})")
    print(f"  Avg total reward:      {statistics.mean(rewards):.3f}  (std: {statistics.stdev(rewards) if episodes > 1 else 0:.3f})")
    print(f"  Avg utility achieved:  {statistics.mean(utilities):.4f}")
    print(f"  Avg rounds used:       {statistics.mean(rounds_list):.1f}")
    print(f"  Min / Max reward:      {min(rewards):.3f} / {max(rewards):.3f}")
    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Compute Allocation Bazaar agents.")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--difficulty", choices=["easy", "hard"], default="hard", help="Curriculum difficulty.")
    parser.add_argument("--max-rounds", type=int, default=12, dest="max_rounds", help="Max rounds per episode.")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed.")
    parser.add_argument(
        "--num-opponents",
        type=int,
        default=2,
        dest="num_opponents",
        help="Number of opponents (learner + N opponents total agents).",
    )
    parser.add_argument(
        "--policy",
        choices=["baseline", "strategic", "model"],
        default="strategic",
        help="Policy used for evaluation.",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline, strategic, and model (if checkpoint provided).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        dest="checkpoint_dir",
        help="Checkpoint path for model policy evaluation.",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        dest="base_model",
        help="Optional base model when loading LoRA adapter from checkpoint-dir.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.compare:
        print("Baseline:")
        evaluate(
            episodes=args.episodes,
            difficulty=args.difficulty,
            max_rounds=args.max_rounds,
            seed=args.seed,
            policy=baseline_policy,
            policy_name="baseline",
            num_opponents=args.num_opponents,
        )
        print("Strategic:")
        evaluate(
            episodes=args.episodes,
            difficulty=args.difficulty,
            max_rounds=args.max_rounds,
            seed=args.seed,
            policy=strategic_baseline_policy,
            policy_name="strategic",
            num_opponents=args.num_opponents,
        )
        if args.checkpoint_dir:
            print("Model:")
            model_policy = load_model_policy(
                checkpoint_dir=args.checkpoint_dir,
                difficulty=args.difficulty,
                max_rounds=args.max_rounds,
                base_model=args.base_model,
            )
            evaluate(
                episodes=args.episodes,
                difficulty=args.difficulty,
                max_rounds=args.max_rounds,
                seed=args.seed,
                policy=model_policy,
                policy_name="model",
                num_opponents=args.num_opponents,
            )
        else:
            print("Model: skipped (provide --checkpoint-dir)")
    else:
        policy_fn = baseline_policy
        if args.policy == "strategic":
            policy_fn = strategic_baseline_policy
        elif args.policy == "model":
            if not args.checkpoint_dir:
                raise ValueError("--checkpoint-dir is required when --policy model")
            policy_fn = load_model_policy(
                checkpoint_dir=args.checkpoint_dir,
                difficulty=args.difficulty,
                max_rounds=args.max_rounds,
                base_model=args.base_model,
            )

        evaluate(
            episodes=args.episodes,
            difficulty=args.difficulty,
            max_rounds=args.max_rounds,
            seed=args.seed,
            policy=policy_fn,
            policy_name=args.policy,
            num_opponents=args.num_opponents,
        )
