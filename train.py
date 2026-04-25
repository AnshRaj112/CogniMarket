"""Training script for Compute Allocation Bazaar using Unsloth + TRL (GRPO).

This script is designed to run in Google Colab (free tier) with Unsloth
installed. It finetunes a small LLM (e.g., Llama-3 8B or Phi-3 Mini) using
Group Relative Policy Optimization (GRPO) so the model learns to negotiate
compute allocations efficiently.

Setup (run once in Colab):
    !pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    !pip install trl peft accelerate bitsandbytes datasets

Usage:
    python train.py              # trains with default hyperparameters
    python train.py --episodes 500 --difficulty easy
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any, Dict, List

from compute_bazaar_env import (
    ComputeBazaarEnv,
    RESOURCE_KEYS,
    AGENT_IDS,
    clean_action,
    normalize_agent_names,
    validate_and_fix_proposal,
)
from evaluate import run_episode, run_rule_baseline_metrics
from reward import NO_DEAL_PENALTY, calculate_proposal_reward


# ---------------------------------------------------------------------------
# Training hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit" # 1.5B model fits perfectly in T4 VRAM for GRPO.
MAX_SEQ_LEN = 1024
LORA_RANK = 16
BATCH_SIZE = 1          # Micro-batch size
GRAD_ACCUM = 16         # Effective batch size of 16
LEARNING_RATE = 2e-5
EPISODES_PER_EPOCH = 50
NUM_EPOCHS = 10
GRPO_CLIP_RATIO = 0.2   # PPO/GRPO clipping epsilon.
KL_COEFF = 0.01         # Reduced from 0.02 to allow more exploration (anti-collapse).
SAVE_DIR = "./checkpoints"


# ---------------------------------------------------------------------------
# Prompt building helper (leverages prompts.py)
# ---------------------------------------------------------------------------

def _build_learner_prompt(obs: Dict[str, Any], round_num: int, difficulty: str, max_rounds: int) -> str:
    """Format observation dict into a learner system prompt string."""
    # Local import to allow train.py to run even if prompts.py had an error.
    from prompts import build_learner_hint  # type: ignore

    rounds_remaining = max_rounds - round_num
    return build_learner_hint(
        utility_vector=obs["private_utility"],
        conversation_history=obs["conversation_history"],
        remaining_pool=obs["remaining_compute_pool"],
        rounds_remaining=rounds_remaining,
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# Action validation layer (CRITICAL -- sits between model output and env)
# ---------------------------------------------------------------------------
#
# The canonical clean_action() and validate_and_fix_proposal() functions
# now live in compute_bazaar_env.py and are imported above.  This ensures
# the EXACT SAME validation pipeline runs in:
#   1. Model evaluation policy (_make_model_policy)
#   2. GRPO reward computation (outcome_reward_fn)
#   3. Environment step() itself (defense-in-depth)
#
# The pipeline is:  raw_output → clean_action() → validate_and_fix_proposal() → env.step()
#

FALLBACK_ACTION = "ACCEPT: NO"


# ---------------------------------------------------------------------------
# Model-based policy helper (used for post-training evaluation)
# ---------------------------------------------------------------------------

def _make_model_policy(model, tokenizer, difficulty: str, max_rounds: int):
    """Return a policy function that generates actions using the trained model.

    The returned callable has the same signature as ``baseline_policy``
    (``(obs, round_num) -> str``) and can be passed directly to
    :func:`evaluate.run_episode` as its ``policy`` argument.

    Args:
        model: A Hugging Face / Unsloth model in inference mode.
        tokenizer: Corresponding tokenizer.
        difficulty: Curriculum difficulty string forwarded to the prompt builder.
        max_rounds: Maximum rounds forwarded to the prompt builder.

    Returns:
        A callable ``(obs: dict, round_num: int) -> str``.
    """
    import torch  # type: ignore

    def _recent_opponent_status(history: List[str]) -> Dict[str, str]:
        status = {"opponent_1": "unknown", "opponent_2": "unknown"}
        for turn in reversed(history):
            lower = turn.lower()
            if lower.startswith("opponent_1:") and status["opponent_1"] == "unknown":
                status["opponent_1"] = "accept" if ("accept" in lower and "reject" not in lower) else ("reject" if "reject" in lower else "unknown")
            elif lower.startswith("opponent_2:") and status["opponent_2"] == "unknown":
                status["opponent_2"] = "accept" if ("accept" in lower and "reject" not in lower) else ("reject" if "reject" in lower else "unknown")
            if status["opponent_1"] != "unknown" and status["opponent_2"] != "unknown":
                break
        return status

    def _targeted_concession(action: str, rejecting_opponent: str, shift: int = 3) -> str:
        if not action.lower().startswith("propose:"):
            return action
        import re
        pattern = re.compile(
            r"(learner|opponent_1|opponent_2)\s*:\s*gpu\s*(\d+(?:\.\d+)?)\s*cpu\s*(\d+(?:\.\d+)?)\s*memory\s*(\d+(?:\.\d+)?)",
            re.IGNORECASE,
        )
        matches = pattern.findall(action)
        if len(matches) < 3:
            return action
        alloc: Dict[str, Dict[str, int]] = {}
        for agent, gpu, cpu, memory in matches[:3]:
            alloc[agent.lower()] = {"gpu": int(float(gpu)), "cpu": int(float(cpu)), "memory": int(float(memory))}
        other = "opponent_2" if rejecting_opponent == "opponent_1" else "opponent_1"
        for res in ("gpu", "cpu", "memory"):
            take_l = min(shift, alloc["learner"][res])
            take_o = min(max(1, shift // 2), alloc[other][res])
            alloc["learner"][res] -= take_l
            alloc[other][res] -= take_o
            alloc[rejecting_opponent][res] += take_l + take_o
        adapted = (
            "PROPOSE: "
            f"learner: gpu {alloc['learner']['gpu']} cpu {alloc['learner']['cpu']} memory {alloc['learner']['memory']}; "
            f"opponent_1: gpu {alloc['opponent_1']['gpu']} cpu {alloc['opponent_1']['cpu']} memory {alloc['opponent_1']['memory']}; "
            f"opponent_2: gpu {alloc['opponent_2']['gpu']} cpu {alloc['opponent_2']['cpu']} memory {alloc['opponent_2']['memory']}"
        )
        return validate_and_fix_proposal(adapted)

    def _policy(obs: Dict[str, Any], round_num: int) -> str:
        prompt = _build_learner_prompt(obs, round_num, difficulty, max_rounds)
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

        print(f"\n{'='*60}")
        print(f"MODEL OUTPUT [R{round_num}]: {repr(raw_text[:220])}")

        # FIX #2, #3, #5: Unified pipeline — clean → validate → ready for env
        cleaned = clean_action(raw_text)
        validated = validate_and_fix_proposal(cleaned)
        history = obs.get("conversation_history", [])
        opp_status = _recent_opponent_status(history)
        last_proposal = str(obs.get("last_proposal", ""))
        threshold = 4.0 if difficulty == "easy" else 5.0
        all_accepted = opp_status["opponent_1"] == "accept" and opp_status["opponent_2"] == "accept"
        opp_util_ok = float(obs.get("last_opponent_utility", 0.0)) >= threshold
        if validated.upper().startswith("ACCEPT: YES") and not (all_accepted or opp_util_ok):
            validated = (
                validate_and_fix_proposal(f"PROPOSE: {last_proposal}")
                if last_proposal
                else validate_and_fix_proposal("PROPOSE: learner: gpu 34 cpu 34 memory 34; opponent_1: gpu 33 cpu 33 memory 33; opponent_2: gpu 33 cpu 33 memory 33")
            )
        if obs.get("last_opponent_response", "") in {"partial_accept", "rejected", "repeated_proposal"}:
            rejecting = "opponent_1" if opp_status["opponent_1"] == "reject" else ("opponent_2" if opp_status["opponent_2"] == "reject" else None)
            if rejecting:
                validated = _targeted_concession(validated, rejecting, shift=2 + min(round_num // 2, 3))

        print(f"DEBUG [Model R{round_num}] cleaned: {cleaned}")
        print(f"DEBUG [Model R{round_num}] validated: {validated}")
        print(f"{'='*60}")

        return validated

    return _policy


def run_eval_suite(model, tokenizer, difficulty: str, max_rounds: int, num_episodes: int = 5, seed: int = 42) -> Dict[str, float]:
    """Run a suite of evaluation episodes and return aggregated stats.

    Includes diagnostics (Component 6): tracks unique proposals, acceptance
    rate, and flags policy collapse when >80% of proposals are identical.
    """
    from unsloth import FastLanguageModel
    from evaluate import run_episode
    import statistics

    # Ensure model is in inference mode
    model = FastLanguageModel.for_inference(model)
    policy = _make_model_policy(model, tokenizer, difficulty, max_rounds)
    env = ComputeBazaarEnv(max_rounds=max_rounds)
    
    results = []
    all_proposals = []  # Track for collapse detection
    for ep in range(1, num_episodes + 1):
        print(f"\n  --- Eval Episode {ep}/{num_episodes} ---")
        metrics = run_episode(env, difficulty=difficulty, seed=seed + ep, policy=policy)
        results.append(metrics)
        success_str = "SUCCESS" if metrics["success"] else "FAIL"
        print(
            f"  [{success_str}] reward={metrics['total_reward']:.2f}, "
            f"utility={metrics['utility_achieved']:.2f}, "
            f"rounds={metrics['rounds_used']}, "
            f"terminated={metrics.get('terminated', '?')}"
        )
        # Collect proposals from history for collapse detection
        for turn in env.history:
            if turn.startswith("learner:") and "gpu" in turn.lower():
                all_proposals.append(turn)

    # --- Diagnostics (Component 6) ---
    unique_proposals = len(set(all_proposals))
    total_proposals = len(all_proposals)
    collapse_ratio = (total_proposals - unique_proposals) / max(total_proposals, 1)
    policy_collapsed = collapse_ratio > 0.80

    print(f"\n  --- Diagnostics ---")
    print(f"  Total proposals: {total_proposals}")
    print(f"  Unique proposals: {unique_proposals}")
    print(f"  Collapse ratio: {collapse_ratio:.1%}")
    if policy_collapsed:
        print(f"  WARNING: Policy collapse detected! >80% identical proposals.")
    else:
        print(f"  Policy diversity: OK")
    
    return {
        "success_rate": sum(r["success"] for r in results) / len(results),
        "avg_reward": statistics.mean(r["total_reward"] for r in results),
        "avg_utility": statistics.mean(r["utility_achieved"] for r in results),
        "avg_rounds": statistics.mean(r["rounds_used"] for r in results),
        "unique_proposals": unique_proposals,
        "total_proposals": total_proposals,
        "policy_collapsed": policy_collapsed,
    }


# ---------------------------------------------------------------------------
# Reward function for GRPO (returns list[float] given list[str] completions)
# ---------------------------------------------------------------------------

def make_reward_fn(difficulty: str = "hard", max_rounds: int = 12):
    """Return a reward function compatible with TRL's GRPOTrainer.

    TRL's GRPO API expects a callable:
        reward_fn(prompts, completions, **kwargs) -> list[float]

    For each completion, we restore the environment state associated with the
    prompt context, apply the completion as a single learner action, and use
    the resulting step reward as the scalar reward.

    Args:
        difficulty: Curriculum difficulty.
        max_rounds: Maximum rounds per evaluation.

    Returns:
        A reward function suitable for ``GRPOConfig``.
    """

def _restore_env_from_state(env: ComputeBazaarEnv, env_state: Dict[str, Any], difficulty: str = "hard") -> None:
    """Restore an environment so reward evaluation matches the prompt context.

    The training pipeline should pass one serialized state per prompt via
    ``kwargs["env_states"]``. Supported keys:
        - ``reset_options``: options to pass to ``env.reset(...)``
        - any additional environment attributes to restore directly
    """
    reset_options = env_state.get("reset_options")
    if reset_options is None:
        reset_options = {"difficulty": env_state.get("difficulty", difficulty)}
    env.reset(options=reset_options)

    for attr_name, attr_value in env_state.items():
        if attr_name == "reset_options":
            continue
        
        # Reconstruct DealState from dictionary if needed
        if attr_name == "_deal" and attr_value is not None and isinstance(attr_value, dict):
            from compute_bazaar_env import DealState
            attr_value = DealState(
                proposal=attr_value["proposal"],
                accepted_by=set(attr_value["accepted_by"]),
                proposer=attr_value["proposer"]
            )
            
        setattr(env, attr_name, attr_value)

def _get_env_state(env: ComputeBazaarEnv) -> Dict[str, Any]:
    """Capture environment state in a JSON-serializable format for PyArrow/Datasets."""
    deal_serialised = None
    if env._deal is not None:
        deal_serialised = {
            "proposal": env._deal.proposal,
            "accepted_by": list(env._deal.accepted_by),
            "proposer": env._deal.proposer
        }

    return {
        "utilities": env.utilities,
        "history": list(env.history),
        "rounds_used": env.rounds_used,
        "_difficulty": env._difficulty,
        "_terminated": env._terminated,
        "_truncated": env._truncated,
        "_deal": deal_serialised,
    }

def make_reward_fn(difficulty: str, max_rounds: int):
    """Return a list of reward functions for GRPOTrainer.

    Three reward signals:
    1. outcome_reward_fn: Environment step reward (terminal deal outcome)
    2. format_reward_fn: Structural format compliance
    3. proposal_quality_reward_fn: Dense per-proposal feedback (Component 5)
    """
    from reward import calculate_format_reward, calculate_proposal_reward

    def outcome_reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        env_states = kwargs.get("env_states")
        if env_states is None:
            return [float(NO_DEAL_PENALTY)] * len(completions)

        rewards: List[float] = []
        for completion, env_state in zip(completions, env_states):
            eval_env = ComputeBazaarEnv(max_rounds=max_rounds)
            _restore_env_from_state(eval_env, env_state)
            try:
                cleaned = clean_action(completion)
                validated = validate_and_fix_proposal(cleaned)
                _, reward, _, _, _ = eval_env.step(validated)
            except (ValueError, RuntimeError):
                reward = NO_DEAL_PENALTY
            rewards.append(float(reward))
        return rewards

    def format_reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        return [calculate_format_reward(c) for c in completions]

    def proposal_quality_reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Dense per-proposal feedback: proximity, improvement, stagnation, diversity."""
        env_states = kwargs.get("env_states")
        if env_states is None:
            return [0.0] * len(completions)

        rewards: List[float] = []
        for completion, env_state in zip(completions, env_states):
            cleaned = clean_action(completion)
            validated = validate_and_fix_proposal(cleaned)
            r = calculate_proposal_reward(
                completion=validated,
                utilities=env_state.get("utilities"),
                history=env_state.get("history", []),
                difficulty=env_state.get("_difficulty", difficulty),
            )
            rewards.append(float(r))
        return rewards

    return [outcome_reward_fn, format_reward_fn, proposal_quality_reward_fn]


# ---------------------------------------------------------------------------
# Dataset builder — converts episodes into (prompt, completion) pairs
# ---------------------------------------------------------------------------

def build_dataset(
    difficulty: str = "hard",
    num_episodes: int = 200,
    max_rounds: int = 12,
    seed: int | None = 42,
    curriculum_window: int = 50,
    curriculum_promote_threshold: float = 0.70,
    curriculum_demote_threshold: float = 0.35,
) -> List[Dict[str, str]]:
    """Generate a dataset of (prompt, completion) pairs via self-play.

    Uses the ``strategic_baseline_policy`` instead of the naive equal-split
    baseline to generate DIVERSE demonstrations that break the equal-split
    prior. The strategic policy:
    - Proposes biased allocations favoring opponents
    - Progressively concedes when rejected
    - Varies allocations based on learner utility weights

    This gives GRPO a richer starting distribution to optimize from.

    Args:
        difficulty: Curriculum difficulty.
        num_episodes: Number of self-play episodes to roll out.
        max_rounds: Maximum rounds per episode.
        seed: Base random seed.

    Returns:
        A list of {"prompt": str, "completion": str} dicts.
    """
    from evaluate import strategic_baseline_policy  # type: ignore

    dataset: List[Dict[str, Any]] = []
    env = ComputeBazaarEnv(max_rounds=max_rounds, seed=seed)

    recent_successes: List[int] = []
    current_difficulty = "easy" if difficulty == "auto" else difficulty
    for i in range(num_episodes):
        ep_seed = None if seed is None else seed + i
        obs, _ = env.reset(seed=ep_seed, options={"difficulty": current_difficulty})
        round_num = 0
        terminated = truncated = False
        episode_success = False

        while not terminated and not truncated:
            round_num += 1
            # Capture state BEFORE the step to associate with the current prompt.
            state_snapshot = _get_env_state(env)
            
            prompt = _build_learner_prompt(obs, round_num, current_difficulty, max_rounds)
            action = strategic_baseline_policy(obs, round_num)
            
            obs, _, terminated, truncated, info = env.step(action)
            dataset.append({
                "prompt": prompt, 
                "completion": action,
                "env_states": state_snapshot
            })
            episode_success = bool(info.get("success", False))

        if difficulty == "auto":
            recent_successes.append(1 if episode_success else 0)
            recent_successes = recent_successes[-max(1, curriculum_window):]
            rolling = sum(recent_successes) / len(recent_successes)
            if current_difficulty == "easy" and len(recent_successes) >= curriculum_window and rolling >= curriculum_promote_threshold:
                current_difficulty = "hard"
                print(f"[curriculum] promote -> hard (rolling_success={rolling:.2%})")
            elif current_difficulty == "hard" and len(recent_successes) >= curriculum_window and rolling <= curriculum_demote_threshold:
                current_difficulty = "easy"
                print(f"[curriculum] demote -> easy (rolling_success={rolling:.2%})")

    return dataset


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    model_name: str = DEFAULT_MODEL,
    difficulty: str = "hard",
    episodes_per_epoch: int = EPISODES_PER_EPOCH,
    num_epochs: int = NUM_EPOCHS,
    max_steps: int = 75,
    max_rounds: int = 12,
    seed: int | None = 42,
    save_dir: str = SAVE_DIR,
    report_to: str = "none",
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    curriculum_window: int = 50,
    curriculum_promote_threshold: float = 0.70,
    curriculum_demote_threshold: float = 0.35,
) -> None:
    """Fine-tune a language model on compute negotiation using Unsloth + TRL GRPO.

    Args:
        model_name: Hugging Face model ID supported by Unsloth.
        difficulty: Curriculum difficulty.
        episodes_per_epoch: Episodes rolled out per epoch for data collection.
        num_epochs: Total training epochs.
        max_rounds: Max negotiation rounds per episode.
        seed: Random seed.
        save_dir: Directory to save LoRA checkpoints.
    """
    # ------------------------------------------------------------------
    # 1. Load model and tokenizer via Unsloth (4-bit NF4 quantisation)
    # ------------------------------------------------------------------
    try:
        from unsloth import FastLanguageModel  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Unsloth is not installed. "
            "Run: pip install 'unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git'"
        ) from exc

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,  # Auto-detect (bfloat16 on Ampere+, float16 on older GPUs).
    )

    # Clean Fix: ensure model and tokenizer use correct length/token limits
    model.config.max_length = None
    model.config.max_new_tokens = 256
    tokenizer.model_max_length = 1024

    # Advanced Fix: Patch generation globally to prevent silent fallback to 32k
    from transformers import GenerationConfig
    model.generation_config = GenerationConfig(
        max_new_tokens=256,
        max_length=None,
        do_sample=True,
        temperature=0.8,
        top_p=0.9,
    )

    # ------------------------------------------------------------------
    # 2. Attach LoRA adapters (PEFT)
    # ------------------------------------------------------------------
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's memory-efficient checkpointing.
        random_state=seed or 3407,
    )

    # ------------------------------------------------------------------
    # 3. Build initial cold-start dataset
    # ------------------------------------------------------------------
    print("Building initial dataset …")
    dataset_records = build_dataset(
        difficulty=difficulty,
        num_episodes=episodes_per_epoch * num_epochs,
        max_rounds=max_rounds,
        seed=seed,
        curriculum_window=curriculum_window,
        curriculum_promote_threshold=curriculum_promote_threshold,
        curriculum_demote_threshold=curriculum_demote_threshold,
    )
    print(f"  {len(dataset_records)} (prompt, completion) pairs generated.")

    # Convert to HuggingFace Dataset.
    try:
        from datasets import Dataset  # type: ignore
    except ImportError as exc:
        raise ImportError("Install 'datasets': pip install datasets") from exc

    hf_dataset = Dataset.from_list(dataset_records)

    # ------------------------------------------------------------------
    # 4. Configure GRPO trainer
    # ------------------------------------------------------------------
    try:
        import transformers
        if not hasattr(transformers.utils.hub, "TRANSFORMERS_CACHE"):
            # Hack to fix broken trl -> llm_blender module dependency
            transformers.utils.hub.TRANSFORMERS_CACHE = getattr(transformers.utils.hub, "HF_HUB_CACHE", None)
            
        from trl import GRPOConfig, GRPOTrainer  # type: ignore
    except ImportError as exc:
        raise ImportError("Install trl: pip install trl>=0.8.0") from exc

    reward_fns = make_reward_fn(difficulty=difficulty, max_rounds=max_rounds)

    grpo_config = GRPOConfig(
        output_dir=save_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        # TRL: generation_batch_size = micro_batch * world_size * steps_per_generation must be
        # divisible by num_generations. Default steps_per_generation == GRAD_ACCUM (16) with
        # num_generations=6 violates that (16 % 6 != 0). Use 12 so gen batch = 1*1*12 = 12.
        steps_per_generation=12,
        learning_rate=LEARNING_RATE,
        max_steps=max_steps,
        num_generations=6,       # Increased from 4 for better GRPO variance (anti-collapse).
        use_vllm=False,
        beta=0.01,               # Reduced from 0.04 to allow more exploration.
        optim="adamw_8bit",
        max_completion_length=80, # Slightly more room for proposals.
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=50,
        seed=seed or 42,
        fp16=True,
        report_to=report_to,
        remove_unused_columns=False,
    )

    if getattr(model, "warnings_issued", None) is None:
        model.warnings_issued = {}

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        processing_class=tokenizer,
        reward_funcs=reward_fns,
        train_dataset=hf_dataset,
    )

    # ------------------------------------------------------------------
    # 5. Pre-training evaluation
    # ------------------------------------------------------------------
    print("\nRunning Pre-training evaluation (Step 0) …")
    pre_eval = run_eval_suite(model, tokenizer, difficulty, max_rounds, num_episodes=5, seed=seed or 42)
    print(f"  Pre-train Success Rate: {pre_eval['success_rate']*100:.1f}%")

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print(f"\nStarting GRPO training for {num_epochs} epoch(s) …")
    model.train() # Ensure back in training mode
    trainer.train()

    # ------------------------------------------------------------------
    # 6. Save LoRA weights
    # ------------------------------------------------------------------
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel saved to {save_dir}")

    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--hub-model-id is required when --push-to-hub is enabled")
        print(f"Pushing adapter/tokenizer to Hub: {hub_model_id}")
        model.push_to_hub(hub_model_id)
        tokenizer.push_to_hub(hub_model_id)

    # ------------------------------------------------------------------
    # 8. Post-training evaluation & Comparison
    # ------------------------------------------------------------------
    print("\nRunning Post-training evaluation …")
    post_eval = run_eval_suite(model, tokenizer, difficulty, max_rounds, num_episodes=10, seed=seed or 42)

    print("\nRule baseline (equal-split policy, same seeds as post-eval) …")
    baseline_eval = run_rule_baseline_metrics(
        num_episodes=10,
        difficulty=difficulty,
        max_rounds=max_rounds,
        seed=seed or 42,
    )
    print(
        f"  Baseline success rate: {baseline_eval['success_rate']*100:.1f}% | "
        f"avg reward: {baseline_eval['avg_reward']:.2f} | "
        f"avg utility: {baseline_eval['avg_utility']:.3f}"
    )

    # ------------------------------------------------------------------
    # 9. Visualization & Final Summary
    # ------------------------------------------------------------------
    print("\nGenerating training reward curves …")
    try:
        from plot_training_rewards import plot_reward_curves

        progress_path = os.path.join(save_dir, "training_progress.json")
        with open(progress_path, "w", encoding="utf-8") as f:
            json.dump({"pre": pre_eval, "post": post_eval, "baseline": baseline_eval}, f, indent=2)
        print(f"  Eval summary saved to {progress_path}")

        plot_path = os.path.join(save_dir, "reward_plot.png")
        plot_reward_curves(
            trainer.state.log_history,
            Path(plot_path),
            pre_post_eval={"pre": pre_eval, "post": post_eval, "baseline": baseline_eval},
        )
        print(f"  Plot saved to {plot_path}")
    except ImportError as exc:
        print(f"  Skipping plot (install matplotlib and pandas): {exc}")
    except Exception as e:
        print(f"  Failed to generate plot: {e}")

    def fmt_delta(pre, post):
        delta = post - pre
        sign = "+" if delta >= 0 else ""
        return f"({sign}{delta:.2f})"

    print(f"\n{'='*60}")
    print(f"{'Metric':<20} | {'Baseline':<10} | {'Before':<10} | {'After':<10} | {'Δ vs base'}")
    print(f"{'-'*60}")
    print(
        f"{'Success Rate':<20} | {baseline_eval['success_rate']*100:>9.1f}% | "
        f"{pre_eval['success_rate']*100:>9.1f}% | {post_eval['success_rate']*100:>9.1f}% | "
        f"{fmt_delta(baseline_eval['success_rate']*100, post_eval['success_rate']*100)}"
    )
    print(
        f"{'Avg Reward':<20} | {baseline_eval['avg_reward']:>10.2f} | "
        f"{pre_eval['avg_reward']:>10.2f} | {post_eval['avg_reward']:>10.2f} | "
        f"{fmt_delta(baseline_eval['avg_reward'], post_eval['avg_reward'])}"
    )
    print(
        f"{'Avg Utility':<20} | {baseline_eval['avg_utility']:>10.3f} | "
        f"{pre_eval['avg_utility']:>10.3f} | {post_eval['avg_utility']:>10.3f} | "
        f"{fmt_delta(baseline_eval['avg_utility'], post_eval['avg_utility'])}"
    )
    print(
        f"{'Avg Rounds':<20} | {baseline_eval['avg_rounds']:>10.1f} | "
        f"{pre_eval['avg_rounds']:>10.1f} | {post_eval['avg_rounds']:>10.1f} | "
        f"{fmt_delta(baseline_eval['avg_rounds'], post_eval['avg_rounds'])}"
    )
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Compute Bazaar negotiation agent with Unsloth + TRL GRPO.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID.")
    parser.add_argument("--difficulty", choices=["easy", "hard", "auto"], default="hard")
    parser.add_argument("--episodes", type=int, default=EPISODES_PER_EPOCH, dest="episodes_per_epoch")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--max-steps", type=int, default=75, dest="max_steps")
    parser.add_argument("--max-rounds", type=int, default=12, dest="max_rounds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", default=SAVE_DIR, dest="save_dir")
    parser.add_argument("--report-to", choices=["none", "wandb", "tensorboard"], default="none", dest="report_to")
    parser.add_argument("--push-to-hub", action="store_true", dest="push_to_hub")
    parser.add_argument("--hub-model-id", default=None, dest="hub_model_id")
    parser.add_argument("--curriculum-window", type=int, default=50, dest="curriculum_window")
    parser.add_argument("--curriculum-promote-threshold", type=float, default=0.70, dest="curriculum_promote_threshold")
    parser.add_argument("--curriculum-demote-threshold", type=float, default=0.35, dest="curriculum_demote_threshold")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    train(
        model_name=args.model,
        difficulty=args.difficulty,
        episodes_per_epoch=args.episodes_per_epoch,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        max_rounds=args.max_rounds,
        seed=args.seed,
        save_dir=args.save_dir,
        report_to=args.report_to,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        curriculum_window=args.curriculum_window,
        curriculum_promote_threshold=args.curriculum_promote_threshold,
        curriculum_demote_threshold=args.curriculum_demote_threshold,
    )
