# CogniMarket — Compute Allocation Bazaar

A lightweight, Colab-friendly research framework for training language models to negotiate compute resource allocations through multi-agent reinforcement learning.

---

## Table of Contents

1. [Overview](#overview)
2. [Repository Structure](#repository-structure)
3. [The Environment — `compute_bazaar_env.py`](#the-environment)
   - [Scenario](#scenario)
   - [Agents](#agents)
   - [Resources & Pool](#resources--pool)
   - [Observation Space](#observation-space)
   - [Action Space](#action-space)
   - [Episode Lifecycle](#episode-lifecycle)
   - [Difficulty Curriculum](#difficulty-curriculum)
   - [Oversight Agent](#oversight-agent)
   - [Key Constants](#key-constants)
4. [Reward Design — `reward.py`](#reward-design)
   - [Reward Components](#reward-components)
   - [Utility Function](#utility-function)
   - [Standalone API](#standalone-api)
5. [System Prompts — `prompts.py`](#system-prompts)
   - [Opponent Prompt](#opponent-prompt)
   - [Oversight Prompt](#oversight-prompt)
   - [Learner Hint](#learner-hint)
6. [Evaluation — `evaluate.py`](#evaluation)
   - [Baseline Policy](#baseline-policy)
   - [Running the Evaluator](#running-the-evaluator)
   - [Sample Output](#sample-output)
7. [Training — `train.py`](#training)
   - [Architecture](#architecture)
   - [GRPO Reward Function](#grpo-reward-function)
   - [Cold-Start Dataset](#cold-start-dataset)
   - [Running Training](#running-training)
   - [Hyperparameters](#hyperparameters)
8. [Tests](#tests)
9. [Installation](#installation)
10. [Quick Start](#quick-start)
11. [Design Decisions](#design-decisions)
12. [Roadmap](#roadmap)

---

## Overview

**CogniMarket** frames compute resource allocation as a multi-agent negotiation game. Three agents — one trainable *learner* and two rule-based *opponents* — compete to divide a shared pool of GPU, CPU, and Memory units. Each agent has a private utility vector (unknown to the others) that determines how much it values each resource type. Agents negotiate in free-form text for up to 12 rounds; a deal closes only when all three accept the same proposal.

The goal is to train the learner agent to:
- Quickly infer what allocations the opponents will accept
- Formulate proposals that satisfy both opponents while maximising its own utility
- Know when to consult the oversight agent for neutral guidance
- Close good deals fast, earning efficiency bonuses

The entire stack — environment, reward function, system prompts, evaluation loop, and training script — is designed to run on a **free-tier Google Colab GPU** using [Unsloth](https://github.com/unslothai/unsloth) and [TRL](https://github.com/huggingface/trl).

---

## Repository Structure

```
CogniMarket/
|-- compute_bazaar_env.py   # Gymnasium-style negotiation environment
|-- reward.py               # Standalone reward calculation, utility, and dense proposal reward
|-- prompts.py              # System prompt templates with negotiation memory
|-- evaluate.py             # Evaluation loop with baseline + strategic policies
|-- train.py                # Unsloth + TRL GRPO fine-tuning with 3-reward pipeline
|-- test_fixes.py           # Integration tests for env/parsing fixes
|-- test_strategic.py       # Strategic baseline + dense reward tests
`-- tests/
    |-- test_compute_bazaar_env.py       # Environment unit tests
    `-- test_reward_prompts_evaluate.py  # Reward, prompt, and evaluation tests
```

---

## The Environment

**File:** `compute_bazaar_env.py`

### Scenario

Each episode simulates a negotiation session in a fictional "Compute Allocation Bazaar". Three agents must agree on how to split a shared pool of computing resources (GPU, CPU, Memory). Agents have hidden preferences — they do not know each other's utility weights — so every negotiation requires inferring what the others need from their responses.

### Agents

| ID | Role | Behaviour |
|----|------|-----------|
| `learner` | Trainable LLM-based agent | Controlled by `env.step(action)` |
| `opponent_1` | Rule-based opponent | Accepts proposals above a utility threshold |
| `opponent_2` | Rule-based opponent | Same rule; independent utility vector |

The two opponents are simulated internally by `_run_opponents()`. They evaluate each incoming proposal against their private utility vector and either `accept` or `reject, please improve my share.`

### Resources & Pool

There are three resource types, each with **100 units** total:

| Resource | Key |
|----------|-----|
| GPU | `"gpu"` |
| CPU | `"cpu"` |
| Memory | `"memory"` |

A proposal is *feasible* when the sum of all three agents' allocations does not exceed 100 for any single resource. Infeasible proposals are silently discarded by `_extract_allocation`.

### Observation Space

`env.reset()` and `env.step()` both return an observation dictionary:

```python
{
    "conversation_history": List[str],      # last 8 turns of dialogue
    "private_utility": List[float],         # learner's [gpu_w, cpu_w, memory_w] (sums to 1)
    "remaining_compute_pool": Dict[str, float],  # unallocated units per resource
}
```

### Action Space

Actions are free-form strings. The parser (`_parse_action`) recognises these patterns:

| Pattern | Action Type | Description |
|---------|-------------|-------------|
| Contains `"walk"` and `"away"` | `walk_away` | Terminate episode immediately (penalty) |
| Contains `"query_oversight"` or `"oversight"` | `query_oversight` | Request an oversight explanation |
| Contains `"accept"` | `accept` | Accept the active proposal |
| Contains `"reject"` | `reject` | Reject the active proposal |
| Contains `"counter"` + valid proposal | `counter_offer` | Replace active proposal |
| Valid proposal (no "counter") | `propose` | Put a new proposal on the table |
| Anything else | `message` | Free-form message (no mechanical effect) |

**Proposal format** (parsed via regex):
```
learner: gpu <N> cpu <N> memory <N>; opponent_1: gpu <N> cpu <N> memory <N>; opponent_2: gpu <N> cpu <N> memory <N>
```
Values can be integers or floats; all three agents must be present.

### Episode Lifecycle

```
reset() → obs
    │
    ▼
step(action)  ←──────────────────────────────┐
    │                                         │
    ├─ parse action                           │
    ├─ append to history                      │
    ├─ update DealState if proposal/accept    │
    ├─ _run_opponents() (auto-react)          │
    ├─ check if all 3 accepted → deal_closed  │
    │       └─ compute reward & terminate     │
    ├─ check max_rounds → truncate            │
    └─ return (obs, reward, terminated,       │
               truncated, info)  ─────────────┘
```

The `info` dict returned on each step contains:

```python
{
    "utility_achieved": float,   # learner's scaled utility [0, 15]
    "rounds_used": int,
    "success": bool,             # True when all 3 agents accepted (deal closed)
    "deal_closed": bool,         # Same as success
    "efficiency_bonus": float,   # 0, 2, or 4
    "sparse_bonus": float,       # 0 or 10
    "deal_completion_bonus": float,  # 0 or 3
    "oversight_queries": int,
}
```

### Difficulty Curriculum

`reset(options={"difficulty": "easy"|"hard"})`

| Mode | Opponent utility sampling |
|------|--------------------------|
| `"easy"` | Opponents' weights are learner's weights ± uniform noise in [−0.05, 0.05], then renormalised — preferences are nearly aligned, making mutually beneficial deals easy to find. |
| `"hard"` | All three agents' weights are sampled independently — preferences are likely conflicting, requiring more sophisticated negotiation. |

### Oversight Agent

When the learner sends an action containing `"oversight"` or `"query_oversight"`, the environment appends an explanation to the conversation history:

```
oversight: Current proposal is feasible; resource usage is {'gpu': 100.0, 'cpu': 100.0, 'memory': 100.0}.
```

The explanation reports feasibility and current resource totals, giving the learner a neutral factual signal. The number of queries is tracked in `info["oversight_queries"]`.

### Key Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `ROUND_PENALTY` | -0.8 | Per-round cost; discourages dragging out negotiations |
| `NO_DEAL_PENALTY` | -5.0 | Applied on walk-away or timeout (NOT on closed deals) |
| `DEAL_COMPLETION_BONUS` | +3.0 | Guaranteed positive reward for closing ANY deal |
| `FAST_DEAL_BONUS` | +4.0 | Deal closed in fewer than 6 rounds |
| `MID_DEAL_BONUS` | +2.0 | Deal closed in fewer than 10 rounds |
| `SPARSE_BONUS` | +10.0 | Utility >= 85% of maximum (sparse signal) |
| `SUCCESS_UTILITY_THRESHOLD` | 5.0 | Sparse bonus tracking only (NOT a success gate) |
| `MAX_UTILITY_SCALE` | 15.0 | Maximum possible utility score |
| `SPARSE_BONUS_THRESHOLD` | 0.85 | Fraction of MAX_UTILITY_SCALE for sparse bonus |
| `EASY_ACCEPTANCE_THRESHOLD` | 4.0 | Utility threshold for opponent acceptance (easy) |
| `HARD_ACCEPTANCE_THRESHOLD` | 5.5 | Utility threshold for opponent acceptance (hard) |

---

## Reward Design

**File:** `reward.py`

The reward function is implemented as a standalone pure function -- separate from the environment class -- so it can be imported directly in training callbacks, offline evaluation, and ablation studies without instantiating the environment.

### Reward Components (Terminal)

| Component | Formula | When Applied |
|-----------|---------|--------------|
| `round_penalty` | `-0.8 x rounds_used` | Every episode |
| `deal_completion_bonus` | +3 (guaranteed) | Deal closed |
| `utility_reward` | `utility(allocation, weights)` in [0, 15] | Deal closed |
| `efficiency_bonus` | +4 if rounds < 6, +2 if rounds < 10 | Deal closed |
| `sparse_bonus` | +10 if utility / 15 >= 0.85 | Deal closed with high utility |
| `no_deal_penalty` | -5 | No deal at all (timeout/walkaway) |
| `oversight_bonus` | +2.5 | Oversight gave accurate explanation (optional) |

> **Important:** `NO_DEAL_PENALTY` is ONLY applied when there is literally no deal (timeout or walkaway). Closed deals ALWAYS get positive reward via `DEAL_COMPLETION_BONUS`.

**Maximum achievable reward** (round 1 deal, perfect utility, sparse bonus):
```
-0.8 + 3.0 + 15.0 + 4.0 + 10.0 = 31.2
```

**Typical no-deal timeout** (12 rounds):
```
-0.8 x 12 - 5.0 = -14.6
```

### Dense Proposal Reward (Per-Step Learning Signal)

`calculate_proposal_reward()` provides directional feedback on every proposal, not just terminal outcomes:

| Signal | Value | Purpose |
|--------|-------|---------|
| `opponent_proximity` | `+2.0 x (opp_utility / threshold)` | Reward proposals that approach acceptance |
| `improvement_bonus` | `+1.5` | Proposal improved opponent utility vs previous |
| `stagnation_penalty` | `-2.0` | Proposal is identical to previous one |
| `self_sacrifice_penalty` | `-1.0` | Learner utility dropped below 3.0 unnecessarily |
| `diversity_bonus` | `+0.5` | Proposal differs from equal split |

### Utility Function

```
utility = (Σ allocation_fraction_i × weight_i) × 15
```

Where `allocation_fraction_i = allocation[resource_i] / 100.0`, and weights are the private utility vector summing to 1. This scales linearly from 0 (no allocation) to 15 (full allocation of the most-valued resource).

### Standalone API

```python
from reward import calculate_reward, calculate_utility, calculate_proposal_reward

# Compute utility alone
u = calculate_utility(
    allocation={"gpu": 60, "cpu": 20, "memory": 20},
    utility_vector=[0.8, 0.1, 0.1],
)
# -> ~9.3

# Compute full terminal reward with breakdown
reward, breakdown = calculate_reward(
    allocation={"gpu": 60, "cpu": 20, "memory": 20},
    utility_vector=[0.8, 0.1, 0.1],
    rounds_used=3,
    deal_closed=True,
    oversight_accurate=False,
)
# breakdown: {
#   "round_penalty": -2.4,
#   "deal_completion_bonus": 3.0,
#   "utility_reward": 9.3,
#   "efficiency_bonus": 4.0,
#   "sparse_bonus": 0.0,
#   "no_deal_penalty": 0.0,
#   "oversight_bonus": 0.0,
# }

# Compute dense per-proposal reward
proposal_reward = calculate_proposal_reward(
    completion="PROPOSE: learner: gpu 25 cpu 20 memory 20; opponent_1: gpu 40 cpu 40 memory 40; opponent_2: gpu 35 cpu 40 memory 40",
    utilities={"learner": [0.5, 0.3, 0.2], "opponent_1": [0.3, 0.4, 0.3], "opponent_2": [0.2, 0.3, 0.5]},
    difficulty="hard",
)
# -> ~2.65 (generous, diverse proposal near acceptance threshold)
```

---

## System Prompts

**File:** `prompts.py`

Three prompt-building functions cover every agent role. Each returns a formatted string ready to pass as a system message to any chat-capable LLM.

### Opponent Prompt

```python
from prompts import build_opponent_prompt

prompt = build_opponent_prompt(
    agent_id="opponent_1",
    utility_vector=[0.6, 0.3, 0.1],
    conversation_history=["learner: propose learner: gpu 50 …"],
    remaining_pool={"gpu": 100.0, "cpu": 100.0, "memory": 100.0},
)
```

The generated prompt:
- Tells the LLM it is `opponent_1` and states its private utility weights
- Explains the proposal format and negotiation rules
- Shows the current remaining pool and recent conversation
- Instructs strategic behaviour: push for preferred resources, reveal preferences only indirectly, compromise when rounds are running low

### Oversight Prompt

```python
from prompts import build_oversight_prompt

prompt = build_oversight_prompt(
    conversation_history=[...],
    current_proposal={
        "learner": {"gpu": 34.0, "cpu": 33.0, "memory": 33.0},
        "opponent_1": {"gpu": 33.0, "cpu": 34.0, "memory": 33.0},
        "opponent_2": {"gpu": 33.0, "cpu": 33.0, "memory": 34.0},
    },
    utilities=None,  # set to all agents' utility dicts for full-observability mode
)
```

The oversight agent is neutral and covers:
1. Feasibility of the current proposal
2. Which agent benefits most and why
3. Fairness relative to stated/inferred preferences
4. An actionable recommendation

When `utilities` is provided, the prompt operates in *full-observability mode* and can reason about true utilities.

### Learner Hint

```python
from prompts import build_learner_hint

hint = build_learner_hint(
    utility_vector=[0.5, 0.3, 0.2],
    conversation_history=["opponent_1: Can you propose…"],
    remaining_pool={"gpu": 100.0, "cpu": 100.0, "memory": 100.0},
    rounds_remaining=10,
    difficulty="hard",
)
```

This is a compact context header (not a full system prompt) prepended to the learner's observation. It includes:
- Difficulty and round count
- Urgency level (`LOW` / `MEDIUM` / `HIGH` based on rounds remaining)
- Private utility weights
- Remaining pool
- Recent conversation
- **Negotiation memory** — contextual strategy hints based on history:
  - Rejection-aware guidance ("Increase opponent shares")
  - Anti-repetition warnings ("Do NOT repeat your previous proposal")
  - Urgency-based strategy ("Make major concessions NOW")
  - Acceptance detection ("Consider ACCEPT: YES")
- The list of valid action formats and critical rules

---

## Evaluation

**File:** `evaluate.py`

### Baseline Policies

Two rule-based policies are provided:

#### `baseline_policy` (legacy)

Always proposes an **equal three-way split** on round 1 (34/33/33). On round 2+, mirrors any accept from opponents, otherwise repeats the equal split.

- `easy` mode: ~100% success rate
- `hard` mode: **0% success rate** (opponents need >= 5.5 utility; equal split gives ~4.95)

#### `strategic_baseline_policy` (new)

Diverse, utility-aware policy that breaks the equal-split bias:

- **Round 1:** Biased proposal favoring opponents (learner keeps ~28%), concentrates learner's share on highest-valued resource
- **Rounds 2+:** If opponents accepted, ACCEPT: YES. Otherwise, progressively concedes — giving more to opponents each rejected round
- **Varies allocations** across resources based on learner utility weights
- **Randomizes** opponent split ratios for dataset diversity

**Strategic baseline performance:**
- `easy` mode: 100% success rate
- `hard` mode: **~70% success rate** (up from 0% with equal split)

The strategic baseline is used for cold-start dataset generation in training.

### Running the Evaluator

```bash
# 10 episodes, hard difficulty (default)
python evaluate.py

# 20 episodes, easy curriculum
python evaluate.py --episodes 20 --difficulty easy

# Reproducible run with fixed seed
python evaluate.py --episodes 10 --difficulty hard --seed 42 --max-rounds 12
```

### Sample Output

```
============================================================
  Compute Allocation Bazaar — Evaluation (10 episodes)
  Difficulty: easy | max_rounds: 12
============================================================

 Ep    Reward   Utility  Rounds  Success  EfficBonus  SparseBonus
------------------------------------------------------------
  1     3.164    4.9644       1        ✗         4.0          0.0
  2     3.171    4.9713       1        ✗         4.0          0.0
  3     8.205    5.0049       1        ✓         4.0          0.0
  ...

============================================================
  Aggregate Metrics
============================================================
  Success rate:          33.3%  (3/10)
  Avg total reward:      4.847  (std: 2.908)
  Avg utility achieved:  4.9802
  Avg rounds used:       1.0
  Min / Max reward:      3.164 / 8.205
============================================================
```

Metrics reported per episode and in aggregate:

| Metric | Description |
|--------|-------------|
| `Reward` | Cumulative episode reward (sum across all `step()` calls) |
| `Utility` | Learner's utility from the final allocation |
| `Rounds` | Number of negotiation rounds used |
| `Success` | Y if deal closed (all agents accepted), N otherwise |
| `EfficBonus` | Efficiency bonus earned (+4 / +2 / 0) |
| `SparseBonus` | Sparse bonus earned (+10 / 0) |

---

## Training

**File:** `train.py`

### Architecture

The training script fine-tunes a small LLM to act as the learner agent using **Group Relative Policy Optimization (GRPO)** — an RL algorithm well-suited to text-action spaces with sparse rewards that does not require a reference model.

```
+-------------------------------------------------------------+
|                       train.py                               |
|                                                              |
|  1. Load base model via Unsloth (4-bit NF4 quantisation)    |
|  2. Attach LoRA adapters (r=16, targeting all attention      |
|     and MLP projection layers)                               |
|  3. Build cold-start dataset via strategic baseline policy   |
|  4. Configure TRL GRPOTrainer with 3-reward pipeline:        |
|     a. Outcome reward (env step result)                      |
|     b. Format reward (structural compliance)                 |
|     c. Proposal quality reward (dense per-step signal)       |
|  5. Train for N epochs                                       |
|  6. Save LoRA weights to ./checkpoints                       |
|  7. Run post-training evaluation with diagnostics            |
+-------------------------------------------------------------+
```

**Default base model:** `unsloth/Phi-3-mini-4k-instruct`
Fits within Colab free-tier VRAM with 4-bit quantisation and Unsloth's memory-efficient gradient checkpointing.

### GRPO Reward Functions

`make_reward_fn(difficulty, max_rounds)` returns a **list of three reward functions** compatible with TRL's `GRPOTrainer`:

| # | Function | Signal Type | Purpose |
|---|----------|-------------|---------|
| 1 | `outcome_reward_fn` | Terminal | Environment step reward (deal outcome) |
| 2 | `format_reward_fn` | Dense | Structural format compliance (+1 prefix, +1 agents, +1 resources) |
| 3 | `proposal_quality_reward_fn` | Dense | Per-proposal feedback (proximity, improvement, stagnation, diversity) |

The third reward function is the key addition — it provides directional learning signal so the agent knows *which direction* to adjust allocations, even when deals don't close.

### Cold-Start Dataset

`build_dataset()` generates `(prompt, completion)` pairs by rolling out the **strategic baseline policy** (not the naive equal-split). This generates diverse demonstrations that include:
- Biased allocations favoring opponents
- Progressive concession patterns
- Utility-weighted resource distribution
- Both accepted and rejected examples

This gives the model a rich starting distribution that breaks the equal-split prior.

Each record looks like:
```python
{
    "prompt": "[SYSTEM: Compute Allocation Bazaar | difficulty=hard | ...]\n...",
    "completion": "PROPOSE: learner: gpu 22 cpu 18 memory 18; opponent_1: gpu 40 cpu 45 memory 38; opponent_2: gpu 38 cpu 37 memory 44"
}
```

### Running Training

**Colab setup (run once):**
```bash
!pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install trl>=0.8.0 peft accelerate bitsandbytes datasets
```

**Then train:**
```bash
# Default: Phi-3 Mini, hard difficulty, 10 epochs
python train.py

# Curriculum: start easy, more epochs
python train.py --difficulty easy --epochs 20

# Custom model and checkpoint path
python train.py --model unsloth/Llama-3-8B-Instruct --save-dir ./my_checkpoints

# All options
python train.py \
    --model unsloth/Phi-3-mini-4k-instruct \
    --difficulty hard \
    --episodes 50 \
    --epochs 10 \
    --max-rounds 12 \
    --seed 42 \
    --save-dir ./checkpoints
```

### Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DEFAULT_MODEL` | `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | Base model (1.5B fits T4 VRAM for GRPO) |
| `MAX_SEQ_LEN` | 1024 | Maximum context length |
| `LORA_RANK` | 16 | LoRA rank (r); `lora_alpha = 32` |
| `BATCH_SIZE` | 1 | Per-device micro-batch size |
| `GRAD_ACCUM` | 16 | Gradient accumulation steps (effective batch = 16) |
| `LEARNING_RATE` | 2e-5 | AdamW learning rate |
| `EPISODES_PER_EPOCH` | 50 | Episodes used per training epoch |
| `NUM_EPOCHS` | 10 | Total fine-tuning epochs |
| `GRPO_CLIP_RATIO` | 0.2 | PPO/GRPO clipping epsilon |
| `KL_COEFF` | 0.01 | KL penalty (reduced from 0.02 for more exploration) |
| `num_generations` | 6 | G in GRPO (increased from 4 for anti-collapse) |
| `temperature` | 0.9 | Sampling temperature (increased from 0.7 for diversity) |
| `max_completion_length` | 80 | Max tokens per completion |

---

## Tests

**Files:** `tests/test_compute_bazaar_env.py`, `tests/test_reward_prompts_evaluate.py`

17 unit tests across four test classes, all passing:

```bash
python -m unittest discover -s tests -q
# Ran 17 tests in 0.001s  OK
```

### `TestComputeBazaarEnv`

| Test | What it checks |
|------|----------------|
| `test_reset_returns_expected_observation_shape` | obs dict has all 3 keys; utility vector has length 3; info has correct difficulty |
| `test_sampled_utilities_are_normalized_and_non_negative` | utility vectors sum to ~1.0 and are non-negative for both difficulties |
| `test_parse_allocation_and_successful_step` | valid proposal string parses correctly; step returns correct types |

### `TestCalculateUtility`

| Test | What it checks |
|------|----------------|
| `test_full_gpu_allocation_with_gpu_preference` | 100 % GPU allocation with weight 1.0 → utility = 15.0 |
| `test_zero_allocation_returns_zero` | empty allocation → 0.0 |
| `test_equal_split_with_uniform_weights` | ~33/33/33 split with uniform weights → ~5.0 |

### `TestCalculateReward`

| Test | What it checks |
|------|----------------|
| `test_successful_fast_deal_grants_all_bonuses` | fast deal with perfect utility triggers efficiency, deal completion, and sparse bonuses |
| `test_no_deal_applies_penalty` | no-deal episode applies `NO_DEAL_PENALTY` and zero utility |
| `test_suboptimal_deal_no_penalty` | deal with utility < 5.0 still gets completion bonus (no penalty) |
| `test_oversight_bonus_applied_when_accurate` | `oversight_accurate=True` adds `OVERSIGHT_ACCURACY_BONUS` |
| `test_reward_total_equals_sum_of_breakdown` | total reward equals the exact sum of all breakdown components |

### `TestPrompts`

| Test | What it checks |
|------|----------------|
| `test_opponent_prompt_contains_agent_id` | opponent prompt mentions the correct agent ID |
| `test_oversight_prompt_includes_proposal` | oversight prompt contains proposal and the word "feasible" |
| `test_learner_hint_contains_utility_and_rounds` | hint includes rounds_left and GPU weight values |

### `TestEvaluateLoop`

| Test | What it checks |
|------|----------------|
| `test_run_episode_returns_expected_keys` | episode metrics dict has all required keys |
| `test_evaluate_runs_without_error` | full 3-episode evaluation loop completes without exceptions |
| `test_baseline_policy_returns_equal_split_on_round1` | baseline produces correct action string on round 1 |

---

## Installation

**Minimal (environment + reward + prompts + evaluation only):**
```bash
# No mandatory external dependencies — falls back to plain object if neither
# openenv nor gymnasium is installed.
pip install gymnasium  # optional but recommended
```

**Full (including training):**
```bash
pip install gymnasium
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl>=0.8.0 peft accelerate bitsandbytes datasets
```

**Python version:** 3.10+ (uses `tuple[str, ...]` and `X | Y` union syntax)

---

## Docker Deployment

To run the CogniMarket Gradio UI in a lightweight Docker container:

1.  **Ensure you have a `.env` file** in the root directory with your API keys (GROQ_API_KEY, etc.).
2.  **Build and run using Docker Compose**:
    ```bash
    docker-compose up --build
    ```
3.  **Access the UI**:
    Open your browser and navigate to `http://localhost:7860`.

To stop the container:
```bash
docker-compose down
```

> **Note:** The Docker deployment is optimized for the **application UI**. Training (`train.py`) requires heavy ML dependencies and is recommended for Google Colab/GPU environments.

---

## Quick Start

```python
from compute_bazaar_env import ComputeBazaarEnv

env = ComputeBazaarEnv(max_rounds=12, seed=42)
obs, info = env.reset(options={"difficulty": "easy"})

print("My utility weights:", obs["private_utility"])
# e.g. [0.612, 0.254, 0.134]  → GPU is most valued

action = (
    "propose "
    "learner: gpu 50 cpu 25 memory 25; "
    "opponent_1: gpu 25 cpu 40 memory 35; "
    "opponent_2: gpu 25 cpu 35 memory 40"
)

obs, reward, terminated, truncated, info = env.step(action)
print(f"Reward: {reward:.2f} | Success: {info['success']} | Rounds: {info['rounds_used']}")
```

```python
# Run the built-in evaluation CLI
# python evaluate.py --episodes 10 --difficulty easy --seed 42
from evaluate import evaluate
evaluate(episodes=10, difficulty="easy", seed=42)
```

```python
# Compute reward breakdown offline
from reward import calculate_reward
reward, breakdown = calculate_reward(
    allocation={"gpu": 55, "cpu": 25, "memory": 20},
    utility_vector=[0.6, 0.2, 0.2],
    rounds_used=4,
    deal_closed=True,
)
print(breakdown)
```

---

## Design Decisions

**Why GRPO instead of PPO?**
GRPO does not require a reference model or value network, making it significantly more memory-efficient. For Colab free-tier (~15 GB VRAM), this is critical when the base model itself occupies most of that space.

**Why three reward functions?**
A single terminal reward is too sparse for the model to learn *which direction* to adjust proposals. The three-reward pipeline provides: (1) outcome reward for deal closure, (2) format compliance to learn the action syntax, and (3) dense per-proposal feedback so the agent knows whether it's getting closer to acceptance — even when deals don't close.

**Why a strategic baseline instead of equal-split?**
The equal-split baseline always proposes 34/33/33, which gives opponents utility ~4.95 — below the hard-mode threshold of 5.5. Training on this data teaches the model to repeat the same failing pattern. The strategic baseline generates diverse, opponent-favorable proposals that break this prior and give GRPO a richer exploration landscape.

**Why deal completion = success (not utility threshold)?**
The original design gated `success` on `utility >= 5.0`, which meant a deal where the learner compromised (utility=3.0) was penalized as a failure. This killed the RL learning signal — the agent was punished for doing exactly what it should (closing deals). Now any closed deal is a success, with utility serving as a continuous reward component rather than a binary gate.

**Why negotiation memory in prompts?**
LLMs have no built-in mechanism to track multi-turn negotiation dynamics. Injecting explicit strategy hints ("opponents rejected 3 times, increase their shares") gives the model actionable guidance at each step, preventing the common failure mode of repeating identical proposals.

**Why free-form text actions?**
LLMs naturally produce text. Forcing discrete action spaces would require an additional mapping layer and would prevent the model from asking questions, sending strategic messages, or querying oversight — all of which are part of the intended behaviour.

**Why a sparse reward bonus?**
The `SPARSE_BONUS` (+10 for utility >= 85%) creates a strong signal for genuinely good deals that the dense round-penalty alone would not differentiate from mediocre ones. This is a deliberate shaping choice to push the model beyond "good enough" allocations.

**Why an oversight agent?**
The oversight agent provides a mechanism for the learner to request a neutral, factual explanation mid-negotiation. This supports research into *process supervision* and *human-in-the-loop* patterns where an external authority can validate or guide agent behaviour.

**Why two difficulty levels?**
Curriculum learning is standard practice in RL. Starting with `easy` (aligned preferences) lets the model learn the proposal format and basic negotiation mechanics before facing `hard` (conflicting preferences) where strategy matters.

---

## Roadmap

- [x] Dense per-proposal reward function (opponent proximity, improvement, stagnation)
- [x] Strategic baseline policy for diverse cold-start dataset
- [x] Negotiation memory with rejection-aware strategy hints
- [x] GRPO anti-collapse config (more generations, lower KL, higher temperature)
- [x] Policy collapse diagnostics in evaluation suite
- [x] LLM-based opponents using `build_opponent_prompt` (with rule-based fallback when LLM/API is unavailable)
- [x] Curriculum scheduler that automatically transitions easy -> hard based on success rate
  - [x] Track rolling success rate over recent episodes (e.g., last 50)
  - [x] Promote to `hard` when success rate crosses threshold (e.g., >= 70%)
  - [x] Optionally demote back to `easy` if performance collapses
  - [x] Expose scheduler knobs as CLI args in `train.py`
- [x] W&B / TensorBoard integration in `train.py`
  - [x] Add optional `--report-to` flag (`none|wandb|tensorboard`)
  - [x] Pipe trainer metrics to selected backend via `GRPOConfig(report_to=...)`
- [x] Hugging Face Hub push for trained LoRA checkpoints
  - [x] Add optional `--push-to-hub` and `--hub-model-id`
  - [x] Upload adapter + tokenizer artifacts after training
- [x] OpenEnv-compatible registration (`openenv.register(...)`)
  - [x] Add `openenv.register(...)` entrypoint module (`openenv_registration.py`)
- [x] Support for more than 3 agents and configurable resource types
  - [x] Refactor hard-coded agent/resource dimensions into env config (`agent_ids`, `resource_keys`)
  - [x] Generalize allocation parsing/validation inside env methods for dynamic schemas
  - [x] Update utility sampling, pool tracking, and observation shapes for dynamic dimensions