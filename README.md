---
title: Cognimarket
emoji: 👀
colorFrom: red
colorTo: purple
sdk: docker
pinned: false
---

# CogniMarket — Compute Allocation Bazaar

An environment for training LLM agents to negotiate scarce compute under hidden preferences.

## Problem

LLMs are fluent, but they are often weak at multi-party bargaining when resources are limited and goals conflict.

CogniMarket targets this capability gap in a practical domain: **compute allocation**.  
If agents are going to run shared systems, they must negotiate trade-offs, not just generate plausible text.

## Environment

Each episode is an **N-agent (3 - 5)** negotiation:
- `learner` (1 trainable policy)
- `opponent_i` agents (`N-1` opponents, rule-based by default)

All agents negotiate allocation of GPU, CPU, and Memory from a shared pool.

The learner sees:
- recent conversation history,
- its private utility weights,
- remaining compute pool.

The learner acts in free text:
- propose or counter-offer allocations,
- accept/reject,
- query oversight,
- or walk away.

Reward is shaped for both quality and efficiency:
- utility from final allocation,
- deal completion bonus,
- speed/efficiency bonuses,
- sparse high-quality bonus,
- no-deal penalty.

Core files:
- Environment: [`compute_bazaar_env.py`](./compute_bazaar_env.py)
- Reward logic: [`reward.py`](./reward.py)
- Composable rubrics: [`openenv_rubrics.py`](./openenv_rubrics.py)
- Rubric evaluator: [`evaluate_openenv_rubrics.py`](./evaluate_openenv_rubrics.py)
- Training: [`train.py`](./train.py)
- OpenEnv manifest: [`openenv.yaml`](./openenv.yaml)

## How the Environment Works

At a high level, each episode follows this loop:
1. **Reset**: sample private utility vectors and initialize the shared resource pool.
2. **Negotiate**: the learner emits a text action (propose/counter/accept/reject/query/walk away).
3. **Opponent response**: each opponent evaluates the current proposal using its own private utility and either accepts or rejects.
4. **State update**: conversation history, active proposal, round count, and pool/accounting are updated.
5. **Termination check**:
   - success if all agents accept the same feasible allocation,
   - failure if max rounds are reached or learner walks away.
6. **Reward**: compute utility-driven reward with efficiency and completion bonuses (or no-deal penalty).

This gives a realistic multi-agent bargaining dynamic where the learner must infer preferences from dialogue and converge quickly.

## Results

Evidence is generated from real rollouts on the environment (`hard` difficulty, matched seeds across series).

Source: [`evidence_summary.md`](./evidence_summary.md)

| Series | Success Rate | Avg Reward | Avg Utility | Avg Rounds | Efficiency |
|---|---:|---:|---:|---:|---:|
| Rule baseline | 100.0% | 12.895 +- 0.000 | 2.082 +- 0.000 | 2.00 | 6.448 |
| Untrained model | 100.0% | 11.524 +- 2.910 | 2.579 +- 1.201 | 2.41 | 4.781 |
| Trained model | 100.0% | 13.298 +- 1.203 | 2.469 +- 1.442 | 1.75 | 7.599 |

What changed after training:
- Average reward: `+1.773` vs untrained
- Convergence speed: `0.66` fewer rounds
- Efficiency: `+2.818` reward-per-round

![CogniMarket reward progress and evaluation](./reward_plot.png)
*Caption: Top panel shows GRPO reward metrics over training steps; bottom panel compares baseline, pre-training, and post-training performance on the same seeds.*

![CogniMarket training loss](./reports/loss_plot.png)
*Caption: Training loss trend over optimization steps (lower is better; smooth decline indicates stable optimization).*

### Key Training Signals

- Reward progression: clear upward trend across training, showing learning over time. **(backbone metric)**
- Outcome reward mean: improving, indicating the agent is converging to better negotiated deals.
- Format reward mean: increasing, showing stronger action-structure compliance and output robustness.
- Loss curve: stabilizes after an early spike, consistent with stable post-warmup optimization.

### Detailed Training Curves (Reports)

![Outcome reward mean curve](./reports/rewards_outcome_reward_fn_mean_plot.png)
*Caption: Mean outcome reward increases over training, indicating improving deal quality.*

![Format reward mean curve](./reports/rewards_format_reward_fn_mean_plot.png)
*Caption: Mean format reward rises, showing stronger structured-action compliance.*

![Proposal quality reward mean curve](./reports/rewards_proposal_quality_reward_fn_mean_plot.png)
*Caption: Proposal-quality reward trend shows improving per-turn negotiation quality.*

![Overall reward plot from reports](./reports/reward_plot.png)
*Caption: Combined training reward signals from the report artifacts.*

### OpenEnv Rubric Check

Composable rubric evaluation is wired through [`openenv_rubrics.py`](./openenv_rubrics.py) and run via [`evaluate_openenv_rubrics.py`](./evaluate_openenv_rubrics.py).

Latest rubric report source: [`rubric_report.json`](./rubric_report.json)

- Rich informative signal: **false** (dense signal currently limited by many 1-step episodes)
- Captures hard-to-measure proxy: **true**
- Uses composable OpenEnv rubrics: **true**
- Hard to game: **true**

## Why It Matters

Who should care:
- **Infra/platform teams** building autonomous resource managers
- **Multi-agent researchers** studying negotiation and coordination
- **LLM+RL practitioners** who want measurable behavior change after training

Why:
- This environment turns negotiation quality into a trainable signal.
- It demonstrates that post-training can improve coordination behavior, not only style.

## Try It

### Hugging Face Space
- [AnshRaj112/cognimarket](https://huggingface.co/spaces/AnshRaj112/cognimarket)

### Colab training notebook
- [CogniMarket training notebook](https://colab.research.google.com/drive/16G-Juzt6g9FrDaB2WJRVElHeMheKWnlz?usp=sharing)

### Minimal local run
```bash
pip install -r requirements.txt
# Includes OpenEnv latest release (verified): openenv==0.1.13
python train.py
python run_training_evidence.py --checkpoint-dir ./checkpoints --difficulty hard --seed 42
python plot_training_rewards.py --checkpoint-dir ./checkpoints --smooth 10
python plot_training_loss.py --checkpoint-dir ./checkpoints --smooth 10
python evaluate_openenv_rubrics.py --episodes 20 --difficulty hard --policy strategic
```

## Submission Assets

- Training script (Unsloth + TRL): [`train.py`](./train.py)
- Evidence runner: [`run_training_evidence.py`](./run_training_evidence.py)
- Evidence summary: [`evidence_summary.md`](./evidence_summary.md)
- Reward/eval plot: [`reward_plot.png`](./reward_plot.png)
- Loss plot generator: [`plot_training_loss.py`](./plot_training_loss.py)

## Additional References

- Environment on Hugging Face Space: [AnshRaj112/cognimarket](https://huggingface.co/spaces/AnshRaj112/cognimarket)
- Colab training notebook: [CogniMarket training notebook](https://colab.research.google.com/drive/1qDy-I_JIA5cdNN3RbnBf0r-1wX1-l6Sf?usp=sharing)
- Training evidence summary: [`evidence_summary.md`](./evidence_summary.md)
- Reward and evaluation plot: [`reward_plot.png`](./reward_plot.png)
- Loss plot (generate and commit): [`reports/loss_plot.png`](./reports/loss_plot.png) via [`plot_training_loss.py`](./plot_training_loss.py)
- Training pipeline: [`train.py`](./train.py), [`run_training_evidence.py`](./run_training_evidence.py)
- OpenEnv rubric report: [`rubric_report.json`](./rubric_report.json)
- W&B run: not used in this run (`train.py` supports optional `--report-to wandb`).
- Additional media (video/blog/slides/presentation): not included in this submission.

