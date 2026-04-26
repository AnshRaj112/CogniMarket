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
- regret-aware terminal penalty when the final deal is dominated by an earlier learner proposal,
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
| Untrained model | 100.0% | 11.583 +- 2.504 | 2.440 +- 1.268 | 2.40 | 4.826 |
| Trained model | 100.0% | 13.268 +- 0.811 | 2.408 +- 1.374 | 1.88 | 7.057 |

What changed after training:
- Average reward: `+1.684` vs untrained
- Convergence speed: `0.52` fewer rounds
- Efficiency: `+2.231` reward-per-round

### Statistical Confidence (Judge-Facing)

Using `n=75` matched-seed episodes per series (`hard` difficulty):

- Reward gain (trained vs untrained): `+1.684`
- 95% CI (reward mean):
  - Untrained: `11.583 +/- 0.567`
  - Trained: `13.268 +/- 0.184`
- Approximate effect size (Cohen's d, reward): `0.90` (large)

Interpretation:
- Success rate is saturated at `100%`, so reward and efficiency are the meaningful indicators.
- The trained policy improves reward with substantially lower variance and faster convergence.

![CogniMarket reward progress and evaluation](./reward_plot.png)
*Caption: Top: Training dynamics under GRPO show consistent improvement in total reward and outcome quality, alongside stable formatting behavior.
Bottom: On identical evaluation seeds, the trained agent achieves higher reward and faster convergence than both the baseline and untrained model, while maintaining near-perfect success rates—demonstrating learned negotiation efficiency. Success rate is saturated; differences are reflected in reward and efficiency.*

![CogniMarket training loss](./reports/loss_plot.png)
*Caption: Training loss exhibits an initial spike as the policy explores new strategies, followed by a steady decline—indicating stabilization and convergence of the optimization process.*

### Key Training Signals

- Reward progression: clear upward trend across training, showing learning over time. **(backbone metric)**
- Outcome reward mean: improving, indicating the agent is converging to better negotiated deals.
- Format reward mean: increasing, showing stronger action-structure compliance and output robustness.
- Regret-aware terminal shaping: discourages accepting strategically inferior final deals when a better Pareto-feasible learner proposal already existed in the same episode.
- Loss curve: stabilizes after an early spike, consistent with stable post-warmup optimization.

### Detailed Training Curves (Reports)

![Outcome reward mean curve](./reports/rewards_outcome_reward_fn_mean_plot.png)
*Caption: Outcome reward improves steadily over training, indicating that the agent learns to propose increasingly acceptable and mutually beneficial allocations.*

![Format reward mean curve](./reports/rewards_format_reward_fn_mean_plot.png)
*Caption: Format reward improves steadily, indicating that the agent learns to produce increasingly well-structured and valid action outputs.*

![Proposal quality reward mean curve](./reports/rewards_proposal_quality_reward_fn_mean_plot.png)
*Caption: Proposal-quality reward improves over time (becomes less negative), indicating that the agent learns to generate increasingly effective and acceptable proposals at each negotiation step.*

![Overall reward plot from reports](./reports/reward_plot.png)
*Caption: Total reward improves steadily over training (becoming less negative), indicating that the agent learns to make progressively better negotiation decisions and achieve more favorable outcomes.*

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

- Project blog: [`Blog.md`](./Blog.md)
- Environment on Hugging Face Space: [AnshRaj112/cognimarket](https://huggingface.co/spaces/AnshRaj112/cognimarket)
- Colab training notebook: [CogniMarket training notebook](https://colab.research.google.com/drive/1qDy-I_JIA5cdNN3RbnBf0r-1wX1-l6Sf?usp=sharing)
- Training evidence summary: [`evidence_summary.md`](./evidence_summary.md)
- Reward and evaluation plot: [`reward_plot.png`](./reward_plot.png)
- Loss plot (generate and commit): [`reports/loss_plot.png`](./reports/loss_plot.png) via [`plot_training_loss.py`](./plot_training_loss.py)
- Training pipeline: [`train.py`](./train.py), [`run_training_evidence.py`](./run_training_evidence.py)
- OpenEnv rubric report: [`rubric_report.json`](./rubric_report.json)
- W&B run: not used in this run (`train.py` supports optional `--report-to wandb`).

## Current Limitations and Improvements

Current limitations:
- Dense informative signal remains weaker than desired in short trajectories
  (`dense_signal: 0.2`, `rich_informative_signal: false`).
- Success rate is saturated at `100%`, so it is less useful as a differentiating metric.

Planned improvements:
- Increase long-horizon negotiation frequency (reduce 1-step saturation).
- Add harder opponent profiles and dynamic preference shifts.
- Report statistical significance tests alongside confidence intervals.
- Expand ablations (remove one reward head at a time) to show causal contribution.
- Add dedicated regret diagnostics (rate of dominated-final-deal episodes) to quantify strategic consistency gains.
