## Training Evidence (Environment-Connected, End-to-End)

All evaluations use the same environment and same per-episode seeds.

### Setup
- Difficulty: `hard`
- Max rounds: `12`
- Episodes per series: `75`
- Base seed: `42` (episode seeds = `seed + ep`)

### Quantitative Comparison
| Series | Success Rate | Avg Reward | Avg Utility | Avg Rounds | Efficiency |
|---|---:|---:|---:|---:|---:|
| Rule baseline | 100.0% | 12.895 ± 0.000 | 2.082 ± 0.000 | 2.00 | 6.448 |
| Untrained model | 100.0% | 11.583 ± 2.504 | 2.440 ± 1.268 | 2.40 | 4.826 |
| Trained model | 100.0% | 13.268 ± 0.811 | 2.408 ± 1.374 | 1.88 | 7.057 |

### Improvement Summary (Trained vs Untrained)
- Reward improvement: `+1.684`
- Average-round reduction (faster convergence): `+0.52` rounds (positive means fewer rounds).
- Efficiency improvement (reward/round): `+2.231`
- Success-rate change: `+0.0 pp` (saturated at 100% across series).
- Utility trade-off: `-0.032` (small decrease can happen when deals close faster).

### Interpretation
Success rate is already saturated at 100%, so it is not a meaningful differentiator here.
The trained agent improves reward and efficiency by converging faster, trading a small amount
of utility for quicker deal closure rather than failing to reach agreements.

### Composite Metric
- `Efficiency Score = Avg Reward / Avg Rounds` (shown in the table).
- `Adjusted Score = Avg Reward - (0.5 x Avg Rounds)`
  - Untrained: `10.383`
  - Trained: `12.328`

### Statistical Confidence
- Episodes per series: `n=75`
- 95% CI for average reward:
  - Untrained: `11.583 ± 0.567`
  - Trained: `13.268 ± 0.184`
- Approximate effect size (Cohen's d, reward): `0.90` (large)

### Key Insight
Since success rate is saturated, improvements are reflected in reward and convergence speed
rather than acceptance probability.

### Curves
Generate training curves (and bar panel from `training_progress.json`) with:
```bash
python plot_training_rewards.py --checkpoint-dir ./checkpoints --smooth 10
```
