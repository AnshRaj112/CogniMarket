"""Test strategic baseline policy on hard mode."""
from evaluate import strategic_baseline_policy, run_episode
from compute_bazaar_env import ComputeBazaarEnv

env = ComputeBazaarEnv(max_rounds=12)
successes = 0
for i in range(10):
    metrics = run_episode(env, difficulty="hard", seed=42+i, policy=strategic_baseline_policy)
    s = "Y" if metrics["success"] else "N"
    print(f"  Ep {i+1}: reward={metrics['total_reward']:>8.2f}  util={metrics['utility_achieved']:>6.2f}  rds={metrics['rounds_used']}  success={s}")
    if metrics["success"]:
        successes += 1
print(f"\nStrategic baseline hard-mode success: {successes}/10 ({successes*10}%)")

# Also test the dense proposal reward
from reward import calculate_proposal_reward

# Should give positive reward for a generous, diverse proposal
r1 = calculate_proposal_reward(
    "PROPOSE: learner: gpu 25 cpu 20 memory 20; opponent_1: gpu 40 cpu 40 memory 40; opponent_2: gpu 35 cpu 40 memory 40",
    utilities={"learner": [0.5, 0.3, 0.2], "opponent_1": [0.3, 0.4, 0.3], "opponent_2": [0.2, 0.3, 0.5]},
    difficulty="hard",
)
print(f"\nDense reward (generous proposal): {r1:.2f}")

# Should penalize a repeated equal split
r2 = calculate_proposal_reward(
    "PROPOSE: learner: gpu 34 cpu 34 memory 34; opponent_1: gpu 33 cpu 33 memory 33; opponent_2: gpu 33 cpu 33 memory 33",
    utilities={"learner": [0.5, 0.3, 0.2], "opponent_1": [0.3, 0.4, 0.3], "opponent_2": [0.2, 0.3, 0.5]},
    history=[
        "learner: PROPOSE: learner: gpu 34 cpu 34 memory 34; opponent_1: gpu 33 cpu 33 memory 33; opponent_2: gpu 33 cpu 33 memory 33",
    ],
    difficulty="hard",
)
print(f"Dense reward (stagnant equal split): {r2:.2f}")
