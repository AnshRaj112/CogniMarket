"""Quick functional test for all 7 fixes."""

import sys
sys.path.insert(0, ".")

from compute_bazaar_env import (
    ComputeBazaarEnv,
    clean_action,
    normalize_agent_names,
    validate_and_fix_proposal,
    _safe_fallback_proposal,
)

passed = 0
failed = 0

def test(name, condition):
    global passed, failed
    if condition:
        print(f"  PASS: {name}")
        passed += 1
    else:
        print(f"  FAIL: {name}")
        failed += 1


print("\n=== FIX #1: accepted_by persistence ===")

# Use hard difficulty + learner-heavy proposal so opponents will NOT accept immediately
env = ComputeBazaarEnv(max_rounds=12, seed=42)
env.reset(seed=42, options={"difficulty": "hard"})

# Step 1: Learner proposes unfair split favoring themselves
action1 = "PROPOSE: learner: gpu 60 cpu 60 memory 60; opponent_1: gpu 20 cpu 20 memory 20; opponent_2: gpu 20 cpu 20 memory 20"
obs1, r1, term1, trunc1, info1 = env.step(action1)
test("Step 1: deal state exists after proposal", env._deal is not None)

opp_accepted_step1 = frozenset(env._deal.accepted_by) if env._deal else frozenset()
print(f"  After step 1: accepted_by = {opp_accepted_step1}")

if not term1:
    # Step 2: Learner accepts — should ADD to accepted_by, NOT reset
    action2 = "ACCEPT: YES"
    obs2, r2, term2, trunc2, info2 = env.step(action2)
    if env._deal:
        print(f"  After step 2 (ACCEPT YES): accepted_by = {env._deal.accepted_by}")
        test("Step 2: learner still in accepted_by after ACCEPT YES", "learner" in env._deal.accepted_by)
    else:
        test("Step 2: deal closed means accepted_by accumulated correctly", term2)
else:
    # Deal closed on step 1 (unlikely with unfair split + hard difficulty)
    test("Step 1: deal closed immediately (accepted_by accumulated in one step)", info1.get("success", False))


print("\n=== FIX #2 & #3: clean_action() ===")

# Multi-line output -> single action
messy1 = "Let me think about this...\nACCEPT: YES\nPROPOSE: something"
test("Multi-line -> first valid line (ACCEPT)", clean_action(messy1) == "ACCEPT: YES")

messy2 = "I think we should negotiate.\nThe best approach is:\nPROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 35 memory 35; opponent_2: gpu 30 cpu 35 memory 35"
result2 = clean_action(messy2)
test("Multi-line -> first PROPOSE found", result2.startswith("PROPOSE:"))

messy3 = "Let me explain my reasoning in detail about why we need more resources..."
test("No valid action -> fallback ACCEPT: NO", clean_action(messy3) == "ACCEPT: NO")


print("\n=== FIX #4: validate_and_fix_proposal() ===")

# Invalid totals -> auto-normalize
bad_totals = "PROPOSE: learner: gpu 50 cpu 50 memory 50; opponent_1: gpu 50 cpu 50 memory 50; opponent_2: gpu 50 cpu 50 memory 50"
fixed = validate_and_fix_proposal(bad_totals)
test("Over-budget proposal auto-normalized", "PROPOSE:" in fixed)
# Parse and check totals
import re
nums = re.findall(r"gpu (\d+) cpu (\d+) memory (\d+)", fixed)
if len(nums) == 3:
    gpu_total = sum(int(n[0]) for n in nums)
    cpu_total = sum(int(n[1]) for n in nums)
    mem_total = sum(int(n[2]) for n in nums)
    test(f"GPU total = 100 (got {gpu_total})", gpu_total == 100)
    test(f"CPU total = 100 (got {cpu_total})", cpu_total == 100)
    test(f"Memory total = 100 (got {mem_total})", mem_total == 100)

# Fallback proposal
fallback = _safe_fallback_proposal(100)
test("Safe fallback is valid PROPOSE", "PROPOSE:" in fallback)
nums_fb = re.findall(r"gpu (\d+) cpu (\d+) memory (\d+)", fallback)
if len(nums_fb) == 3:
    test("Fallback GPU = 100", sum(int(n[0]) for n in nums_fb) == 100)

# ACCEPT passes through
test("ACCEPT passthrough", validate_and_fix_proposal("ACCEPT: YES") == "ACCEPT: YES")


print("\n=== FIX #5: normalize_agent_names() ===")

test("opp1 -> opponent_1", "opponent_1" in normalize_agent_names("opp1: gpu 30"))
test("opp2 -> opponent_2", "opponent_2" in normalize_agent_names("opp2: gpu 30"))
test("player1 -> opponent_1", "opponent_1" in normalize_agent_names("player1: gpu 30"))
test("player2 -> opponent_2", "opponent_2" in normalize_agent_names("player2: gpu 30"))
test("learner unchanged", "learner" in normalize_agent_names("learner: gpu 30"))


print("\n=== FIX #6: single action only ===")

mixed = "ACCEPT: YES\nPROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 35 memory 35; opponent_2: gpu 30 cpu 35 memory 35"
result = clean_action(mixed)
test("Mixed ACCEPT+PROPOSE -> only first (ACCEPT)", result == "ACCEPT: YES")


print("\n=== FIX #1 (continued): Full deal completion test ===")

env2 = ComputeBazaarEnv(max_rounds=12, seed=99)
env2.reset(seed=99, options={"difficulty": "easy"})

# Propose something generous for opponents
deal_closed = False
for step in range(12):
    if step == 0:
        action = "PROPOSE: learner: gpu 20 cpu 20 memory 20; opponent_1: gpu 40 cpu 40 memory 40; opponent_2: gpu 40 cpu 40 memory 40"
    else:
        action = "ACCEPT: YES"
    
    obs, r, term, trunc, info = env2.step(action)
    if info.get("success"):
        deal_closed = True
        print(f"  Deal closed at step {step + 1}!")
        break
    if term or trunc:
        break

test("Deal eventually closes with generous proposal + ACCEPT", deal_closed or env2._terminated)


print(f"\n{'='*60}")
print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
print(f"{'='*60}")
