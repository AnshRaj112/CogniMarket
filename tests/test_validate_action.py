"""Unit tests for validate_and_fix_action (strict version)."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import validate_and_fix_action, FALLBACK_ACTION


def test_clean_accept_yes_with_deal():
    result = validate_and_fix_action("ACCEPT: YES", has_deal=True)
    assert result == "ACCEPT: YES", f"Got '{result}'"

def test_clean_accept_yes_without_deal():
    """ACCEPT: YES must be downgraded to NO when no deal exists."""
    result = validate_and_fix_action("ACCEPT: YES", has_deal=False)
    assert result == "ACCEPT: NO", f"Got '{result}'"

def test_clean_accept_no():
    result = validate_and_fix_action("ACCEPT: NO", has_deal=False)
    assert result == "ACCEPT: NO", f"Got '{result}'"

def test_clean_propose():
    action = "PROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 40 memory 30; opponent_2: gpu 30 cpu 30 memory 40"
    result = validate_and_fix_action(action)
    assert result.startswith("PROPOSE:"), f"Got '{result}'"
    assert "learner:" in result and "opponent_1:" in result and "opponent_2:" in result

def test_multi_line_takes_first_accept():
    action = "ACCEPT: YES\nPROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 40 memory 30; opponent_2: gpu 30 cpu 30 memory 40"
    result = validate_and_fix_action(action, has_deal=True)
    assert result == "ACCEPT: YES", f"Got '{result}'"

def test_multi_line_propose_first():
    action = "Some preamble\nPROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 40 memory 30; opponent_2: gpu 30 cpu 30 memory 40\nACCEPT: YES"
    result = validate_and_fix_action(action)
    assert result.startswith("PROPOSE:"), f"Got '{result}'"

def test_invalid_falls_back():
    result = validate_and_fix_action("I think we should negotiate more")
    assert result == FALLBACK_ACTION, f"Got '{result}'"

def test_empty_falls_back():
    result = validate_and_fix_action("")
    assert result == FALLBACK_ACTION, f"Got '{result}'"

def test_normalize_sums_exactly_100():
    """Proposal with sums > 100 must be normalized to EXACTLY 100."""
    action = "PROPOSE: learner: gpu 50 cpu 50 memory 50; opponent_1: gpu 50 cpu 50 memory 50; opponent_2: gpu 50 cpu 50 memory 50"
    result = validate_and_fix_action(action)
    assert result.startswith("PROPOSE:"), f"Got '{result}'"
    # Parse the result and verify sums
    import re
    pattern = re.compile(r"(learner|opponent_1|opponent_2): gpu (\d+) cpu (\d+) memory (\d+)")
    matches = pattern.findall(result)
    assert len(matches) == 3, f"Expected 3 agents, got {len(matches)} in '{result}'"
    gpu_sum = sum(int(m[1]) for m in matches)
    cpu_sum = sum(int(m[2]) for m in matches)
    mem_sum = sum(int(m[3]) for m in matches)
    assert gpu_sum == 100, f"GPU sum is {gpu_sum}, expected 100"
    assert cpu_sum == 100, f"CPU sum is {cpu_sum}, expected 100"
    assert mem_sum == 100, f"Memory sum is {mem_sum}, expected 100"

def test_normalize_under_exactly_100():
    """Proposal with sums < 100 must be normalized to EXACTLY 100."""
    action = "PROPOSE: learner: gpu 10 cpu 10 memory 10; opponent_1: gpu 10 cpu 10 memory 10; opponent_2: gpu 10 cpu 10 memory 10"
    result = validate_and_fix_action(action)
    import re
    pattern = re.compile(r"(learner|opponent_1|opponent_2): gpu (\d+) cpu (\d+) memory (\d+)")
    matches = pattern.findall(result)
    gpu_sum = sum(int(m[1]) for m in matches)
    cpu_sum = sum(int(m[2]) for m in matches)
    mem_sum = sum(int(m[3]) for m in matches)
    assert gpu_sum == 100, f"GPU sum is {gpu_sum}"
    assert cpu_sum == 100, f"CPU sum is {cpu_sum}"
    assert mem_sum == 100, f"Memory sum is {mem_sum}"

def test_already_100_stays_100():
    """Proposal that already sums to 100 must not be changed."""
    action = "PROPOSE: learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 40 memory 30; opponent_2: gpu 30 cpu 30 memory 40"
    result = validate_and_fix_action(action)
    import re
    pattern = re.compile(r"(learner|opponent_1|opponent_2): gpu (\d+) cpu (\d+) memory (\d+)")
    matches = pattern.findall(result)
    gpu_sum = sum(int(m[1]) for m in matches)
    cpu_sum = sum(int(m[2]) for m in matches)
    mem_sum = sum(int(m[3]) for m in matches)
    assert gpu_sum == 100, f"GPU sum is {gpu_sum}"
    assert cpu_sum == 100, f"CPU sum is {cpu_sum}"
    assert mem_sum == 100, f"Memory sum is {mem_sum}"

def test_rescue_unstructured():
    """Proposal without PROPOSE: prefix but with valid allocations."""
    action = "I think learner: gpu 40 cpu 30 memory 30; opponent_1: gpu 30 cpu 40 memory 30; opponent_2: gpu 30 cpu 30 memory 40"
    result = validate_and_fix_action(action)
    assert result.startswith("PROPOSE:"), f"Got '{result}'"

def test_case_insensitive():
    result = validate_and_fix_action("accept: yes", has_deal=True)
    assert result == "ACCEPT: YES", f"Got '{result}'"

def test_values_are_integers():
    """All resource values in output must be integers (no decimals)."""
    action = "PROPOSE: learner: gpu 33.3 cpu 33.3 memory 33.3; opponent_1: gpu 33.3 cpu 33.3 memory 33.3; opponent_2: gpu 33.3 cpu 33.3 memory 33.3"
    result = validate_and_fix_action(action)
    assert result.startswith("PROPOSE:"), f"Got '{result}'"
    # No decimal points should appear in output
    import re
    nums = re.findall(r"\d+\.\d+", result)
    assert len(nums) == 0, f"Found decimal values in output: {nums}"
    # Verify sums
    pattern = re.compile(r"(learner|opponent_1|opponent_2): gpu (\d+) cpu (\d+) memory (\d+)")
    matches = pattern.findall(result)
    for res_idx in [1, 2, 3]:
        total = sum(int(m[res_idx]) for m in matches)
        assert total == 100, f"Resource {res_idx} sum is {total}"

def test_accept_with_trailing_text():
    """ACCEPT: YES with trailing explanation must still be parsed."""
    result = validate_and_fix_action("ACCEPT: YES I agree with this deal", has_deal=True)
    assert result == "ACCEPT: YES", f"Got '{result}'"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"  PASS {test_fn.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL {test_fn.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test_fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed}/{passed+failed} tests passed")
    if failed:
        sys.exit(1)
