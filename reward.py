"""Standalone reward calculation for Compute Allocation Bazaar.

This module exposes ``calculate_reward`` as a pure function so it can be
imported independently of the environment (e.g., in training loss callbacks).
Shared numeric constants are imported directly from ``compute_bazaar_env.py``
to avoid drift. The only constant defined here is ``OVERSIGHT_ACCURACY_BONUS``,
which is specific to the reward module and not used by the environment.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from compute_bazaar_env import (  # noqa: F401 – re-exported for callers
    DEAL_COMPLETION_BONUS,
    FAST_DEAL_BONUS,
    MAX_UTILITY_SCALE,
    MID_DEAL_BONUS,
    NO_DEAL_PENALTY,
    RESOURCE_KEYS,
    ROUND_PENALTY,
    SPARSE_BONUS,
    SPARSE_BONUS_THRESHOLD,
    SUCCESS_UTILITY_THRESHOLD,
    EASY_ACCEPTANCE_THRESHOLD,
    HARD_ACCEPTANCE_THRESHOLD,
    AGENT_IDS,
)


# ---------------------------------------------------------------------------
# reward.py-only constants
# ---------------------------------------------------------------------------

OVERSIGHT_ACCURACY_BONUS: float = 2.5
"""Optional bonus when the oversight agent provides an accurate explanation.

This constant is intentionally absent from ``compute_bazaar_env.py`` because
the environment itself does not implement oversight scoring; it is applied
only by external reward-shaping code (e.g. training callbacks).
"""

# --- Dense proposal-level reward constants ---
PROXIMITY_SCALE: float = 0.5
"""Scale factor for opponent utility proximity reward."""

IMPROVEMENT_BONUS: float = 1.5
"""Bonus when a proposal improves opponent utility vs the previous proposal."""

STAGNATION_PENALTY: float = -3.0
"""Penalty when a proposal is identical to the previous one."""

SELF_SACRIFICE_PENALTY: float = -1.0
"""Penalty when learner drops own utility below a floor unnecessarily."""

DIVERSITY_BONUS: float = 0.5
"""Bonus for proposals that differ from the naive equal split."""

LEARNER_UTILITY_FLOOR: float = 3.0
"""Minimum learner utility before self-sacrifice penalty kicks in."""

DIVERSITY_MIN_DIFF: float = 10.0
"""Minimum total-unit difference from recent proposals to avoid penalty."""

DIVERSITY_HISTORY_DEPTH: int = 3
"""Number of recent proposals to check for diversity."""


def calculate_utility(
    allocation: Dict[str, float],
    utility_vector: List[float],
    total_pool: float = 100.0,
) -> float:
    """Compute the learner's utility for a given resource allocation.

    Utility is the dot product of the fractional allocation and the private
    utility weight vector, scaled to [0, MAX_UTILITY_SCALE].

    Args:
        allocation: Resource amounts received, e.g. {"gpu": 40, "cpu": 30, "memory": 30}.
        utility_vector: Private weights [gpu_w, cpu_w, memory_w] summing to 1.
        total_pool: Total available units per resource (default 100).

    Returns:
        Scaled utility in [0, MAX_UTILITY_SCALE].
    """
    if not allocation:
        return 0.0
    # Clamp values to valid range then scale each resource to [0, 1].
    scaled = [
        min(max(float(allocation.get(k, 0.0)), 0.0), total_pool) / total_pool
        for k in RESOURCE_KEYS
    ]
    return float(sum(s * w for s, w in zip(scaled, utility_vector)) * MAX_UTILITY_SCALE)


def calculate_reward(
    allocation: Optional[Dict[str, float]],
    utility_vector: List[float],
    rounds_used: int,
    deal_closed: bool,
    oversight_accurate: bool = False,
    total_pool: float = 100.0,
) -> tuple[float, Dict[str, float]]:
    """Compute the full episode reward for the learner agent.

    This standalone helper centralizes the reward calculation used for
    offline evaluation, reward shaping, and custom callbacks.

    Reward components
    -----------------
    1. ``round_penalty``:        ROUND_PENALTY x rounds_used (dense cost per turn).
    2. ``deal_completion_bonus``: +3 guaranteed bonus for closing ANY deal.
    3. ``utility_reward``:       Utility gained from allocation (if deal closed).
    4. ``efficiency_bonus``:     +4 if deal < 6 rounds, +2 if < 10 rounds.
    5. ``sparse_bonus``:         +10 if utility >= 85 % of MAX_UTILITY_SCALE.
    6. ``no_deal_penalty``:      -5 if no deal at all.
    7. ``oversight_bonus``:      +2.5 if oversight gave an accurate explanation (optional).

    Args:
        allocation: Learner's final allocation dict, or None if no deal.
        utility_vector: Learner's private utility weights.
        rounds_used: Number of rounds elapsed this episode.
        deal_closed: Whether all three agents accepted the same proposal.
        oversight_accurate: Whether the oversight explanation was accurate.
        total_pool: Total available resource units (default 100).

    Returns:
        A tuple of (total_reward: float, breakdown: dict) where breakdown
        maps component names to their float values for logging/analysis.
    """
    breakdown: Dict[str, float] = {
        "round_penalty": 0.0,
        "utility_reward": 0.0,
        "deal_completion_bonus": 0.0,
        "efficiency_bonus": 0.0,
        "sparse_bonus": 0.0,
        "no_deal_penalty": 0.0,
        "oversight_bonus": 0.0,
    }

    # 1. Dense round cost -- charged regardless of outcome.
    breakdown["round_penalty"] = ROUND_PENALTY * rounds_used

    if deal_closed and allocation:
        # 2. Deal completion bonus -- guaranteed positive for closing ANY deal.
        breakdown["deal_completion_bonus"] = DEAL_COMPLETION_BONUS

        # 3. Utility reward: how well does the allocation match private preferences?
        utility = calculate_utility(allocation, utility_vector, total_pool)
        breakdown["utility_reward"] = utility

        # 4. Efficiency bonus: reward for closing deals quickly.
        if rounds_used < 6:
            breakdown["efficiency_bonus"] = FAST_DEAL_BONUS
        elif rounds_used < 10:
            breakdown["efficiency_bonus"] = MID_DEAL_BONUS

        # 5. Sparse bonus: reward for achieving a high-utility deal.
        if utility / MAX_UTILITY_SCALE >= SPARSE_BONUS_THRESHOLD:
            breakdown["sparse_bonus"] = SPARSE_BONUS

        # NOTE: NO penalty applied for closed deals. NO_DEAL_PENALTY is ONLY
        # for episodes that end without any agreement.
    else:
        # No deal: walked away or timed out.
        breakdown["no_deal_penalty"] = NO_DEAL_PENALTY

    # 6. Optional oversight accuracy bonus.
    if oversight_accurate:
        breakdown["oversight_bonus"] = OVERSIGHT_ACCURACY_BONUS

    total = sum(breakdown.values())
    return total, breakdown


def calculate_format_reward(completion: str) -> float:
    """Compute a dense reward for following the structured output format.

    Rewards:
    - +1.0 for using a valid prefix (PROPOSE: or ACCEPT:).
    - +1.0 for containing a parsable allocation if proposing.
    - +0.5 for mentioning all three resource types (gpu, cpu, memory).
    - Penalty (-2.0) for extremely long or repetitive strings.

    Args:
        completion: The model's generated text.

    Returns:
        Scalar reward in [-2.0, 3.0].
    """
    reward = 0.0
    text = completion.strip().lower()

    # 1. Prefix reward
    if text.startswith("propose:") or text.startswith("accept:"):
        reward += 1.0

    # 2. Structure/Content reward (Proposals)
    if "propose:" in text:
        # Check for agent mentions
        if "learner:" in text and "opponent_1:" in text and "opponent_2:" in text:
            reward += 1.0
        # Check for resource mentions
        if all(k in text for k in RESOURCE_KEYS):
            reward += 1.0
    
    # 3. Structure/Content reward (Acceptance)
    elif "accept:" in text:
        if "yes" in text or "no" in text:
            # Simple binary choice is good
            reward += 2.0

    # 4. Length/Sanity penalty
    if len(completion) > 100 or completion.count("\n") > 5:
        reward -= 1.0
    
    # Very harsh penalty for repetition (common failure mode)
    words = text.split()
    if len(words) > 10 and len(set(words)) / len(words) < 0.3:
        reward -= 2.0

    return float(reward)


# ---------------------------------------------------------------------------
# Dense proposal-quality reward (Component 1: Learning Signal Fix)
# ---------------------------------------------------------------------------

def _parse_proposal_allocations(text: str) -> Optional[Dict[str, Dict[str, float]]]:
    """Extract per-agent allocations from a completion string.

    Returns a dict like {"learner": {"gpu": 40, ...}, ...} or None.
    """
    pattern = re.compile(
        r"(learner|opponent_1|opponent_2)\s*:\s*gpu\s*(\d+(?:\.\d+)?)\s*cpu\s*(\d+(?:\.\d+)?)\s*memory\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    matches = pattern.findall(text.lower())
    if len(matches) < 3:
        return None

    matched_agents = {m[0] for m in matches}
    if matched_agents != set(AGENT_IDS):
        return None

    result: Dict[str, Dict[str, float]] = {}
    for agent, gpu, cpu, mem in matches[:3]:
        result[agent] = {"gpu": float(gpu), "cpu": float(cpu), "memory": float(mem)}
    return result


def _extract_previous_proposal(history: List[str]) -> Optional[Dict[str, Dict[str, float]]]:
    """Find the most recent proposal from conversation history."""
    for turn in reversed(history):
        if "gpu" in turn.lower() and "cpu" in turn.lower():
            alloc = _parse_proposal_allocations(turn)
            if alloc:
                return alloc
    return None


def _proposals_identical(p1: Dict[str, Dict[str, float]], p2: Dict[str, Dict[str, float]]) -> bool:
    """Check if two proposals have identical allocations (within tolerance)."""
    for agent in AGENT_IDS:
        if agent not in p1 or agent not in p2:
            return False
        for res in RESOURCE_KEYS:
            if abs(p1[agent].get(res, 0) - p2[agent].get(res, 0)) > 0.5:
                return False
    return True


def _is_equal_split(proposal: Dict[str, Dict[str, float]], total_pool: float = 100.0) -> bool:
    """Check if a proposal is essentially an equal split."""
    per_agent = total_pool / 3.0
    for agent in AGENT_IDS:
        if agent not in proposal:
            return False
        for res in RESOURCE_KEYS:
            if abs(proposal[agent].get(res, 0) - per_agent) > 2.0:
                return False
    return True


def _total_unit_difference(p1: Dict[str, Dict[str, float]], p2: Dict[str, Dict[str, float]]) -> float:
    """Sum of absolute differences across all agents and resources."""
    total = 0.0
    for agent in AGENT_IDS:
        a1 = p1.get(agent, {})
        a2 = p2.get(agent, {})
        for res in RESOURCE_KEYS:
            total += abs(a1.get(res, 0) - a2.get(res, 0))
    return total


def calculate_proposal_reward(
    completion: str,
    utilities: Optional[Dict[str, List[float]]] = None,
    history: Optional[List[str]] = None,
    difficulty: str = "hard",
    total_pool: float = 100.0,
    deal_closed: bool = False,
) -> float:
    """Compute a dense per-proposal reward for learning signal (Component 1).

    This function provides directional feedback on EVERY proposal, not just
    on terminal deal outcomes. It tells the agent:
    - How close this proposal is to being accepted by opponents
    - Whether it improved vs the previous proposal
    - Whether it is just repeating the same proposal (stagnation)
    - Whether the learner is unnecessarily sacrificing utility

    Args:
        completion: The model's generated text (a single action).
        utilities: Dict of utility vectors {"learner": [...], "opponent_1": [...], "opponent_2": [...]}.
        history: Conversation history list from the environment.
        difficulty: "easy" or "hard".
        total_pool: Resource pool size.

    Returns:
        Dense reward scalar (typically in [-3.0, +5.0]).
    """
    if deal_closed:
        return 10.0

    reward = 0.0
    text = completion.strip().lower()

    # ACCEPT penalty: if the model says ACCEPT but the deal hasn't closed,
    # it's wasting a turn. Heavy penalty to break ACCEPT-spam loops.
    if not text.startswith("propose:"):
        if ("accept" in text) and not deal_closed:
            return -5.0  # ACCEPT spam penalty
        return 0.0

    # Parse the proposal
    proposal = _parse_proposal_allocations(text)
    if proposal is None:
        return -1.0  # Unparseable proposal

    if utilities is None:
        return 0.0  # Can't evaluate without utility vectors

    threshold = EASY_ACCEPTANCE_THRESHOLD if difficulty == "easy" else HARD_ACCEPTANCE_THRESHOLD

    # --- Signal 1: Opponent proximity to acceptance threshold ---
    # Reward proportional to how close each opponent is to accepting
    for opp in ("opponent_1", "opponent_2"):
        opp_alloc = proposal.get(opp, {})
        opp_vec = utilities.get(opp)
        if opp_vec and opp_alloc:
            opp_utility = calculate_utility(opp_alloc, opp_vec, total_pool)
            proximity = min(opp_utility / threshold, 1.2)  # Cap at 1.2x threshold
            reward += PROXIMITY_SCALE * proximity * 0.5  # Per opponent, so * 0.5
            if opp_utility < threshold:
                reward -= 1.5

    # --- Signal 2: Improvement over previous proposal ---
    if history:
        prev_proposal = _extract_previous_proposal(history)
        if prev_proposal is not None:
            # Check if this proposal improves opponent utility
            improved = False
            for opp in ("opponent_1", "opponent_2"):
                opp_vec = utilities.get(opp)
                if opp_vec:
                    prev_util = calculate_utility(prev_proposal.get(opp, {}), opp_vec, total_pool)
                    curr_util = calculate_utility(proposal.get(opp, {}), opp_vec, total_pool)
                    if curr_util > prev_util + 0.3:
                        improved = True
                        break

            if improved:
                reward += IMPROVEMENT_BONUS

            # --- Signal 3: Stagnation penalty ---
            if _proposals_identical(proposal, prev_proposal):
                reward += STAGNATION_PENALTY

    # --- Signal 4: Self-sacrifice penalty ---
    learner_alloc = proposal.get("learner", {})
    learner_vec = utilities.get("learner")
    if learner_vec and learner_alloc:
        learner_utility = calculate_utility(learner_alloc, learner_vec, total_pool)
        if learner_utility < LEARNER_UTILITY_FLOOR:
            reward += SELF_SACRIFICE_PENALTY

    # --- Signal 5: Diversity bonus (not equal split) ---
    if not _is_equal_split(proposal, total_pool):
        reward += DIVERSITY_BONUS

    # --- Signal 6: History diversity (last N proposals must differ by >= 10 units) ---
    if history:
        recent_proposals = []
        for turn in reversed(history):
            if turn.startswith("learner:") and "gpu" in turn.lower():
                p = _parse_proposal_allocations(turn)
                if p:
                    recent_proposals.append(p)
                if len(recent_proposals) >= DIVERSITY_HISTORY_DEPTH:
                    break

        for prev in recent_proposals:
            diff = _total_unit_difference(proposal, prev)
            if diff < DIVERSITY_MIN_DIFF and not _proposals_identical(proposal, prev):
                # Too similar but not identical (identical already penalized above)
                reward -= 1.5
                break  # Only penalize once

    # Global failure pressure + safety clamp against accidental bonus stacking.
    reward -= 2.0
    reward = max(reward, -3.0)
    reward = min(reward, 3.0)
    return float(reward)
