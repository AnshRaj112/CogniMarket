"""Compute Allocation Bazaar environment.

A lightweight, Colab-friendly, Gymnasium-style environment for multi-agent
negotiation with partially observable utilities.
"""

from __future__ import annotations

import random
import re
import os
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from groq import Groq
    _GROQ_AVAILABLE = True
except ImportError:
    _GROQ_AVAILABLE = False

try:
    from prompts import build_opponent_prompt, build_oversight_prompt
except ImportError:
    pass

try:  # Prefer OpenEnv if installed.
    from openenv import OpenEnvBase as _BaseEnv  # type: ignore

    _GYM_AVAILABLE = False
except ImportError:  # pragma: no cover - optional dependency
    try:
        import gymnasium as gym

        _BaseEnv = gym.Env
        _GYM_AVAILABLE = True
    except ImportError:  # pragma: no cover - optional dependency
        _BaseEnv = object  # fallback to keep module importable
        _GYM_AVAILABLE = False


RESOURCE_KEYS = ("gpu", "cpu", "memory")
AGENT_IDS = ("learner", "opponent_1", "opponent_2")
ROUND_PENALTY = -0.5
NO_DEAL_PENALTY = -8.0
DEAL_COMPLETION_BONUS = 3.0   # Guaranteed positive reward for closing ANY deal
FAST_DEAL_BONUS = 4.0
MID_DEAL_BONUS = 2.0
SPARSE_BONUS = 10.0
SUCCESS_UTILITY_THRESHOLD = 5.0  # Used only for sparse bonus tracking, NOT for success gate
MAX_UTILITY_SCALE = 15.0
SPARSE_BONUS_THRESHOLD = 0.85
EASY_ACCEPTANCE_THRESHOLD = 4.0
HARD_ACCEPTANCE_THRESHOLD = 5.0
MIN_UTILITY_VALUE = 0.001


def build_agent_ids(num_opponents: int) -> List[str]:
    """Return canonical agent IDs for learner + N opponents."""
    if num_opponents < 1:
        raise ValueError("num_opponents must be >= 1")
    return ["learner"] + [f"opponent_{i + 1}" for i in range(num_opponents)]

# ---------------------------------------------------------------------------
# Agent name aliases → canonical names (FIX #5)
# ---------------------------------------------------------------------------
_AGENT_NAME_ALIASES: Dict[str, str] = {
    "opp1": "opponent_1",
    "opp2": "opponent_2",
    "opp_1": "opponent_1",
    "opp_2": "opponent_2",
    "player1": "opponent_1",
    "player2": "opponent_2",
    "player_1": "opponent_1",
    "player_2": "opponent_2",
    "agent1": "opponent_1",
    "agent2": "opponent_2",
    "agent_1": "opponent_1",
    "agent_2": "opponent_2",
    "opponent1": "opponent_1",
    "opponent2": "opponent_2",
}


def normalize_agent_names(text: str) -> str:
    """Replace common agent name aliases with canonical names (FIX #5).

    Performs case-insensitive replacement of known aliases like opp1, player1,
    etc. to their canonical forms (opponent_1, opponent_2).
    """
    result = text
    # Sort by length descending to avoid partial matches (e.g. "opponent1" before "opp1")
    for alias in sorted(_AGENT_NAME_ALIASES, key=len, reverse=True):
        pattern = re.compile(re.escape(alias), re.IGNORECASE)
        result = pattern.sub(_AGENT_NAME_ALIASES[alias], result)
    return result


def clean_action(raw_output: str) -> str:
    """Extract the FIRST valid action line from raw model output (FIX #2, #3, #6).

    Scans line-by-line and returns the FIRST line starting with ACCEPT: or PROPOSE:.
    If no valid line is found, returns 'ACCEPT: NO' as safe fallback.

    This ensures:
    - Only ONE action per step (FIX #6)
    - No extra text / explanations (FIX #2)
    - Robust against messy model output (FIX #3)
    """
    # First normalize agent names throughout
    raw_output = normalize_agent_names(raw_output)

    lines = raw_output.strip().split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check for ACCEPT: prefix
        if re.match(r"^ACCEPT\s*:", line, re.IGNORECASE):
            # Normalize to canonical format
            val_part = re.sub(r"^ACCEPT\s*:\s*", "", line, flags=re.IGNORECASE).strip().upper()
            if "YES" in val_part:
                cleaned = "ACCEPT: YES"
            else:
                cleaned = "ACCEPT: NO"
            print(f"DEBUG [clean_action]: Found ACCEPT line -> {cleaned}")
            return cleaned

        # Check for PROPOSE: prefix
        if re.match(r"^PROPOSE\s*:", line, re.IGNORECASE):
            proposal_body = re.sub(r"^PROPOSE\s*:\s*", "", line, flags=re.IGNORECASE).strip()
            cleaned = f"PROPOSE: {proposal_body}"
            print(f"DEBUG [clean_action]: Found PROPOSE line -> {cleaned[:120]}")
            return cleaned

    # No valid line found — safe fallback
    print(f"DEBUG [clean_action]: No valid ACCEPT/PROPOSE line found in {len(lines)} lines -> fallback ACCEPT: NO")
    print(f"DEBUG [clean_action]: Raw output was: {repr(raw_output[:200])}")
    return "ACCEPT: NO"


def _parse_proposal_structured(
    proposal_text: str,
    agent_ids: Tuple[str, ...],
    resource_keys: Tuple[str, ...],
) -> Optional[Dict[str, Dict[str, float]]]:
    """Parse proposal text into structured allocation data without regex."""
    allocations: Dict[str, Dict[str, float]] = {}
    chunks = [chunk.strip() for chunk in proposal_text.split(";") if chunk.strip()]
    for chunk in chunks:
        if ":" not in chunk:
            continue
        agent_raw, body = chunk.split(":", 1)
        agent = agent_raw.strip().lower()
        if agent not in agent_ids:
            continue

        tokens = body.strip().split()
        if len(tokens) % 2 != 0:
            return None
        parsed: Dict[str, float] = {}
        for i in range(0, len(tokens), 2):
            key = tokens[i].strip().lower()
            value_text = tokens[i + 1].strip()
            if key not in resource_keys:
                continue
            try:
                value = float(value_text)
            except ValueError:
                return None
            parsed[key] = value
        if set(parsed.keys()) != set(resource_keys):
            return None
        allocations[agent] = parsed

    if set(allocations.keys()) != set(agent_ids):
        return None
    return allocations


def proposal_has_all_agents(
    action: str,
    agent_ids: Tuple[str, ...] = AGENT_IDS,
    resource_keys: Tuple[str, ...] = RESOURCE_KEYS,
) -> bool:
    """Return True when a PROPOSE action contains all required agents/resources."""
    stripped = action.strip()
    if not re.match(r"^PROPOSE\s*:", stripped, re.IGNORECASE):
        return False
    proposal_body = re.sub(r"^PROPOSE\s*:\s*", "", stripped, flags=re.IGNORECASE)
    parsed = _parse_proposal_structured(proposal_body, agent_ids, resource_keys)
    if parsed is None:
        return False
    return all(v >= 0.0 for a in parsed.values() for v in a.values())


def extract_agents_from_proposal_text(action: str) -> List[str]:
    """Extract mentioned agent ids from a PROPOSE action."""
    stripped = action.strip()
    if not re.match(r"^PROPOSE\s*:", stripped, re.IGNORECASE):
        return []
    proposal_body = re.sub(r"^PROPOSE\s*:\s*", "", stripped, flags=re.IGNORECASE)
    agents: List[str] = []
    for chunk in [c.strip() for c in proposal_body.split(";") if c.strip()]:
        if ":" not in chunk:
            continue
        aid = chunk.split(":", 1)[0].strip().lower()
        agents.append(aid)
    return agents


def validate_and_fix_proposal_with_meta(
    action: str,
    total_pool: int = 100,
    agent_ids: Tuple[str, ...] = AGENT_IDS,
    resource_keys: Tuple[str, ...] = RESOURCE_KEYS,
) -> Tuple[str, bool]:
    """Validate a PROPOSE action and fix resource totals if needed (FIX #4).

    If the action is ACCEPT, passes through unchanged.
    If the action is PROPOSE, validates:
    - All 3 agents present (learner, opponent_1, opponent_2)
    - All 3 resources present (gpu, cpu, memory)
    - Each resource sums to EXACTLY total_pool (auto-normalizes if not)

    If proposal is unparseable, returns a safe fallback equal-split proposal.

    Returns:
        A validated action string ready for env.step().
    """
    stripped = action.strip()

    # Pass through ACCEPT unchanged
    if re.match(r"^ACCEPT\s*:", stripped, re.IGNORECASE):
        return stripped, False

    # Must be PROPOSE
    if not re.match(r"^PROPOSE\s*:", stripped, re.IGNORECASE):
        print("DEBUG [validate_proposal]: Not ACCEPT or PROPOSE")
        return stripped, True

    proposal_body = re.sub(r"^PROPOSE\s*:\s*", "", stripped, flags=re.IGNORECASE)

    alloc = _parse_proposal_structured(proposal_body, agent_ids, resource_keys)
    if alloc is None:
        print("DEBUG [validate_proposal]: Structured parse failed")
        return stripped, True
    if any(value < 0.0 for a in alloc.values() for value in a.values()):
        print("DEBUG [validate_proposal]: Negative value detected")
        return stripped, True

    # Normalize each resource to sum to EXACTLY total_pool using largest-remainder
    agents = list(agent_ids)
    int_alloc: Dict[str, Dict[str, int]] = {a: {} for a in agents}

    for res in resource_keys:
        raw_vals = {a: alloc[a][res] for a in agents}
        raw_total = sum(raw_vals.values())

        if raw_total == 0:
            base = total_pool // len(agents)
            rem = total_pool - base * len(agents)
            for i, a in enumerate(agents):
                int_alloc[a][res] = base + (1 if i < rem else 0)
        else:
            scale = total_pool / raw_total
            scaled = {a: raw_vals[a] * scale for a in agents}
            floored = {a: int(scaled[a]) for a in agents}
            remainder = total_pool - sum(floored.values())

            fractionals = sorted(agents, key=lambda a: scaled[a] - floored[a], reverse=True)
            for a in agents:
                int_alloc[a][res] = floored[a]
            for i in range(remainder):
                int_alloc[fractionals[i]][res] += 1

            if abs(raw_total - total_pool) > 0.5:
                print(f"DEBUG [validate_proposal]: Normalized {res}: {raw_total:.1f} -> {total_pool}")

    # Verify sums
    for res in resource_keys:
        res_sum = sum(int_alloc[a][res] for a in agents)
        assert res_sum == total_pool, f"BUG: {res} sum is {res_sum}, expected {total_pool}"

    # Build clean output
    parts = []
    for agent in agent_ids:
        a = int_alloc[agent]
        segments = [f"{rk} {a[rk]}" for rk in resource_keys]
        parts.append(f"{agent}: " + " ".join(segments))

    result = "PROPOSE: " + "; ".join(parts)
    print(
        f"DEBUG [validate_proposal]: Valid proposal | "
        f"agents={agent_ids} "
        f"resources={resource_keys}"
    )
    return result, False


def validate_and_fix_proposal(
    action: str,
    total_pool: int = 100,
    agent_ids: Tuple[str, ...] = AGENT_IDS,
    resource_keys: Tuple[str, ...] = RESOURCE_KEYS,
) -> str:
    """Backward-compatible wrapper that returns only the normalized action."""
    fixed, _ = validate_and_fix_proposal_with_meta(
        action=action,
        total_pool=total_pool,
        agent_ids=agent_ids,
        resource_keys=resource_keys,
    )
    return fixed


def _safe_fallback_proposal(
    total_pool: int = 100,
    agent_ids: Tuple[str, ...] = AGENT_IDS,
    resource_keys: Tuple[str, ...] = RESOURCE_KEYS,
) -> str:
    """Return a safe equal-split proposal as fallback (FIX #4)."""
    base = total_pool // len(agent_ids)
    rem = total_pool - base * len(agent_ids)
    shares = [base + (1 if i < rem else 0) for i in range(len(agent_ids))]
    parts = []
    for i, agent in enumerate(agent_ids):
        parts.append(
            f"{agent}: " + " ".join(f"{rk} {shares[i]}" for rk in resource_keys)
        )
    result = "PROPOSE: " + "; ".join(parts)
    print(f"DEBUG [validate_proposal]: Using safe fallback proposal: {result}")
    return result


@dataclass
class DealState:
    """Container for active negotiation proposal data."""

    proposal: Dict[str, Dict[str, float]]
    accepted_by: Set[str]
    proposer: str


class ComputeBazaarEnv(_BaseEnv):
    """Turn-based negotiation environment for compute allocation bargaining."""

    def __init__(
        self,
        max_rounds: int = 12,
        seed: Optional[int] = None,
        agent_ids: Optional[List[str]] = None,
        resource_keys: Optional[List[str]] = None,
        total_pool: float = 100.0,
    ):
        """Initialize environment configuration.

        Args:
            max_rounds: Maximum negotiation rounds before truncation.
            seed: Optional RNG seed used for utility sampling and stochastic behavior.
        """
        self.max_rounds = max_rounds
        self.total_pool = float(total_pool)
        self.agent_ids: Tuple[str, ...] = tuple(agent_ids or AGENT_IDS)
        self.resource_keys: Tuple[str, ...] = tuple(resource_keys or RESOURCE_KEYS)
        if "learner" not in self.agent_ids:
            raise ValueError("agent_ids must include 'learner'")
        self.opponent_ids: Tuple[str, ...] = tuple(a for a in self.agent_ids if a != "learner")
        self.rng = random.Random(seed)
        self.utilities: Dict[str, List[float]] = {}
        self.history: List[str] = []
        self.rounds_used = 0
        self._terminated = False
        self._truncated = False
        self._difficulty = "hard"
        self._deal: Optional[DealState] = None
        self._oversight_queries = 0
        self._last_learner_proposal: Optional[Dict[str, Dict[str, float]]] = None
        self._last_opponent_response = "none"
        self._last_opponent_utility = 0.0

        # Gymnasium-compatible spaces (only defined when gymnasium is available).
        if _GYM_AVAILABLE:
            import gymnasium as gym  # noqa: PLC0415

            # Actions are free-form text strings.
            self.action_space = gym.spaces.Text(min_length=0, max_length=512)
            # Observations are a dict of conversation history and numeric vectors.
            self.observation_space = gym.spaces.Dict(
                {
                    "conversation_history": gym.spaces.Sequence(
                        gym.spaces.Text(min_length=0, max_length=512)
                    ),
                    "private_utility": gym.spaces.Box(
                        low=0.0, high=1.0, shape=(len(self.resource_keys),), dtype=float
                    ),
                    "remaining_compute_pool": gym.spaces.Dict(
                        {k: gym.spaces.Box(low=0.0, high=self.total_pool, shape=(), dtype=float) for k in self.resource_keys}
                    ),
                    "last_proposal": gym.spaces.Text(min_length=0, max_length=512),
                    "last_opponent_response": gym.spaces.Text(min_length=0, max_length=32),
                    "last_opponent_utility": gym.spaces.Box(
                        low=0.0, high=MAX_UTILITY_SCALE, shape=(), dtype=float
                    ),
                }
            )

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Reset environment state for a new episode.

        options:
            difficulty: "easy" or "hard"
        """
        if seed is not None:
            self.rng.seed(seed)
        self._difficulty = (options or {}).get("difficulty", "hard")
        self.utilities = self._sample_utilities(self._difficulty)
        self.history = []
        self.rounds_used = 0
        self._terminated = False
        self._truncated = False
        self._deal = None
        self._oversight_queries = 0
        self._last_learner_proposal = None
        self._last_opponent_response = "none"
        self._last_opponent_utility = 0.0
        return self._build_obs(), {"difficulty": self._difficulty}

    def step(self, action: str) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        """Execute one learner turn with automatic opponent reactions.

        FIX #1: accepted_by is ONLY reset when a NEW proposal is made.
        Previous accepted_by state is preserved across steps when the
        learner sends ACCEPT (not a new proposal).
        """
        if self._terminated or self._truncated:
            raise RuntimeError("Episode has ended. Call reset() before step().")

        self.rounds_used += 1
        round_penalty = ROUND_PENALTY
        reward = round_penalty
        fallback_used = False
        invalid_proposal = False
        rejected_last_round = self._last_opponent_response in {"rejected", "partial_accept", "repeated_proposal"}
        accept_blocked = False
        accept_block_reason = "none"

        # --- FIX #2, #3, #5: Clean, normalize, and extract single action ---
        action = clean_action(action)

        # Block invalid ACCEPT behavior and force proposal to avoid idle loops.
        if action.strip().upper().startswith("ACCEPT: YES"):
            deal_complete_now = self._deal is not None and self._deal.accepted_by >= set(self.agent_ids)
            if not deal_complete_now:
                action = self._force_new_proposal()
                accept_blocked = True
                accept_block_reason = "accept_without_complete_deal"
                print(f"DEBUG [env.step R{self.rounds_used}]: Blocked invalid ACCEPT -> forced PROPOSE")

        # If prior round did not converge, require a new proposal this round.
        if rejected_last_round and not action.lower().startswith("propose:"):
            action = self._force_new_proposal()
            if not accept_blocked:
                accept_blocked = True
                accept_block_reason = "must_repropose_after_rejection"
            print(f"DEBUG [env.step R{self.rounds_used}]: Forced PROPOSE after rejection loop")

        if action.lower().startswith("propose:"):
            expected_agents = list(self.agent_ids)
            generated_agents = extract_agents_from_proposal_text(action)
            print(f"DEBUG [env.step R{self.rounds_used}]: EXPECTED AGENTS: {expected_agents}")
            print(f"DEBUG [env.step R{self.rounds_used}]: GENERATED AGENTS: {generated_agents}")
            action, invalid_proposal = validate_and_fix_proposal_with_meta(
                action,
                int(self.total_pool),
                self.agent_ids,
                self.resource_keys,
            )
            if invalid_proposal:
                reward -= 10.0
                forced = self._force_new_proposal()
                print(f"DEBUG [env.step R{self.rounds_used}]: Invalid proposal -> forced regenerate proposal")
                action, _ = validate_and_fix_proposal_with_meta(
                    forced,
                    int(self.total_pool),
                    self.agent_ids,
                    self.resource_keys,
                )
                fallback_used = True
        if accept_blocked:
            # Penalize useless ACCEPT attempts that cannot close a deal.
            reward -= 3.0

        action_type, parsed_proposal = self._parse_action(action)
        prev_opp_utility = float(self._last_opponent_utility)

        print(f"DEBUG [env.step R{self.rounds_used}]: action_type={action_type}, "
              f"has_proposal={parsed_proposal is not None}, "
              f"deal_exists={self._deal is not None}, "
              f"accepted_by={self._deal.accepted_by if self._deal else 'N/A'}")

        self.history.append(f"learner: {action}")

        # --- FIX #1: ONLY reset accepted_by on NEW proposals ---
        if action_type in {"propose", "counter_offer"} and parsed_proposal:
            if self._last_learner_proposal is not None and self._proposals_equal(
                parsed_proposal, self._last_learner_proposal
            ):
                reward -= 3.0
                self._last_opponent_response = "repeated_proposal"
            # New proposal from learner: reset accepted_by to just the proposer
            self._deal = DealState(proposal=parsed_proposal, accepted_by={"learner"}, proposer="learner")
            self._last_learner_proposal = parsed_proposal
            print(f"DEBUG [env.step R{self.rounds_used}]: NEW proposal from learner, "
                  f"accepted_by reset to {{'learner'}}")
        elif action_type == "accept" and self._deal is not None:
            # Learner accepts existing deal — add to accepted_by, DO NOT reset
            self._deal.accepted_by.add("learner")
            print(f"DEBUG [env.step R{self.rounds_used}]: Learner ACCEPT, "
                  f"accepted_by now: {self._deal.accepted_by}")
        elif action_type == "reject" and self._deal is not None:
            self._deal.accepted_by.discard("learner")
            print(f"DEBUG [env.step R{self.rounds_used}]: Learner REJECT, "
                  f"accepted_by now: {self._deal.accepted_by}")
        # NOTE: If action_type is "message" or "accept" with no deal, accepted_by is UNCHANGED

        if action_type == "walk_away":
            self._terminated = True
            reward += NO_DEAL_PENALTY

        if action_type == "query_oversight":
            self._oversight_queries += 1
            self.history.append(f"oversight: {self._oversight_explanation()}")

        if not self._terminated:
            self._run_opponents()
            if self._deal is not None:
                print(f"DEBUG [env.step R{self.rounds_used}]: After opponents: "
                      f"deal.accepted_by={self._deal.accepted_by}, need={set(self.agent_ids)}")

        deal_closed = self._deal is not None and self._deal.accepted_by >= set(self.agent_ids)
        utility = 0.0
        efficiency_bonus = 0.0
        sparse_bonus = 0.0
        deal_completion_bonus = 0.0
        success = False

        if deal_closed:
            # ----- DEAL CLOSED: success = True ALWAYS -----
            # The old code gated success on utility >= 5.0, which meant a deal
            # where the learner compromised (utility=3.0) was PENALIZED as a
            # failure. This killed the RL learning signal — the agent was
            # punished for doing exactly what it should (closing deals).
            success = True
            self._terminated = True

            print(f"DEBUG [env.step R{self.rounds_used}]: *** DEAL CLOSED! ***")
            print(f"DEBUG [env.step R{self.rounds_used}]: accepted_by={self._deal.accepted_by}")
            print(f"DEBUG [env.step R{self.rounds_used}]: SUCCESS={success}")

            utility = self._calculate_utility(
                allocation=self._deal.proposal.get("learner", {}),
                utility_vector=self.utilities["learner"],
            )
        opponent_utility = self._avg_opponent_utility(self._deal.proposal if self._deal else None)
        delta_opponent_utility = opponent_utility - prev_opp_utility
        threshold = EASY_ACCEPTANCE_THRESHOLD if self._difficulty == "easy" else HARD_ACCEPTANCE_THRESHOLD
        threshold = self._effective_acceptance_threshold(threshold)
        reward += 0.5 * (opponent_utility / max(threshold, 1e-6))
        if opponent_utility < threshold:
            # Multi-agent difficulty: soften hard threshold into decayed penalty.
            deficit = max(0.0, threshold - opponent_utility)
            decay = 1.0 / max(1.0, math.sqrt(len(self.opponent_ids)))
            reward -= 1.5 * deficit / max(threshold, 1e-6) * decay
        if rejected_last_round:
            # Encourage adaptation only when previous outcome needed improvement.
            reward += delta_opponent_utility
        if self._deal is not None:
            accepted_count = len(self._deal.accepted_by)
            coalition_ratio = accepted_count / max(1, len(self.agent_ids))
            # Partial-agreement shaping: reward convergence even before full closure.
            reward += 2.0 * coalition_ratio
            if 1 < accepted_count < len(self.agent_ids):
                # Explicit coalition incentive for >=2 aligned agents.
                reward += 1.0
        if success:
            reward += 10.0
        else:
            # With many agents, strict failure penalty is too harsh; decay it.
            reward -= 2.0 / max(1.0, math.sqrt(len(self.agent_ids)))
            # Guardrail: rejected / non-terminal states should not end up net-positive.
            reward = min(reward, -0.1)
        if action_type == "accept" and not deal_closed:
            reward -= 3.0

        if self.rounds_used >= self.max_rounds and not self._terminated:
            self._truncated = True
            reward += NO_DEAL_PENALTY
        if self._deal is not None:
            if self._deal.accepted_by >= set(self.agent_ids):
                self._last_opponent_response = "accepted"
            elif any(opp in self._deal.accepted_by for opp in self.opponent_ids):
                self._last_opponent_response = "partial_accept"
            else:
                self._last_opponent_response = "rejected"
        self._last_opponent_utility = opponent_utility

        info = {
            "utility_achieved": round(utility, 4),
            "rounds_used": self.rounds_used,
            "success": success,
            "deal_closed": deal_closed,
            "efficiency_bonus": efficiency_bonus,
            "sparse_bonus": sparse_bonus,
            "deal_completion_bonus": deal_completion_bonus,
            "oversight_queries": self._oversight_queries,
            "opponent_utility": round(opponent_utility, 4),
            "delta_opponent_utility": round(delta_opponent_utility, 4),
            "fallback_used": fallback_used,
            "invalid_proposal": invalid_proposal,
            "accept_blocked": accept_blocked,
            "accept_block_reason": accept_block_reason,
            "rejected_last_round": rejected_last_round,
        }
        print(f"DEBUG [env.step R{self.rounds_used}]: reward={reward:.2f}, "
              f"terminated={self._terminated}, truncated={self._truncated}, "
              f"success={success}, utility={utility:.2f}")
        return self._build_obs(), float(reward), self._terminated, self._truncated, info

    def _force_new_proposal(self) -> str:
        """Return a deterministic proposal when ACCEPT must be blocked.

        Prefers adapting the current deal by shifting a small amount from learner
        to opponents; falls back to equal split if no deal exists yet.
        """
        if self._deal is None:
            return _safe_fallback_proposal(
                total_pool=int(self.total_pool),
                agent_ids=self.agent_ids,
                resource_keys=self.resource_keys,
            )

        shifted: Dict[str, Dict[str, float]] = {
            a: {rk: float(self._deal.proposal.get(a, {}).get(rk, 0.0)) for rk in self.resource_keys}
            for a in self.agent_ids
        }
        learner = "learner"
        if learner not in shifted:
            return _safe_fallback_proposal(
                total_pool=int(self.total_pool),
                agent_ids=self.agent_ids,
                resource_keys=self.resource_keys,
            )

        opponents = [a for a in self.agent_ids if a != learner]
        if not opponents:
            return _safe_fallback_proposal(
                total_pool=int(self.total_pool),
                agent_ids=self.agent_ids,
                resource_keys=self.resource_keys,
            )

        for rk in self.resource_keys:
            transfer = min(3.0, shifted[learner][rk])
            if transfer <= 0:
                continue
            shifted[learner][rk] -= transfer
            per_opp = transfer / len(opponents)
            for opp in opponents:
                shifted[opp][rk] += per_opp

        parts = []
        for agent in self.agent_ids:
            segments = [f"{rk} {shifted[agent][rk]:.3f}" for rk in self.resource_keys]
            parts.append(f"{agent}: " + " ".join(segments))
        forced = "PROPOSE: " + "; ".join(parts)
        normalized, _ = validate_and_fix_proposal_with_meta(
            forced,
            total_pool=int(self.total_pool),
            agent_ids=self.agent_ids,
            resource_keys=self.resource_keys,
        )
        return normalized

    def _build_obs(self) -> Dict[str, Any]:
        return {
            "conversation_history": self.history[-8:],
            "private_utility": self.utilities.get("learner", [0.0, 0.0, 0.0]),
            "total_pool": float(self.total_pool),
            "remaining_compute_pool": self._remaining_pool(),
            "agent_ids": list(self.agent_ids),
            "resource_keys": list(self.resource_keys),
            "last_proposal": self._format_proposal(self._last_learner_proposal),
            "last_opponent_response": self._last_opponent_response,
            "last_opponent_utility": float(self._last_opponent_utility),
        }

    def _proposals_equal(
        self,
        p1: Dict[str, Dict[str, float]],
        p2: Dict[str, Dict[str, float]],
        tol: float = 1e-6,
    ) -> bool:
        for agent in self.agent_ids:
            a1 = p1.get(agent, {})
            a2 = p2.get(agent, {})
            for res in self.resource_keys:
                if abs(float(a1.get(res, 0.0)) - float(a2.get(res, 0.0))) > tol:
                    return False
        return True

    def _avg_opponent_utility(self, proposal: Optional[Dict[str, Dict[str, float]]]) -> float:
        if not proposal:
            return 0.0
        utils = []
        for opp in self.opponent_ids:
            utils.append(
                self._calculate_utility(
                    allocation=proposal.get(opp, {}),
                    utility_vector=self.utilities.get(opp, [0.0, 0.0, 0.0]),
                )
            )
        # Use bottleneck utility (minimum opponent utility) so reward aligns with
        # the true acceptance cliff: both opponents must be above threshold.
        return float(min(utils) if utils else 0.0)

    def _format_proposal(self, proposal: Optional[Dict[str, Dict[str, float]]]) -> str:
        if not proposal:
            return ""
        parts = []
        for agent in self.agent_ids:
            a = proposal.get(agent, {})
            parts.append(
                f"{agent}: " + " ".join(f"{rk} {a.get(rk, 0)}" for rk in self.resource_keys)
            )
        return "; ".join(parts)

    def _parse_action(self, action_text: str) -> Tuple[str, Optional[Dict[str, Dict[str, float]]]]:
        """Parse a CLEANED action into an action type and optional allocation proposal.

        At this point the action has already been through clean_action() and
        validate_and_fix_proposal(), so it should always be either:
            ACCEPT: YES / ACCEPT: NO
            PROPOSE: learner: gpu X cpu Y memory Z; opponent_1: ...; opponent_2: ...

        FIX #1: Accept/reject no longer modify accepted_by here — that is done
        in step() to keep the logic in one place.
        """
        text = action_text.strip().lower()

        # 1. Structured "ACCEPT: YES/NO" (primary path)
        if text.startswith("accept:"):
            val = text.replace("accept:", "").strip()
            if "yes" in val:
                print(f"DEBUG [_parse_action]: ACCEPT YES detected")
                return "accept", None
            else:
                print(f"DEBUG [_parse_action]: ACCEPT NO / reject detected")
                return "reject", None

        # 2. Structured "PROPOSE: ..." (primary path)
        if text.startswith("propose:"):
            proposal_str = text.replace("propose:", "").strip()
            proposal = self._extract_allocation(proposal_str)
            if proposal:
                return "propose", proposal
            else:
                print(f"DEBUG [_parse_action]: PROPOSE prefix found but allocation extraction failed")
                return "message", None

        # 3. Walk away (rarely used but valid)
        if re.search(r"\bwalk\s+away\b", text):
            return "walk_away", None

        # 4. Oversight query
        if re.search(r"\bquery_oversight\b", text) or re.search(r"\boversight\b", text):
            return "query_oversight", None

        # 5. Last resort: try to extract allocation from raw text
        proposal = self._extract_allocation(text)
        if proposal:
            return "propose", proposal

        print(f"DEBUG [_parse_action]: Unrecognized action, treating as message: {text[:80]}")
        return "message", None

    def _extract_allocation(self, text: str) -> Optional[Dict[str, Dict[str, float]]]:
        proposal: Dict[str, Dict[str, float]] = {}
        chunks = [c.strip() for c in text.split(";") if c.strip()]
        for chunk in chunks:
            if ":" not in chunk:
                continue
            agent, body = chunk.split(":", 1)
            agent = agent.strip().lower()
            if agent not in self.agent_ids:
                continue
            alloc: Dict[str, float] = {}
            for rk in self.resource_keys:
                match = re.search(rf"\b{re.escape(rk)}\s*(-?\d+(?:\.\d+)?)", body, re.IGNORECASE)
                if not match:
                    alloc = {}
                    break
                alloc[rk] = max(0.0, float(match.group(1)))
            if alloc:
                proposal[agent] = alloc

        if set(proposal.keys()) != set(self.agent_ids):
            print(
                f"DEBUG [_extract_allocation]: Agent mismatch: got {set(proposal.keys())}, "
                f"need {set(self.agent_ids)}"
            )
            return None

        if not self._proposal_is_feasible(proposal):
            return None
        return proposal

    def _proposal_is_feasible(self, proposal: Dict[str, Dict[str, float]]) -> bool:
        if not isinstance(proposal, dict):
            print("DEBUG [_proposal_is_feasible]: proposal is not a dict")
            return False
        for resource in self.resource_keys:
            total = 0.0
            for agent in self.agent_ids:
                allocation = proposal.get(agent)
                if not isinstance(allocation, dict):
                    print(f"DEBUG [_proposal_is_feasible]: {agent} allocation is not a dict")
                    return False
                value = allocation.get(resource)
                if value is None:
                    print(f"DEBUG [_proposal_is_feasible]: {agent} missing {resource}")
                    return False
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    print(f"DEBUG [_proposal_is_feasible]: {agent}.{resource}={value!r} not numeric")
                    return False
                if value < 0.0:
                    print(f"DEBUG [_proposal_is_feasible]: {agent}.{resource}={value} is negative")
                    return False
                total += value
            if total > self.total_pool + 0.5:  # Allow small float tolerance
                print(f"DEBUG [_proposal_is_feasible]: {resource} total={total} exceeds pool={self.total_pool}")
                return False
        return True

    def _remaining_pool(self) -> Dict[str, float]:
        if not self._deal:
            return {k: self.total_pool for k in self.resource_keys}
        return {
            k: max(0.0, self.total_pool - sum(self._deal.proposal[a][k] for a in self.agent_ids))
            for k in self.resource_keys
        }

    def _run_opponents(self) -> None:
        """Run opponent reactions to the current deal.

        FIX #1: Opponents add/remove themselves from accepted_by on the
        EXISTING deal. A new DealState is only created when an opponent
        makes a counter-proposal (which correctly resets accepted_by).
        """
        if self._deal is None:
            for opponent in self.opponent_ids:
                self.history.append(
                    f"{opponent}: Can you propose a full allocation covering all agents/resources?"
                )
            return

        api_key = os.environ.get("GROQ_API_KEY")
        use_llm = _GROQ_AVAILABLE and api_key
        opponent_utilities_log: Dict[str, float] = {}

        for opponent in self.opponent_ids:
            utility = self._calculate_utility(
                allocation=self._deal.proposal.get(opponent, {}),
                utility_vector=self.utilities[opponent],
            )
            opponent_utilities_log[opponent] = round(float(utility), 4)
            
            # fallback/rule-based threshold evaluation
            threshold = (
                EASY_ACCEPTANCE_THRESHOLD
                if self._difficulty == "easy"
                else HARD_ACCEPTANCE_THRESHOLD
            )
            threshold = self._effective_acceptance_threshold(threshold)
            rule_based_accept = utility >= threshold
            
            if use_llm:
                try:
                    client = Groq(api_key=api_key)
                    sys_prompt = build_opponent_prompt(
                        agent_id=opponent,
                        utility_vector=self.utilities[opponent],
                        conversation_history=self.history,
                        remaining_pool=self._remaining_pool(),
                        agent_ids=self.agent_ids,
                    )
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": sys_prompt}],
                        temperature=0.7,
                        max_tokens=512,
                    )
                    response_text = completion.choices[0].message.content.strip()
                    
                    # FIX #1: Modify accepted_by on EXISTING deal, don't reset
                    if "accept" in response_text.lower() and "reject" not in response_text.lower():
                        self._deal.accepted_by.add(opponent)
                        print(f"DEBUG [_run_opponents]: {opponent} ACCEPTS, accepted_by={self._deal.accepted_by}")
                    elif "reject" in response_text.lower():
                        self._deal.accepted_by.discard(opponent)
                        print(f"DEBUG [_run_opponents]: {opponent} REJECTS, accepted_by={self._deal.accepted_by}")
                    elif "propose" in response_text.lower() or "counter" in response_text.lower():
                        # Opponent counter-proposal — this SHOULD reset accepted_by
                        # because it's a genuinely new proposal
                        counter_text = normalize_agent_names(response_text)
                        proposal = self._extract_allocation(counter_text.lower())
                        if proposal:
                            self._deal = DealState(proposal=proposal, accepted_by={opponent}, proposer=opponent)
                            print(f"DEBUG [_run_opponents]: {opponent} COUNTER-PROPOSES, "
                                  f"accepted_by reset to {{{opponent}}}")
                        else:
                            # Counter-proposal failed to parse — treat as reject
                            self._deal.accepted_by.discard(opponent)
                            print(f"DEBUG [_run_opponents]: {opponent} counter-proposal "
                                  f"failed to parse, treating as reject")
                    
                    self.history.append(f"{opponent}: {response_text}")
                    continue
                except Exception:
                    # Fallback to rule-based on any error
                    pass

            # Fallback (rule-based) — FIX #1: properly accumulate accepted_by
            if rule_based_accept:
                self._deal.accepted_by.add(opponent)
                self.history.append(f"{opponent}: accept")
                print(f"DEBUG [_run_opponents]: {opponent} rule-accept (utility={utility:.2f} >= {threshold}), "
                      f"accepted_by={self._deal.accepted_by}")
            else:
                self._deal.accepted_by.discard(opponent)
                self.history.append(f"{opponent}: reject, please improve my share.")
                print(f"DEBUG [_run_opponents]: {opponent} rule-reject (utility={utility:.2f} < {threshold}), "
                      f"accepted_by={self._deal.accepted_by}")
        print(f"DEBUG [_run_opponents]: Opponent utilities: {opponent_utilities_log}")

    def get_opponent_personalities(self) -> Dict[str, str]:
        """Infer opponent personalities from their utility weights."""
        personalities = {}
        for opp in self.opponent_ids:
            weights = self.utilities.get(opp, [0.0, 0.0, 0.0])
            max_idx = weights.index(max(weights))
            resource = self.resource_keys[max_idx].upper()
            trait = "Balanced"
            if weights[max_idx] > 0.6:
                trait = f"Obsessed with {resource}"
            elif weights[max_idx] > 0.45:
                trait = f"Prioritizes {resource}"
            personalities[opp] = f"Needs: {resource} ({trait})"
        return personalities

    def get_utility_summary(self) -> Dict[str, float]:
        """Return current utility scores for all agents."""
        summary = {}
        # Calculate learner utility relative to current deal or last learner proposal
        prop = self._deal.proposal if self._deal else self._last_learner_proposal
        for agent in self.agent_ids:
            summary[agent] = self._calculate_utility(
                allocation=prop.get(agent, {}) if prop else {},
                utility_vector=self.utilities.get(agent, [0.0, 0.0, 0.0])
            )
        return summary

    def _calculate_utility(self, allocation: Dict[str, float], utility_vector: List[float]) -> float:

        """Compute scaled utility in [0, MAX_UTILITY_SCALE] using weighted resource value."""
        if not allocation:
            return 0.0
        vec = [min(max(float(allocation.get(k, 0.0)), 0.0), self.total_pool) for k in self.resource_keys]
        scaled = [v / self.total_pool for v in vec]
        return float(sum(v * w for v, w in zip(scaled, utility_vector)) * MAX_UTILITY_SCALE)

    def _effective_acceptance_threshold(self, base_threshold: float) -> float:
        """Return a feasible acceptance threshold for current agent count.

        For many-opponent settings with integer allocations, a strict fixed threshold
        can become practically unreachable for all opponents simultaneously.
        """
        if len(self.opponent_ids) <= 2:
            return base_threshold
        # Best uniform per-opponent integer share when learner is minimized.
        max_uniform_share = self.total_pool // max(len(self.opponent_ids), 1)
        feasible_uniform_utility = (max_uniform_share / max(self.total_pool, 1e-6)) * MAX_UTILITY_SCALE
        # Keep a learner reserve so deals do not collapse learner utility.
        learner_reserve = 10.0
        per_opp_with_reserve = max(0.0, (self.total_pool - learner_reserve) / max(len(self.opponent_ids), 1))
        feasible_with_reserve = (per_opp_with_reserve / max(self.total_pool, 1e-6)) * MAX_UTILITY_SCALE
        # Use a small stability margin to avoid deadlocks around integer/tie edges.
        stability_margin = 0.05
        return min(
            base_threshold,
            max(0.0, float(feasible_uniform_utility) - stability_margin),
            max(0.0, float(feasible_with_reserve) - stability_margin),
        )

    def _sample_utilities(self, difficulty: str = "hard") -> Dict[str, List[float]]:
        """Sample private utility vectors for learner and two opponents."""
        learner = self._normalize([self.rng.random() for _ in self.resource_keys])
        sampled = {"learner": learner}
        for opp in self.opponent_ids:
            if difficulty == "easy":
                sampled[opp] = self._normalize(
                    [weight + self.rng.uniform(-0.05, 0.05) for weight in learner]
                )
            else:
                sampled[opp] = self._normalize([self.rng.random() for _ in self.resource_keys])
        return sampled

    @staticmethod
    def _normalize(vec: List[float]) -> List[float]:
        """Normalize vector to sum to 1 while applying a minimum floor to each value."""
        clipped = [max(float(v), MIN_UTILITY_VALUE) for v in vec]
        total = sum(clipped)
        return [v / total for v in clipped]

    def _oversight_explanation(self) -> str:
        if self._deal is None:
            return "No concrete allocation yet. Fairness and efficiency cannot be assessed."
            
        api_key = os.environ.get("GROQ_API_KEY")
        if _GROQ_AVAILABLE and api_key:
            try:
                client = Groq(api_key=api_key)
                sys_prompt = build_oversight_prompt(
                    conversation_history=self.history,
                    current_proposal=self._deal.proposal,
                    utilities=self.utilities,
                )
                completion = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[{"role": "system", "content": sys_prompt}],
                    temperature=0.3,
                    max_tokens=512,
                )
                return completion.choices[0].message.content.strip()
            except Exception:
                pass

        # Fallback
        totals = {k: sum(self._deal.proposal[a][k] for a in self.agent_ids) for k in self.resource_keys}
        feasible = all(v <= self.total_pool for v in totals.values())
        return (
            "Current proposal is "
            + ("feasible" if feasible else "infeasible")
            + f"; resource usage is {totals}."
        )
