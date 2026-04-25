"""Compute Allocation Bazaar environment.

A lightweight, Colab-friendly, Gymnasium-style environment for multi-agent
negotiation with partially observable utilities.
"""

from __future__ import annotations

import random
import re
import os
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


def validate_and_fix_proposal(action: str, total_pool: int = 100) -> str:
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
        return stripped

    # Must be PROPOSE
    if not re.match(r"^PROPOSE\s*:", stripped, re.IGNORECASE):
        print(f"DEBUG [validate_proposal]: Not ACCEPT or PROPOSE, returning fallback")
        return _safe_fallback_proposal(total_pool)

    proposal_body = re.sub(r"^PROPOSE\s*:\s*", "", stripped, flags=re.IGNORECASE)

    # Parse agent allocations
    agent_pattern = re.compile(
        r"(learner|opponent_1|opponent_2)\s*:\s*gpu\s*(\d+(?:\.\d+)?)\s*cpu\s*(\d+(?:\.\d+)?)\s*memory\s*(\d+(?:\.\d+)?)",
        re.IGNORECASE,
    )
    matches = agent_pattern.findall(proposal_body)

    if len(matches) < 3:
        print(f"DEBUG [validate_proposal]: Only {len(matches)} agents parsed (need 3), using fallback")
        print(f"DEBUG [validate_proposal]: Proposal body was: {proposal_body[:120]}")
        return _safe_fallback_proposal(total_pool)

    matched_agents = {m[0].lower() for m in matches}
    required = {"learner", "opponent_1", "opponent_2"}
    if matched_agents != required:
        print(f"DEBUG [validate_proposal]: Agent mismatch: got {matched_agents}, need {required}, using fallback")
        return _safe_fallback_proposal(total_pool)

    # Build raw allocation dict
    alloc: Dict[str, Dict[str, float]] = {}
    for agent, gpu, cpu, memory in matches[:3]:
        agent = agent.lower()
        try:
            alloc[agent] = {"gpu": float(gpu), "cpu": float(cpu), "memory": float(memory)}
        except ValueError:
            print(f"DEBUG [validate_proposal]: Non-numeric value for {agent}, using fallback")
            return _safe_fallback_proposal(total_pool)

        if any(v < 0 for v in alloc[agent].values()):
            print(f"DEBUG [validate_proposal]: Negative value for {agent}, using fallback")
            return _safe_fallback_proposal(total_pool)

    # Normalize each resource to sum to EXACTLY total_pool using largest-remainder
    agents = list(required)  # stable order
    agents.sort()  # deterministic: learner, opponent_1, opponent_2
    int_alloc: Dict[str, Dict[str, int]] = {a: {} for a in agents}

    for res in RESOURCE_KEYS:
        raw_vals = {a: alloc[a][res] for a in agents}
        raw_total = sum(raw_vals.values())

        if raw_total == 0:
            base = total_pool // 3
            rem = total_pool - base * 3
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
    for res in RESOURCE_KEYS:
        res_sum = sum(int_alloc[a][res] for a in agents)
        assert res_sum == total_pool, f"BUG: {res} sum is {res_sum}, expected {total_pool}"

    # Build clean output
    parts = []
    for agent in ("learner", "opponent_1", "opponent_2"):
        a = int_alloc[agent]
        parts.append(f"{agent}: gpu {a['gpu']} cpu {a['cpu']} memory {a['memory']}")

    result = "PROPOSE: " + "; ".join(parts)
    print(
        f"DEBUG [validate_proposal]: Valid proposal | "
        f"gpu={[int_alloc[a]['gpu'] for a in ('learner', 'opponent_1', 'opponent_2')]} "
        f"cpu={[int_alloc[a]['cpu'] for a in ('learner', 'opponent_1', 'opponent_2')]} "
        f"mem={[int_alloc[a]['memory'] for a in ('learner', 'opponent_1', 'opponent_2')]}"
    )
    return result


def _safe_fallback_proposal(total_pool: int = 100) -> str:
    """Return a safe equal-split proposal as fallback (FIX #4)."""
    base = total_pool // 3
    rem = total_pool - base * 3
    shares = [base + (1 if i < rem else 0) for i in range(3)]
    result = (
        f"PROPOSE: learner: gpu {shares[0]} cpu {shares[0]} memory {shares[0]}; "
        f"opponent_1: gpu {shares[1]} cpu {shares[1]} memory {shares[1]}; "
        f"opponent_2: gpu {shares[2]} cpu {shares[2]} memory {shares[2]}"
    )
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

        # --- FIX #2, #3, #5: Clean, normalize, and extract single action ---
        action = clean_action(action)
        if self.agent_ids == AGENT_IDS and self.resource_keys == RESOURCE_KEYS:
            action = validate_and_fix_proposal(action, int(self.total_pool))

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
        reward += 0.5 * (opponent_utility / max(threshold, 1e-6))
        if opponent_utility < threshold:
            reward -= 1.5
        reward += delta_opponent_utility
        if success:
            reward += 10.0
        else:
            reward -= 2.0
            # Guardrail: rejected / non-terminal states should not end up net-positive.
            reward = min(reward, -0.1)
        if action_type == "accept" and not deal_closed:
            reward -= 5.0  # Heavy penalty for ACCEPT spam when deal isn't closed

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
        }
        print(f"DEBUG [env.step R{self.rounds_used}]: reward={reward:.2f}, "
              f"terminated={self._terminated}, truncated={self._truncated}, "
              f"success={success}, utility={utility:.2f}")
        return self._build_obs(), float(reward), self._terminated, self._truncated, info

    def _build_obs(self) -> Dict[str, Any]:
        return {
            "conversation_history": self.history[-8:],
            "private_utility": self.utilities.get("learner", [0.0, 0.0, 0.0]),
            "remaining_compute_pool": self._remaining_pool(),
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

        for opponent in self.opponent_ids:
            utility = self._calculate_utility(
                allocation=self._deal.proposal.get(opponent, {}),
                utility_vector=self.utilities[opponent],
            )
            
            # fallback/rule-based threshold evaluation
            threshold = (
                EASY_ACCEPTANCE_THRESHOLD
                if self._difficulty == "easy"
                else HARD_ACCEPTANCE_THRESHOLD
            )
            rule_based_accept = utility >= threshold
            
            if use_llm:
                try:
                    client = Groq(api_key=api_key)
                    sys_prompt = build_opponent_prompt(
                        agent_id=opponent,
                        utility_vector=self.utilities[opponent],
                        conversation_history=self.history,
                        remaining_pool=self._remaining_pool(),
                    )
                    completion = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[{"role": "system", "content": sys_prompt}],
                        temperature=0.7,
                        max_tokens=64,
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

    def _calculate_utility(self, allocation: Dict[str, float], utility_vector: List[float]) -> float:
        """Compute scaled utility in [0, MAX_UTILITY_SCALE] using weighted resource value."""
        if not allocation:
            return 0.0
        vec = [min(max(float(allocation.get(k, 0.0)), 0.0), self.total_pool) for k in self.resource_keys]
        scaled = [v / self.total_pool for v in vec]
        return float(sum(v * w for v, w in zip(scaled, utility_vector)) * MAX_UTILITY_SCALE)

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
                    max_tokens=64,
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
