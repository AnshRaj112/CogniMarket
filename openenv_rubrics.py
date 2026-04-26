"""Composable rubric scoring for Compute Allocation Bazaar.

This module implements a small OpenEnv-style rubric system in code:
- each rubric returns a scalar in [0, 1]
- rubrics are composable via weighted sums
- final reports include per-rubric breakdowns
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Sequence, Tuple


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


@dataclass
class EpisodeTrace:
    rewards: List[float]
    infos: List[Dict[str, object]]
    actions: List[str]

    @property
    def total_steps(self) -> int:
        return len(self.rewards)

    @property
    def non_terminal_steps(self) -> int:
        if not self.rewards:
            return 0
        return max(0, len(self.rewards) - 1)


class Rubric:
    """Base rubric with OpenEnv-like forward contract."""

    name: str = "rubric"

    def score(self, trace: EpisodeTrace) -> float:
        raise NotImplementedError


class DenseSignalRubric(Rubric):
    """Rewards environments that provide informative per-step feedback."""

    name = "dense_signal"

    def score(self, trace: EpisodeTrace) -> float:
        if trace.total_steps == 0:
            return 0.0
        if trace.non_terminal_steps == 0:
            return 0.2

        non_terminal_rewards = trace.rewards[:-1]
        shaped_steps = sum(1 for r in non_terminal_rewards if abs(float(r)) > 1e-6)
        shaped_ratio = shaped_steps / max(1, len(non_terminal_rewards))
        return _clip01(shaped_ratio)


class HardToMeasureProxyRubric(Rubric):
    """Checks whether reward captures strategic progress signals.

    We treat these as hard-to-measure but meaningful proxies:
    - opponent utility trend feedback (`delta_opponent_utility`)
    - coalition progress / partial agreements (`last_opponent_response`)
    """

    name = "hard_to_measure_proxy"

    def score(self, trace: EpisodeTrace) -> float:
        if not trace.infos:
            return 0.0

        delta_count = 0
        coalition_count = 0
        for info in trace.infos:
            if "delta_opponent_utility" in info:
                delta_count += 1
            if str(info.get("last_opponent_response", "")).lower() in {"partial_accept", "accepted"}:
                coalition_count += 1

        delta_ratio = delta_count / max(1, len(trace.infos))
        coalition_ratio = coalition_count / max(1, len(trace.infos))
        # Strongly weight presence of utility-trend signals.
        return _clip01(0.7 * delta_ratio + 0.3 * coalition_ratio)


class AntiGamingRubric(Rubric):
    """Penalizes traces that earn reward via obvious exploit patterns."""

    name = "anti_gaming"

    def score(self, trace: EpisodeTrace) -> float:
        if not trace.infos:
            return 0.0

        invalid = 0
        accept_abuse = 0
        repeated = 0
        success = 0

        for info in trace.infos:
            if bool(info.get("invalid_proposal", False)):
                invalid += 1
            if bool(info.get("accept_blocked", False)):
                accept_abuse += 1
            if str(info.get("last_opponent_response", "")).lower() == "repeated_proposal":
                repeated += 1
            if bool(info.get("success", False)):
                success += 1

        total = max(1, len(trace.infos))
        exploit_rate = (invalid + accept_abuse + repeated) / total
        success_rate = success / total
        # High score if exploit patterns are rare and success occurs.
        return _clip01((1.0 - exploit_rate) * 0.75 + success_rate * 0.25)


class RubricWeightedSum(Rubric):
    """Weighted composition of child rubrics."""

    name = "weighted_sum"

    def __init__(self, children: Sequence[Tuple[Rubric, float]]) -> None:
        if not children:
            raise ValueError("children cannot be empty")
        total_weight = sum(float(weight) for _, weight in children)
        if total_weight <= 0:
            raise ValueError("weights must sum to > 0")
        self.children = list(children)
        self.total_weight = total_weight

    def score(self, trace: EpisodeTrace) -> float:
        weighted = 0.0
        for rubric, weight in self.children:
            weighted += float(weight) * rubric.score(trace)
        return _clip01(weighted / self.total_weight)

    def score_with_breakdown(self, trace: EpisodeTrace) -> Dict[str, float]:
        breakdown: Dict[str, float] = {}
        for rubric, _weight in self.children:
            breakdown[rubric.name] = _clip01(rubric.score(trace))
        breakdown["composite"] = _clip01(self.score(trace))
        return breakdown


class ComputeBazaarRubric(RubricWeightedSum):
    """OpenEnv-style composable rubric for this environment."""

    name = "compute_bazaar_composite"

    def __init__(self) -> None:
        super().__init__(
            children=[
                (DenseSignalRubric(), 0.35),
                (HardToMeasureProxyRubric(), 0.30),
                (AntiGamingRubric(), 0.35),
            ]
        )


def summarize_breakdowns(breakdowns: Sequence[Dict[str, float]]) -> Dict[str, float]:
    """Average rubric breakdowns over many episodes."""
    if not breakdowns:
        return {"dense_signal": 0.0, "hard_to_measure_proxy": 0.0, "anti_gaming": 0.0, "composite": 0.0}

    keys = sorted({k for b in breakdowns for k in b.keys()})
    return {k: float(mean([b.get(k, 0.0) for b in breakdowns])) for k in keys}
