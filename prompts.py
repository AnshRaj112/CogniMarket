"""System prompt templates for Compute Allocation Bazaar agents.

Three prompts are provided:
- ``build_opponent_prompt``: system prompt for an LLM-based negotiating opponent.
- ``build_oversight_prompt``: system prompt for the lightweight oversight agent.
- ``build_learner_hint``: optional context hint prepended to the learner's input.
"""

from __future__ import annotations

from typing import Dict, List, Sequence


def build_opponent_prompt(
    agent_id: str,
    utility_vector: List[float],
    conversation_history: List[str],
    remaining_pool: Dict[str, float],
    agent_ids: Sequence[str] | None = None,
) -> str:
    """Return a system prompt for an LLM-based opponent.

    The prompt describes the opponent's private utility preferences, the
    current state of negotiations, and encourages strategic but fair play.

    Args:
        agent_id: Agent identifier for the current opponent.
        utility_vector: Private utility weights [gpu_weight, cpu_weight, memory_weight].
            Values are non-negative and sum to 1.
        conversation_history: Recent turns of public conversation, newest last.
        remaining_pool: Remaining unallocated resources per type.
        agent_ids: Full list of participating agent IDs.

    Returns:
        A formatted system-prompt string ready to send to an LLM.
    """
    gpu_w, cpu_w, mem_w = utility_vector
    history_text = "\n".join(conversation_history) if conversation_history else "(no messages yet)"
    pool_str = (
        f"GPU: {remaining_pool.get('gpu', 100):.1f}, "
        f"CPU: {remaining_pool.get('cpu', 100):.1f}, "
        f"Memory: {remaining_pool.get('memory', 100):.1f}"
    )

    all_agent_ids = list(agent_ids) if agent_ids else ["learner", "opponent_1", "opponent_2"]
    if agent_id not in all_agent_ids:
        all_agent_ids.append(agent_id)
    other_agents = [a for a in all_agent_ids if a != agent_id]
    all_agents_str = ", ".join(all_agent_ids)
    proposal_example = "; ".join(
        f"{aid}: gpu <N> cpu <N> memory <N>" for aid in all_agent_ids
    )

    return f"""You are {agent_id}, a strategic compute resource negotiator in the Compute Allocation Bazaar.

== Your Private Utility Preferences ==
You care about three resources: GPU, CPU, and Memory.
Your personal utility weights (private  do NOT reveal these exactly):
  GPU importance:    {gpu_w:.3f}
  CPU importance:    {cpu_w:.3f}
  Memory importance: {mem_w:.3f}

These weights mean you prefer larger shares of resources with higher weights.
Your goal is to maximize your own utility (dot product of your allocation fraction and these weights).

== Negotiation Rules ==
- Participating agents: {all_agents_str}.
- You are {agent_id}; other agents are: {", ".join(other_agents) if other_agents else "(none)"}.
- Total compute pool: 100 units of each resource (GPU, CPU, Memory).
- Each proposal must specify allocations for ALL agents that fit within the pool.
- Proposals use the format:
    {proposal_example}
- You may: accept a proposal, reject it, make a counter-proposal, or send a strategic message.
- A deal closes when ALL agents accept the SAME proposal.

== Current State ==
Remaining unallocated pool: {pool_str}

== Recent Conversation ==
{history_text}

== Behavior Instructions ==
- Think strategically: push for a larger share of resources you care most about.
- Be willing to accept slightly suboptimal deals if the alternative is no deal at all.
- Reveal preferences only indirectly (e.g., say "I value GPU highly" but do not share exact weights).
- Keep messages concise (13 sentences) and negotiation-focused.
- If rounds are running out, be more willing to compromise to close a deal.

Respond with exactly one action: a proposal, acceptance, rejection, counter-offer, or a short strategic message."""


def build_oversight_prompt(
    conversation_history: List[str],
    current_proposal: Dict[str, Dict[str, float]] | None,
    utilities: Dict[str, List[float]] | None = None,
) -> str:
    """Return a system prompt for the oversight agent.

    The oversight agent has access to the full conversation and (optionally)
    all private utilities to reason about fairness and efficiency.

    Args:
        conversation_history: Full negotiation history.
        current_proposal: Active proposal allocation keyed by agent ID, or None.
        utilities: All agents' private utility vectors (optional  full observability mode).

    Returns:
        A formatted system-prompt string.
    """
    history_text = "\n".join(conversation_history) if conversation_history else "(no messages yet)"

    proposal_text = "(no proposal on the table yet)"
    if current_proposal:
        lines = []
        for agent_id, alloc in current_proposal.items():
            lines.append(
                f"  {agent_id}: GPU {alloc.get('gpu', 0):.1f}, "
                f"CPU {alloc.get('cpu', 0):.1f}, "
                f"Memory {alloc.get('memory', 0):.1f}"
            )
        proposal_text = "\n".join(lines)

    utility_text = ""
    if utilities:
        lines = []
        for agent_id, vec in utilities.items():
            lines.append(f"  {agent_id}: gpu={vec[0]:.3f}, cpu={vec[1]:.3f}, memory={vec[2]:.3f}")
        utility_text = "\n== Private Utilities (full-observability mode) ==\n" + "\n".join(lines)

    return f"""You are the Oversight Agent in the Compute Allocation Bazaar.
Your role is to observe the full negotiation and provide neutral, helpful commentary on:
  1. Fairness: are resources distributed proportionally to each agent's needs?
  2. Efficiency: is the overall surplus/utilization high, or is compute being wasted?
  3. Dynamics: is the negotiation converging or stalling?
{utility_text}
== Full Negotiation History ==
{history_text}

== Current Proposal on the Table ==
{proposal_text}

When queried, provide a concise explanation (24 sentences) covering:
- Whether the current proposal is feasible (total allocation  100 per resource).
- Which agent benefits most and why.
- Whether the deal appears fair relative to stated or inferred preferences.
- A recommendation (e.g., "Agent learner should push for more GPU" or "This deal looks efficient  suggest accepting").

Be neutral, factual, and helpful. Avoid advocating for any single agent's interests."""


def build_learner_hint(
    utility_vector: List[float],
    conversation_history: List[str],
    remaining_pool: Dict[str, float],
    rounds_remaining: int,
    difficulty: str = "hard",
    agent_ids: Sequence[str] | None = None,
) -> str:
    """Return a concise context string to prepend to the learner agent's input.

    Includes negotiation memory: analyzes previous rejections and injects
    strategy guidance to prevent stagnation and encourage adaptive proposals.

    Args:
        utility_vector: Learner's private utility weights [gpu, cpu, memory].
        conversation_history: Last few turns of public conversation.
        remaining_pool: Remaining unallocated resources.
        rounds_remaining: How many negotiation rounds are left.
        difficulty: "easy" or "hard" curriculum mode.

    Returns:
        A compact context string.
    """
    gpu_w, cpu_w, mem_w = utility_vector
    history_text = "\n".join(conversation_history) if conversation_history else "(no messages yet)"
    pool_str = (
        f"GPU {remaining_pool.get('gpu', 100):.1f} / "
        f"CPU {remaining_pool.get('cpu', 100):.1f} / "
        f"Memory {remaining_pool.get('memory', 100):.1f}"
    )
    urgency = "LOW" if rounds_remaining > 6 else ("MEDIUM" if rounds_remaining > 2 else "HIGH")

    # --- Negotiation memory analysis (Component 4) ---
    strategy_hints = _build_strategy_hints(conversation_history, rounds_remaining, urgency)

    all_agent_ids = list(agent_ids) if agent_ids else ["learner", "opponent_1", "opponent_2"]
    proposal_template = "; ".join(
        f"{aid}: gpu <N> cpu <N> memory <N>" for aid in all_agent_ids
    )
    agent_count = len(all_agent_ids)
    opponent_ids = [a for a in all_agent_ids if a != "learner"]
    num_opponents = len(opponent_ids)
    base = 100 // max(agent_count, 1)
    rem = 100 - (base * max(agent_count, 1))
    shares = [base + (1 if i < rem else 0) for i in range(agent_count)]
    example_parts = [
        f"{aid}: gpu {shares[i]} cpu {shares[i]} memory {shares[i]}"
        for i, aid in enumerate(all_agent_ids)
    ]
    example_proposal = "; ".join(example_parts)

    return f"""[SYSTEM: Compute Allocation Bazaar | difficulty={difficulty} | rounds_left={rounds_remaining} | urgency={urgency}]
Your private utility weights: GPU={gpu_w:.3f}, CPU={cpu_w:.3f}, Memory={mem_w:.3f}
Remaining pool: {pool_str}

Recent conversation:
{history_text}
{strategy_hints}
You are negotiating with {num_opponents} opponents: {", ".join(opponent_ids) if opponent_ids else "(none)"}.
You MUST include all agents in every proposal.
If any agent is missing, the proposal is INVALID.
Produce a single negotiation action. You MUST use one of these two formats:
1. PROPOSE: {proposal_template}
2. ACCEPT: YES (to accept current deal) or ACCEPT: NO (to reject/counter)

CRITICAL RULES:
- Each resource (gpu, cpu, memory) must sum to EXACTLY 100 across all listed agents.
- Do NOT repeat the same proposal. If rejected, change the allocation.
- Output ONLY the action line. No explanations.

Example: PROPOSE: {example_proposal}"""


def _build_strategy_hints(
    history: List[str],
    rounds_remaining: int,
    urgency: str,
) -> str:
    """Build contextual strategy hints based on negotiation history (Component 4).

    Analyzes what happened in previous rounds to give the model explicit
    guidance on what to do next, preventing stagnation and repetition.
    """
    if not history:
        return "\n[STRATEGY: This is round 1. Propose an allocation that gives opponents generous shares to start negotiation.]\n"

    hints = []

    # Count rejections
    rejections = sum(1 for turn in history if "reject" in turn.lower() and not turn.startswith("learner"))
    accepts = sum(1 for turn in history if "accept" in turn.lower() and "reject" not in turn.lower() and not turn.startswith("learner"))

    if rejections > 0 and accepts == 0:
        hints.append(f"Opponents have rejected {rejections} time(s). INCREASE their shares significantly.")
        hints.append("Give opponents MORE of the resources they seem to want.")

    if accepts > 0:
        hints.append(f"{accepts} opponent(s) have shown willingness to accept. Consider ACCEPT: YES if a deal is on the table.")

    # Anti-repetition
    learner_proposals = [t for t in history if t.startswith("learner:") and "gpu" in t.lower()]
    if len(learner_proposals) >= 2:
        hints.append("WARNING: Do NOT repeat your previous proposal. You must try a DIFFERENT allocation.")

    # Urgency-based hints
    if urgency == "HIGH":
        hints.append("URGENT: Very few rounds left! Make major concessions to close a deal NOW.")
        hints.append("Consider giving opponents 40+ units of each resource.")
    elif urgency == "MEDIUM":
        hints.append("Time is running out. Be more generous to opponents.")

    if not hints:
        return ""

    return "\n[STRATEGY: " + " | ".join(hints) + "]\n"
