"""
Semantic Coherence Scoring

Measures whether the agent's reasoning stays topically coherent across turns.

Core topic = significant tokens from the first 3 THINK/RESPONSE steps.
Each subsequent step is scored on:
- overlap_with_core: Jaccard similarity to core topic vocabulary
- overlap_with_prev: Jaccard similarity to immediately previous step

Drift: overlap with core falls below 0.10 (after the core is established).
Return: drift followed by overlap recovering above 0.15.

Low coherence suggests the agent wandered from the task, got sidetracked,
or lost context mid-trajectory.

No external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

_STOP_WORDS: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "that", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "then", "so", "as", "at", "by", "for",
    "of", "on", "to", "in", "with", "about", "from", "not", "no",
    "what", "which", "who", "when", "where", "how", "very", "just",
    "also", "can", "all", "any", "more", "into", "than", "here",
    "like", "get", "got", "let", "use", "used", "new", "now",
    "one", "two", "first", "last", "next", "back", "each",
}

_DRIFT_THRESHOLD   = 0.10  # overlap below this = drift (fixed baseline)
_RETURN_THRESHOLD  = 0.15  # overlap above this after drift = return
_MIN_TOKENS        = 5     # skip steps with fewer significant tokens
_SLIDING_WINDOW    = 5     # steps to use for rolling core (activated for long trajs)
_LONG_TRAJ_STEPS   = 10   # threshold: above this count, use sliding core


@dataclass
class TurnCoherence:
    turn_number:        int
    step_id:            str
    overlap_with_core:  float
    overlap_with_prev:  float
    is_drift:           bool
    is_return:          bool


@dataclass
class CoherenceResult:
    turn_coherence:     List[TurnCoherence] = field(default_factory=list)
    core_topic_tokens:  Set[str] = field(default_factory=set)
    avg_core_overlap:   float = 0.0
    drift_events:       int = 0
    return_events:      int = 0
    overall_coherence:  float = 0.0

    def drift_rate(self) -> float:
        n = len(self.turn_coherence)
        return self.drift_events / n if n > 0 else 0.0


def _significant_tokens(text: str) -> Set[str]:
    tokens = re.findall(r'\b[a-z][a-z]{3,}\b', text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _score_single_agent_coherence(
    think_steps: list,
    step_token_sets: list,
    use_sliding: bool,
) -> "CoherenceResult":
    """
    Internal: compute coherence for a list of think steps.
    Shared by both single-agent and per-agent multi-agent paths.
    """
    seed_count  = min(3, len(think_steps))
    core_tokens: Set[str] = set()
    for i in range(seed_count):
        core_tokens |= step_token_sets[i]

    turn_results: List[TurnCoherence] = []
    prev_tokens: Optional[Set[str]] = None
    prev_was_drift = False
    drift_events   = 0
    return_events  = 0

    for i, step in enumerate(think_steps):
        step_tokens = step_token_sets[i]

        if use_sliding and i >= seed_count:
            window_start = max(0, i - _SLIDING_WINDOW)
            sliding_core: Set[str] = set()
            for j in range(window_start, i):
                sliding_core |= step_token_sets[j]
            effective_core = sliding_core
        else:
            effective_core = core_tokens

        core_overlap = _jaccard(step_tokens, effective_core)
        prev_overlap = _jaccard(step_tokens, prev_tokens) if prev_tokens is not None else 1.0

        is_drift = (
            len(step_tokens) >= _MIN_TOKENS
            and core_overlap < _DRIFT_THRESHOLD
            and i >= seed_count
        )
        is_return = prev_was_drift and core_overlap >= _RETURN_THRESHOLD

        if is_drift:
            drift_events += 1
        if is_return:
            return_events += 1

        turn_results.append(TurnCoherence(
            turn_number       = step.turn_number,
            step_id           = step.step_id,
            overlap_with_core = round(core_overlap, 4),
            overlap_with_prev = round(prev_overlap, 4),
            is_drift          = is_drift,
            is_return         = is_return,
        ))

        prev_tokens    = step_tokens
        prev_was_drift = is_drift

    eval_results = turn_results[seed_count:] if len(turn_results) > seed_count else turn_results
    avg_core = (
        sum(r.overlap_with_core for r in eval_results) / len(eval_results)
        if eval_results else 1.0
    )

    drift_penalty = drift_events * 0.10
    return_credit = return_events * 0.03
    base          = min(avg_core * 2.0, 1.0)
    overall       = max(0.0, min(1.0, base - drift_penalty + return_credit))

    return CoherenceResult(
        turn_coherence    = turn_results,
        core_topic_tokens = core_tokens,
        avg_core_overlap  = round(avg_core, 4),
        drift_events      = drift_events,
        return_events     = return_events,
        overall_coherence = round(overall, 4),
    )


def score_semantic_coherence(traj: "AgentTrajectory") -> CoherenceResult:
    """
    Score semantic coherence across THINK and RESPONSE steps.

    Multi-agent mode (agent_id-tagged steps):
      Each agent is scored independently against its own vocabulary baseline
      (role-aware coherence).  Cross-agent vocabulary difference is NOT
      penalised as drift — a forensics agent using different tokens than a
      compliance agent is healthy specialisation, not incoherence.
      The overall_coherence is a step-count-weighted average of per-agent scores.

    Single-agent / short trajectory:
      Fixed core topic built from first 3 steps (original behaviour).

    Long trajectory (> _LONG_TRAJ_STEPS steps):
      Sliding core window of previous _SLIDING_WINDOW steps — tracks the
      current phase rather than the initial topic.
    """
    from harpo.trajectory.schema import StepType

    think_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE)
        and len(s.output_text.split()) >= 5
    ]

    if len(think_steps) < 2:
        return CoherenceResult(overall_coherence=1.0)

    agent_ids = {getattr(s, "agent_id", "") for s in think_steps} - {""}
    is_multiagent = len(agent_ids) > 1

    # ── Multi-agent: score each agent independently, then aggregate ───────────
    if is_multiagent and len(agent_ids) >= 2:
        from collections import defaultdict
        agent_step_map: dict = defaultdict(list)
        for s in think_steps:
            aid = getattr(s, "agent_id", "")
            agent_step_map[aid].append(s)

        per_agent_results: list = []
        per_agent_weights: list = []

        for aid, a_steps in agent_step_map.items():
            if len(a_steps) < 2:
                # Too few steps for coherence — treat as coherent (agent is specialised)
                per_agent_results.append(CoherenceResult(overall_coherence=0.85))
                per_agent_weights.append(len(a_steps))
                continue
            a_tok_sets = [_significant_tokens(s.output_text) for s in a_steps]
            use_sliding = len(a_steps) > _LONG_TRAJ_STEPS
            result = _score_single_agent_coherence(a_steps, a_tok_sets, use_sliding)
            per_agent_results.append(result)
            per_agent_weights.append(len(a_steps))

        total_weight = sum(per_agent_weights) or 1
        # Weighted average of per-agent overall_coherence
        weighted_overall = sum(
            r.overall_coherence * w
            for r, w in zip(per_agent_results, per_agent_weights)
        ) / total_weight
        # Merge all turn_coherence lists for detailed records
        all_turns: List[TurnCoherence] = []
        for r in per_agent_results:
            all_turns.extend(r.turn_coherence)
        all_turns.sort(key=lambda tc: tc.turn_number)

        total_drift   = sum(r.drift_events for r in per_agent_results)
        total_return  = sum(r.return_events for r in per_agent_results)
        all_core_toks: Set[str] = set()
        for r in per_agent_results:
            all_core_toks |= r.core_topic_tokens
        avg_core_all = (
            sum(r.avg_core_overlap * w for r, w in zip(per_agent_results, per_agent_weights))
            / total_weight
        )

        return CoherenceResult(
            turn_coherence    = all_turns,
            core_topic_tokens = all_core_toks,
            avg_core_overlap  = round(avg_core_all, 4),
            drift_events      = total_drift,
            return_events     = total_return,
            overall_coherence = round(weighted_overall, 4),
        )

    # ── Single-agent path ─────────────────────────────────────────────────────
    use_sliding   = len(think_steps) > _LONG_TRAJ_STEPS
    step_tok_sets = [_significant_tokens(s.output_text) for s in think_steps]
    return _score_single_agent_coherence(think_steps, step_tok_sets, use_sliding)
