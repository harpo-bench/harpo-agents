"""
HARPO-Open Behavioral Metrics

10 evaluation families, each computing one DimensionScore from a trajectory.

Design principle: every metric answers a question about the *process*,
not merely the final output. Inspired by HARPO CHARM/STAR/MAVEN but
generalized from conversational recommendation to any long-horizon agent.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

from .schema import (
    AgentTrajectory, DimensionScore, FailureMode, FailureReport,
    MemoryAccess, StepOutcome, StepType, TrajectoryStep,
)

# Semantic analyzers — imported lazily inside each function so that missing
# the src/semantic package never crashes the baseline metric computation.
def _semantic_contradictions(traj: AgentTrajectory):
    try:
        from harpo.semantic.contradiction import detect_contradictions
        return detect_contradictions(traj)
    except Exception:
        return None

def _semantic_assumptions(traj: AgentTrajectory):
    try:
        from harpo.semantic.assumptions import analyze_assumption_propagation
        return analyze_assumption_propagation(traj)
    except Exception:
        return None

def _semantic_reflections(traj: AgentTrajectory):
    try:
        from harpo.semantic.reflection import analyze_reflection_effectiveness
        return analyze_reflection_effectiveness(traj)
    except Exception:
        return None

def _semantic_coherence(traj: AgentTrajectory):
    try:
        from harpo.semantic.coherence import score_semantic_coherence
        return score_semantic_coherence(traj)
    except Exception:
        return None

def _collaboration_intelligence(traj: AgentTrajectory):
    try:
        from harpo.semantic.collaboration_intelligence import analyze_collaboration
        return analyze_collaboration(traj)
    except Exception:
        return None


# ================================================================
# 1. REASONING STABILITY
#    Measures whether the agent's logic is consistent across turns.
#    Detects: self-contradictions, flip-flopping on conclusions,
#    claim volatility over the trajectory.
# ================================================================

def score_reasoning_stability(traj: AgentTrajectory) -> DimensionScore:
    """
    Consistency of logical claims across the trajectory.

    Algorithm:
    - Extract think/response steps.
    - Detect lexical contradictions ("X is Y" vs "X is not Y").
    - Penalise goal-flip: agent changes primary plan direction.
    - Reward maintaining core reasoning chain.
    """
    think_steps = [s for s in traj.steps if s.step_type in (StepType.THINK, StepType.RESPONSE)]
    if len(think_steps) < 2:
        return DimensionScore(value=1.0, explanation="Too few reasoning steps to evaluate.")

    texts = [s.output_text.lower() for s in think_steps]

    # Contradiction pairs detection (simple but effective)
    negation_patterns = [
        (r"\b(\w+)\s+is\s+(?:not|never|incorrect)", r"\b\1\s+is\s+(?!not|never|incorrect)\w+"),
        (r"\bdo\s+not\s+(\w+)", r"\bdo\s+\1\b"),
        (r"\bcan(?:not|'t)\s+(\w+)", r"\bcan\s+\1\b"),
    ]

    contradiction_count = 0
    evidence = []
    for i in range(len(texts) - 1):
        for pat_neg, pat_pos in negation_patterns:
            neg_matches = re.findall(pat_neg, texts[i])
            for m in neg_matches:
                pos_pat = pat_pos.replace(r"\1", re.escape(m))
                if re.search(pos_pat, texts[i + 1]):
                    contradiction_count += 1
                    evidence.append(think_steps[i].step_id)
                    break

    # Semantic drift: cosine similarity of sequential thought embeddings
    drift_penalties = 0.0
    if all(s.hidden_vector for s in think_steps[:5]):
        for i in range(len(think_steps) - 1):
            v1 = think_steps[i].hidden_vector
            v2 = think_steps[i + 1].hidden_vector
            if v1 and v2:
                sim = _cosine(v1, v2)
                if sim < 0.4:
                    drift_penalties += (0.4 - sim) / 0.4

    # Semantic contradiction detection (reversal markers + plan/negation flips)
    sem = _semantic_contradictions(traj)
    sem_penalty = sem.severity() * 0.35 if sem else 0.0
    sem_note = (
        f"; sem: {sem.reversal_count} reversal(s), {sem.flip_count} flip(s)"
        if sem else ""
    )

    penalty = (contradiction_count * 0.15) + (drift_penalties * 0.05) + sem_penalty
    score = max(0.0, 1.0 - penalty)

    return DimensionScore(
        value=round(score, 4),
        explanation=f"{contradiction_count} lexical contradiction(s); "
                    f"drift={drift_penalties:.2f}{sem_note}",
        evidence_steps=evidence[:5],
        confidence=0.85 if sem else 0.7,
    )


# ================================================================
# 2. CONVERSATIONAL DRIFT
#    Does the agent stray from the user's original intent
#    as the conversation lengthens?
# ================================================================

def score_conversational_drift(traj: AgentTrajectory) -> DimensionScore:
    """
    Measures alignment between each response and the original user intent.

    Low drift = agent stays on-topic throughout.
    High drift = topic wanders or agent injects unrelated sub-goals.
    """
    if not traj.user_intent:
        return DimensionScore(value=0.5, explanation="No user_intent provided; cannot evaluate drift.")

    # Use THINK + RESPONSE steps: in many adapters (Hive, LangGraph) the
    # substantive output lives in THINK steps; RESPONSE may only carry a
    # verdict signal ("accepted", "comprehensive", etc.) from the judge.
    output_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE)
        and len(s.output_text.split()) >= 10  # ignore trivially short steps
    ]
    if not output_steps:
        return DimensionScore(value=0.5, explanation="No substantive output steps found.")

    intent_tokens = set(traj.user_intent.lower().split())

    drift_curve: List[float] = []
    for resp in output_steps:
        resp_tokens = set(resp.output_text.lower().split())
        if not resp_tokens:
            drift_curve.append(1.0)
            continue
        # Intent coverage: what fraction of the intent words appear in the response?
        # This rewards being a superset of the intent (relevant but verbose responses).
        intent_coverage = (
            len(intent_tokens & resp_tokens) / max(len(intent_tokens), 1)
        )
        # Jaccard: classic bidirectional overlap
        jaccard = len(intent_tokens & resp_tokens) / max(len(intent_tokens | resp_tokens), 1)
        # Blend: coverage-weighted (0.6 coverage + 0.4 jaccard)
        overlap = 0.60 * intent_coverage + 0.40 * jaccard
        drift_curve.append(1.0 - overlap)  # high drift = low overlap

    # Penalise trajectories where drift increases monotonically
    if len(drift_curve) >= 3:
        slope = _linear_slope(drift_curve)
        trend_penalty = max(0.0, slope * 0.3)  # rising drift is bad
    else:
        trend_penalty = 0.0

    avg_drift = sum(drift_curve) / len(drift_curve)
    score = max(0.0, 1.0 - avg_drift - trend_penalty)

    return DimensionScore(
        value=round(score, 4),
        explanation=f"Average intent-overlap drift={avg_drift:.3f}; "
                    f"trend_penalty={trend_penalty:.3f}",
    )


# ================================================================
# 3. MEMORY UTILITY
#    Were memory reads actually useful? Did retrieved context
#    improve downstream decisions vs ignoring memory?
# ================================================================

def score_memory_utility(traj: AgentTrajectory) -> DimensionScore:
    """
    Evaluates how productively the agent uses its memory.

    Checks:
    - Hit rate of memory reads (were reads satisfied?).
    - Relevance of retrieved content (if relevance_score is set).
    - Whether memory was used BEFORE a response (recall-then-generate pattern).
    - Penalty for memory writes that are never read back.
    """
    mem_steps = [s for s in traj.steps if s.step_type in (StepType.MEMORY_READ, StepType.MEMORY_WRITE)]

    if not mem_steps:
        return DimensionScore(value=0.5, explanation="No memory operations found.")

    reads = [s for s in mem_steps if s.step_type == StepType.MEMORY_READ and s.memory_access]
    writes = [s for s in mem_steps if s.step_type == StepType.MEMORY_WRITE and s.memory_access]

    if not reads:
        return DimensionScore(value=0.3, explanation="Memory written but never read back.")

    hit_rate = sum(1 for r in reads if r.memory_access.hit) / len(reads)
    avg_relevance = (
        sum(r.memory_access.relevance_score for r in reads if r.memory_access.relevance_score)
        / max(1, len(reads))
    )

    # Recall-then-generate: a read step just before a response step is ideal
    resp_steps = [s for s in traj.steps if s.step_type == StepType.RESPONSE]
    useful_reads = 0
    for resp in resp_steps:
        preceding = [
            s for s in reads
            if s.timestamp < resp.timestamp and resp.timestamp - s.timestamp < 2.0
        ]
        if preceding:
            useful_reads += 1

    recall_gen_rate = useful_reads / max(len(resp_steps), 1)

    # Orphan writes penalty
    write_keys = {w.memory_access.key for w in writes if w.memory_access}
    read_keys  = {r.memory_access.key for r in reads if r.memory_access}
    orphan_ratio = len(write_keys - read_keys) / max(len(write_keys), 1)

    score = (
        0.35 * hit_rate
        + 0.30 * avg_relevance
        + 0.25 * recall_gen_rate
        - 0.10 * orphan_ratio
    )
    score = max(0.0, min(1.0, score))

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"hit_rate={hit_rate:.2f}, avg_relevance={avg_relevance:.2f}, "
            f"recall_gen_rate={recall_gen_rate:.2f}, orphan_ratio={orphan_ratio:.2f}"
        ),
    )


# ================================================================
# 4. ASSUMPTION ACCUMULATION
#    Tracks unverified assumptions the agent makes across turns.
#    High assumption-debt → brittle reasoning, likely downstream failures.
# ================================================================

def score_assumption_accumulation(traj: AgentTrajectory) -> DimensionScore:
    """
    Penalise agents that pile up unverified assumptions.

    Assumption signals (heuristic):
    - Phrases like "I assume", "assuming that", "probably", "likely"
    - Conditional logic "if X then" where X was not confirmed
    - Contradicted assumptions detected in later steps
    """
    assumption_phrases = [
        r"\bI assume\b", r"\bassuming\b", r"\bprobably\b", r"\blikely\b",
        r"\bI think\b", r"\bit seems\b", r"\bperhaps\b", r"\bI believe\b",
        r"\bif\s+\w+\s+is\b", r"\bshould\s+be\b",
    ]

    all_assumptions: List[Tuple[int, str]] = []  # (turn, text)
    for step in traj.steps:
        if step.step_type not in (StepType.THINK, StepType.RESPONSE):
            continue
        for pat in assumption_phrases:
            if re.search(pat, step.output_text, re.IGNORECASE):
                all_assumptions.append((step.turn_number, step.step_id))
                break

    total_turns = max((s.turn_number for s in traj.steps), default=0) + 1
    if total_turns == 0:
        return DimensionScore(value=1.0, explanation="No turns found.")

    explicit_contradictions = sum(
        1 for s in traj.steps
        for a in s.assumptions
        if a.contradicted
    )

    assumption_rate = len(all_assumptions) / total_turns
    contradiction_rate = explicit_contradictions / max(total_turns, 1)

    # Semantic propagation analysis
    prop = _semantic_assumptions(traj)
    if prop:
        propagation_penalty  = prop.propagation_density() * 0.25
        reinforcement_penalty = (prop.reinforced_count / max(prop.total_assumptions, 1)) * 0.15
        radius_penalty       = min(prop.max_radius * 0.03, 0.12)
        sem_note = (
            f"; prop={prop.propagating_count}/{prop.total_assumptions}"
            f" (r={prop.max_radius}), reinf={prop.reinforced_count}"
        )
    else:
        propagation_penalty = reinforcement_penalty = radius_penalty = 0.0
        sem_note = ""

    score = max(0.0, 1.0
        - 0.30 * min(assumption_rate, 1.5)
        - 0.35 * contradiction_rate
        - propagation_penalty
        - reinforcement_penalty
        - radius_penalty
    )

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"rate={assumption_rate:.2f}/turn, contradictions={explicit_contradictions}{sem_note}"
        ),
        evidence_steps=[sid for _, sid in all_assumptions[:5]],
        confidence=0.85 if prop else 0.7,
    )


# ================================================================
# 5. RECOVERY ABILITY
#    When the agent hits a failure, how well does it diagnose
#    and recover vs silently fail or give up?
# ================================================================

def score_recovery_ability(traj: AgentTrajectory) -> DimensionScore:
    """
    Evaluates agent resilience to errors.

    Signals:
    - RECOVERY step following a FAILURE outcome.
    - Retry on tool failure vs abandoning the tool path.
    - Quality of recovery: does the next response improve after recovery?
    """
    failure_steps = [s for s in traj.steps if s.outcome == StepOutcome.FAILURE]
    recovery_steps = [s for s in traj.steps if s.step_type == StepType.RECOVERY]

    if not failure_steps:
        return DimensionScore(value=1.0, explanation="No failures detected; perfect resilience.")

    recovery_rate = min(len(recovery_steps) / len(failure_steps), 1.0)

    # Check if recovery was followed by success
    successful_recoveries = 0
    for rec in recovery_steps:
        subsequent = [
            s for s in traj.steps
            if s.timestamp > rec.timestamp
            and s.outcome == StepOutcome.SUCCESS
            and s.timestamp - rec.timestamp < 3.0
        ]
        if subsequent:
            successful_recoveries += 1

    recovery_quality = successful_recoveries / max(len(recovery_steps), 1)

    # Loop detection: same failure repeated > 2 times in same turn = no recovery
    failure_by_turn: Counter = Counter(s.turn_number for s in failure_steps)
    loop_turns = sum(1 for cnt in failure_by_turn.values() if cnt > 2)
    loop_penalty = loop_turns * 0.15

    score = max(0.0, 0.4 * recovery_rate + 0.6 * recovery_quality - loop_penalty)

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"{len(failure_steps)} failure(s), "
            f"{len(recovery_steps)} recovery attempt(s), "
            f"{successful_recoveries} successful"
        ),
        evidence_steps=[s.step_id for s in recovery_steps[:5]],
    )


# ================================================================
# 6. COLLABORATION QUALITY (multi-agent)
#    Does agent-to-agent communication actually improve outcomes?
#    Based on HARPO MAVEN but generalized to any hand-off pattern.
# ================================================================

def score_collaboration_quality(traj: AgentTrajectory) -> DimensionScore:
    """
    Measures the value of inter-agent interactions.

    When agent_id-tagged steps are present (multi-agent trajectories):
    - Uses CollaborationIntelligenceReport: contribution scores, adoption
      edges, contradiction repairs, stabilization events, silo detection.

    Fallback (single-agent or no agent_id tagging):
    - Handoff coherence, integration rate, redundancy penalty.
    """
    # ── Path 1: semantic collaboration intelligence ───────────────────────────
    agent_ids = {getattr(s, "agent_id", "") for s in traj.steps} - {""}
    if len(agent_ids) >= 2:
        ci = _collaboration_intelligence(traj)
        if ci and ci.collaborative_quality_score > 0:
            return DimensionScore(
                value=round(ci.collaborative_quality_score, 4),
                explanation=(
                    f"agents={len(ci.agent_profiles)}, "
                    f"integration={ci.overall_integration:.2f}, "
                    f"repairs={ci.contradiction_repair_count}, "
                    f"strongest={ci.strongest_contributor or 'N/A'}"
                ),
                confidence=0.85,
            )

    # ── Path 2: handoff-based (legacy single-agent or explicit HANDOFF steps) ─
    handoffs = [s for s in traj.steps if s.step_type == StepType.HANDOFF]

    if not traj.agent_roles or len(traj.agent_roles) < 2:
        return DimensionScore(value=0.5, explanation="Single-agent trajectory; collaboration N/A.")

    if not handoffs:
        return DimensionScore(value=0.5, explanation="No handoff steps found.")

    coherent_handoffs = sum(1 for h in handoffs if len(h.output_text.strip()) > 20)
    coherence_rate = coherent_handoffs / len(handoffs)

    integrated = 0
    for h in handoffs:
        after = [
            s for s in traj.steps
            if s.step_type == StepType.RESPONSE and s.timestamp > h.timestamp
        ]
        if after:
            integrated += 1
    integration_rate = integrated / len(handoffs)

    texts = [h.output_text for h in handoffs if h.output_text]
    redundancy = 0.0
    if len(texts) > 1:
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                sim = SequenceMatcher(None, texts[i], texts[j]).ratio()
                if sim > 0.85:
                    redundancy += 1
        redundancy = min(redundancy / len(texts), 1.0)

    score = max(0.0, min(1.0,
        0.40 * coherence_rate + 0.40 * integration_rate - 0.20 * redundancy
    ))

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"coherence={coherence_rate:.2f}, "
            f"integration={integration_rate:.2f}, "
            f"redundancy={redundancy:.2f}"
        ),
    )


# ================================================================
# 7. REFLECTION USEFULNESS
#    Do self-reflection steps actually change subsequent behavior
#    for the better?
# ================================================================

def score_reflection_usefulness(traj: AgentTrajectory) -> DimensionScore:
    """
    Evaluates whether reflections are actionable.

    A useful reflection:
    1. Follows a failure or sub-optimal step.
    2. Contains a specific corrective plan.
    3. Is followed by a demonstrably different (better) action.
    """
    reflections = [s for s in traj.steps if s.step_type == StepType.REFLECTION]

    if not reflections:
        return DimensionScore(value=0.5, explanation="No reflection steps found.")

    # Trigger rate: reflection after failure or low-quality response
    triggered_reflections = 0
    for ref in reflections:
        prior_failures = [
            s for s in traj.steps
            if s.outcome == StepOutcome.FAILURE and s.timestamp < ref.timestamp
            and ref.timestamp - s.timestamp < 5.0
        ]
        if prior_failures:
            triggered_reflections += 1

    trigger_rate = triggered_reflections / len(reflections)

    # Specificity: reflection text mentions concrete next actions
    action_patterns = [r"\bI will\b", r"\bshould\b", r"\bnext\b", r"\bwill try\b", r"\bcorrect\b"]
    specific_count = 0
    for ref in reflections:
        for pat in action_patterns:
            if re.search(pat, ref.output_text, re.IGNORECASE):
                specific_count += 1
                break
    specificity_rate = specific_count / len(reflections)

    # Behavior change: semantic analysis of pre/post THINK steps
    sem_ref = _semantic_reflections(traj)
    if sem_ref and sem_ref.total > 0:
        semantic_effectiveness = sem_ref.effectiveness_rate()
        semantic_action_rate   = sem_ref.action_oriented_count / sem_ref.total
        avg_change             = sem_ref.avg_behavior_change
        sem_note = (
            f"; sem: {sem_ref.effective_count}/{sem_ref.total} effective"
            f", avg_change={avg_change:.2f}"
        )
    else:
        # Fallback: SequenceMatcher on RESPONSE steps (original approach)
        behavior_change_rate = 0.0
        for ref in reflections:
            prior_resp = [s for s in traj.steps
                         if s.step_type == StepType.RESPONSE and s.timestamp < ref.timestamp]
            after_resp = [s for s in traj.steps
                         if s.step_type == StepType.RESPONSE and s.timestamp > ref.timestamp]
            if prior_resp and after_resp:
                sim = SequenceMatcher(None, prior_resp[-1].output_text, after_resp[0].output_text).ratio()
                if sim < 0.85:
                    behavior_change_rate += 1
        behavior_change_rate /= max(len(reflections), 1)
        semantic_effectiveness = behavior_change_rate
        semantic_action_rate   = specificity_rate
        avg_change             = behavior_change_rate
        sem_note               = ""

    score = (
        0.20 * trigger_rate
        + 0.20 * specificity_rate
        + 0.35 * semantic_effectiveness
        + 0.25 * min(avg_change * 2.0, 1.0)  # scale: 0.5 Jaccard change → 1.0
    )

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"trigger_rate={trigger_rate:.2f}, "
            f"specificity={specificity_rate:.2f}, "
            f"effectiveness={semantic_effectiveness:.2f}{sem_note}"
        ),
        confidence=0.85 if sem_ref and sem_ref.total > 0 else 0.7,
    )


# ================================================================
# 8. LONG-HORIZON RELIABILITY
#    Does performance degrade over long interactions?
#    Checks for quality decay, context window strain, fatigue patterns.
# ================================================================

def score_long_horizon_reliability(traj: AgentTrajectory) -> DimensionScore:
    """
    Measures whether quality is maintained over the full length.

    - Tool success rate per-quartile: should not fall in later quartiles.
    - Response length consistency: large drops may signal context loss.
    - Error clustering: errors concentrated late in trajectory = reliability issue.
    """
    steps = [s for s in traj.steps if s.step_type != StepType.OBSERVATION]
    if len(steps) < 6:
        # Too few steps for meaningful decay analysis — return neutral with mild penalty
        # for short trajectories (they haven't proven long-horizon reliability)
        base = 0.65 if steps else 0.5
        return DimensionScore(
            value=base,
            explanation=f"Only {len(steps)} steps — insufficient for quartile analysis.",
            confidence=0.5,
        )

    # Split into quartiles
    q = len(steps) // 4
    quartiles = [steps[i * q: (i + 1) * q] for i in range(4)]

    # Success rate per quartile
    success_rates = []
    for qsteps in quartiles:
        successes = sum(1 for s in qsteps if s.outcome == StepOutcome.SUCCESS)
        success_rates.append(successes / max(len(qsteps), 1))

    # Trend: success should not drop significantly
    if len(success_rates) == 4:
        early_avg = (success_rates[0] + success_rates[1]) / 2
        late_avg  = (success_rates[2] + success_rates[3]) / 2
        decay = max(0.0, early_avg - late_avg)
    else:
        decay = 0.0

    # Response length stability (CV of response lengths)
    resp_lengths = [
        len(s.output_text) for s in traj.steps
        if s.step_type == StepType.RESPONSE and s.output_text
    ]
    length_cv = _coefficient_of_variation(resp_lengths) if resp_lengths else 0.0

    # Late-error clustering: fraction of failures in last quartile
    all_failures = [s for s in steps if s.outcome == StepOutcome.FAILURE]
    late_failures = [s for s in quartiles[-1] if s.outcome == StepOutcome.FAILURE]
    late_failure_ratio = (
        len(late_failures) / max(len(all_failures), 1) if all_failures else 0.0
    )

    score = max(0.0,
        1.0 - 0.5 * decay
            - 0.2 * min(length_cv, 1.0)
            - 0.3 * late_failure_ratio
    )

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"quartile_success_rates={[round(r, 2) for r in success_rates]}, "
            f"decay={decay:.3f}, length_cv={length_cv:.3f}"
        ),
    )


# ================================================================
# 9. TRAJECTORY COHERENCE
#    Does the overall trajectory form a logical, goal-directed arc?
#    Checks for: plan/execute/verify pattern, unnecessary loops,
#    completeness (all sub-goals addressed).
# ================================================================

def score_trajectory_coherence(traj: AgentTrajectory) -> DimensionScore:
    """
    Holistic coherence of the full trajectory arc.

    A coherent trajectory:
    - Opens with goal understanding.
    - Executes in logical phases (plan → act → verify).
    - Closes with a definitive response.
    - Does not revisit resolved sub-goals.
    """
    if not traj.steps:
        return DimensionScore(value=0.0, explanation="Empty trajectory.")

    step_types = [s.step_type for s in traj.steps]

    # Phase ordering reward: THINK early, TOOL_CALL mid, RESPONSE last
    first_resp_idx = next((i for i, t in enumerate(step_types) if t == StepType.RESPONSE), None)
    last_think_idx = max((i for i, t in enumerate(step_types) if t == StepType.THINK), default=-1)
    ordering_ok = (
        first_resp_idx is not None
        and last_think_idx < first_resp_idx
    )
    ordering_bonus = 0.2 if ordering_ok else 0.0

    # Loop detection: consecutive identical step types with same output
    loop_score = 0.0
    for i in range(1, len(traj.steps) - 1):
        if (
            traj.steps[i].step_type == traj.steps[i - 1].step_type
            and SequenceMatcher(
                None, traj.steps[i].output_text, traj.steps[i - 1].output_text
            ).ratio() > 0.9
        ):
            loop_score += 0.1
    loop_penalty = min(loop_score, 0.5)

    # Completeness: trajectory ends with a RESPONSE step
    ends_with_response = step_types[-1] == StepType.RESPONSE
    completeness = 0.3 if ends_with_response else 0.0

    # Tool-observation pairing: every TOOL_CALL should have an OBSERVATION
    tool_calls = [s for s in traj.steps if s.step_type == StepType.TOOL_CALL]
    observations = [s for s in traj.steps if s.step_type == StepType.OBSERVATION]
    pairing_ratio = min(len(observations) / max(len(tool_calls), 1), 1.0)

    # Semantic coherence: topic consistency across turns
    sem_coh = _semantic_coherence(traj)
    if sem_coh:
        semantic_score = sem_coh.overall_coherence
        sem_note = (
            f", sem_coherence={semantic_score:.2f}"
            f" (drift={sem_coh.drift_events})"
        )
    else:
        semantic_score = 0.5
        sem_note = ""

    score = (
        0.20 * pairing_ratio     # structural: tool/obs pairing (reduced from 0.25)
        + completeness           # 0 or 0.3
        + ordering_bonus         # 0 or 0.2
        - loop_penalty           # 0 to 0.5
        + 0.15 * semantic_score  # semantic coherence contribution
    )
    score = max(0.0, min(1.0, score + 0.25))  # base credit

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"ordering={ordering_ok}, ends_response={ends_with_response}, "
            f"tool_pairing={pairing_ratio:.2f}, loop_penalty={loop_penalty:.2f}{sem_note}"
        ),
        confidence=0.85 if sem_coh else 0.7,
    )


# ================================================================
# 10. USER-ALIGNED INTERACTION QUALITY
#     Borrowing directly from HARPO CHARM but lifted to behavioral level:
#     relevance, engagement, and alignment with stated preferences.
# ================================================================

def score_user_aligned_quality(traj: AgentTrajectory) -> DimensionScore:
    """
    Mirrors HARPO's CHARM reward model at trajectory level.

    Dimensions (weighted like CHARM):
    - Relevance  (35%): responses stay on-topic with user_intent.
    - Engagement (15%): responses are substantive, not curt.
    - Satisfaction (35%): absence of failure modes, task completed.
    - Diversity  (15%): non-repetitive across turns.
    """
    responses = [s for s in traj.steps if s.step_type == StepType.RESPONSE]
    if not responses:
        return DimensionScore(value=0.0, explanation="No response steps found.")

    # Relevance: intent coverage + Jaccard blend (same as conversational_drift)
    intent_tokens = set((traj.user_intent or "").lower().split())
    if intent_tokens:
        relevances = []
        for r in responses:
            r_tokens = set(r.output_text.lower().split())
            coverage = len(intent_tokens & r_tokens) / max(len(intent_tokens), 1)
            jaccard  = len(intent_tokens & r_tokens) / max(len(intent_tokens | r_tokens), 1)
            relevances.append(0.60 * coverage + 0.40 * jaccard)
        relevance = sum(relevances) / len(relevances)
    else:
        relevance = 0.5

    # Engagement: average response length normalized (100-500 tokens = ideal)
    lengths = [len(r.output_text.split()) for r in responses]
    avg_len = sum(lengths) / len(lengths) if lengths else 0
    engagement = min(avg_len / 300.0, 1.0) if avg_len < 300 else max(0.5, 1.0 - (avg_len - 300) / 700)

    # Satisfaction: absence of failures, trajectory completed
    failure_rate = sum(1 for s in traj.steps if s.outcome == StepOutcome.FAILURE) / max(len(traj.steps), 1)
    satisfaction = 1.0 - failure_rate

    # Diversity: distinct n-gram ratio across all responses
    all_tokens = " ".join(r.output_text.lower() for r in responses).split()
    bigrams = [tuple(all_tokens[i:i+2]) for i in range(len(all_tokens) - 1)]
    diversity = len(set(bigrams)) / max(len(bigrams), 1)

    # CHARM-style weighted combination
    score = (
        0.35 * relevance
        + 0.15 * engagement
        + 0.35 * satisfaction
        + 0.15 * diversity
    )

    return DimensionScore(
        value=round(score, 4),
        explanation=(
            f"relevance={relevance:.2f}, engagement={engagement:.2f}, "
            f"satisfaction={satisfaction:.2f}, diversity={diversity:.2f}"
        ),
    )


# ================================================================
# Failure analysis
# ================================================================

def detect_failure_modes(traj: AgentTrajectory) -> FailureReport:
    """
    Identifies which failure modes are present and where they originated.
    """
    modes: List[FailureMode] = []
    evidence: List[str] = []

    all_text = " ".join(s.output_text for s in traj.steps).lower()

    # Hallucination indicators
    halluc_phrases = ["i cannot verify", "as of my knowledge", "i'm not sure but", "i think it was"]
    if any(p in all_text for p in halluc_phrases):
        modes.append(FailureMode.HALLUCINATION)

    # Tool misuse: tool call with empty/invalid arguments
    for s in traj.steps:
        if s.step_type == StepType.TOOL_CALL and s.tool_call:
            if not s.tool_call.arguments or s.tool_call.error:
                modes.append(FailureMode.TOOL_MISUSE)
                evidence.append(s.step_id)
                break

    # Context loss: response in late turns has very low overlap with early turns
    turns = traj.turns()
    if len(turns) > 5:
        early_text = " ".join(s.output_text for t in turns[:2] for s in t).lower()
        late_text  = " ".join(s.output_text for t in turns[-2:] for s in t).lower()
        overlap = _token_overlap(early_text, late_text)
        if overlap < 0.05:
            modes.append(FailureMode.CONTEXT_LOSS)

    # Loop detection
    step_types = [s.step_type for s in traj.steps]
    for i in range(len(step_types) - 4):
        window = step_types[i:i+4]
        if len(set(window)) == 1:
            modes.append(FailureMode.LOOP_DETECTED)
            evidence.append(traj.steps[i].step_id)
            break

    # Contradiction
    contradiction_phrases = ["however earlier", "but i said", "correcting my earlier", "contrary to what"]
    if any(p in all_text for p in contradiction_phrases):
        modes.append(FailureMode.CONTRADICTION)

    # Premature stop: no RESPONSE step
    response_steps = [s for s in traj.steps if s.step_type == StepType.RESPONSE]
    if not response_steps:
        modes.append(FailureMode.PREMATURE_STOP)

    # Over-reasoning: > 20 consecutive THINK steps with no action
    consecutive_think = 0
    max_consecutive_think = 0
    for s in traj.steps:
        if s.step_type == StepType.THINK:
            consecutive_think += 1
            max_consecutive_think = max(max_consecutive_think, consecutive_think)
        else:
            consecutive_think = 0
    if max_consecutive_think > 20:
        modes.append(FailureMode.OVER_REASONING)

    # First failure turn
    failure_steps = [s for s in traj.steps if s.outcome == StepOutcome.FAILURE]
    first_failure = failure_steps[0].turn_number if failure_steps else None

    # Cascade: multiple distinct failure modes
    cascade = len(set(modes)) >= 3

    severity = min(len(modes) * 0.15 + (0.3 if cascade else 0), 1.0)

    return FailureReport(
        failure_modes=list(set(modes)),
        first_failure_turn=first_failure,
        cascade_detected=cascade,
        recovery_attempted=bool([s for s in traj.steps if s.step_type == StepType.RECOVERY]),
        recovery_succeeded=bool(
            [s for s in traj.steps
             if s.step_type == StepType.RESPONSE and s.outcome == StepOutcome.SUCCESS]
        ),
        root_cause=modes[0].value if modes else "",
        contributing_steps=evidence[:10],
        severity=round(severity, 4),
    )


# ================================================================
# Internal helpers
# ================================================================

def _cosine(v1: List[float], v2: List[float]) -> float:
    if len(v1) != len(v2):
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    n1 = math.sqrt(sum(a * a for a in v1))
    n2 = math.sqrt(sum(b * b for b in v2))
    return dot / (n1 * n2 + 1e-8)


def _linear_slope(values: List[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / (den + 1e-8)


def _coefficient_of_variation(values: List[float]) -> float:
    if not values or len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
    return std / mean


def _token_overlap(text_a: str, text_b: str) -> float:
    tokens_a = set(text_a.split())
    tokens_b = set(text_b.split())
    return len(tokens_a & tokens_b) / max(len(tokens_a | tokens_b), 1)
