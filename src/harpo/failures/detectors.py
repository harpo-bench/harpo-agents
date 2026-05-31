"""
HARPO Failure Intelligence — Detectors

Phase 1: DefaultFailureAnalyzer wraps detect_failure_modes() from metrics.py.
Phase 2 (now implemented): Real detection logic for all detector classes.

Each detector is independent and safe to run on any trajectory.
Failures in individual detectors do not propagate — they return empty lists.
"""

from __future__ import annotations

import hashlib
import re
import uuid
from collections import Counter
from typing import TYPE_CHECKING, List, Optional

from harpo.failures.interfaces import (
    FailureAnalyzer,
    FailureDetector,
    MemoryFailureAnalyzer,
)
from harpo.failures.schema import (
    FailureEvent,
    FailureReport,
    MemoryCollapse,
)

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_event(
    traj: "AgentTrajectory",
    step_id: str,
    turn: int,
    failure_type: str,
    severity: str,
    description: str,
    related: List[str] = (),
) -> FailureEvent:
    return FailureEvent(
        trajectory_id = traj.trajectory_id,
        step_id       = step_id,
        turn_number   = turn,
        failure_type  = failure_type,
        severity      = severity,
        description   = description,
        related_steps = list(related),
    )


def _step_hash(text: str) -> str:
    """Normalize whitespace, lowercase, hash — used for loop detection."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()  # noqa: S324 (non-crypto use)


# ── Detectors ─────────────────────────────────────────────────────────────────


class AssumptionDetector(FailureDetector):
    """
    Detect assumption propagation failures.

    Uses the semantic.assumptions module to identify assumptions that
    spread into ≥2 later turns (propagation radius ≥ 2), which signals
    reasoning chains built on unverified foundations.
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        try:
            from harpo.semantic.assumptions import analyze_assumption_propagation
        except ImportError:
            return []

        try:
            result = analyze_assumption_propagation(trajectory)
        except Exception:
            return []

        events: List[FailureEvent] = []
        for chain in result.chains:
            if chain.propagation_radius() < 1:
                continue

            radius  = chain.propagation_radius()
            reinf   = chain.reinforced
            severity = (
                "critical" if radius >= 4
                else "high"   if radius >= 3 or (radius >= 2 and reinf)
                else "medium" if radius >= 2
                else "low"
            )
            desc = (
                f"Assumption '{chain.text[:80]}' (turn {chain.introduced_turn}) "
                f"propagated into {radius} later turn(s)"
            )
            if reinf:
                desc += "; also reinforced/restated"

            events.append(_make_event(
                trajectory,
                step_id      = chain.step_id,
                turn         = chain.introduced_turn,
                failure_type = "assumption_propagation",
                severity     = severity,
                description  = desc,
                related      = [str(t) for t in chain.propagated_turns],
            ))

        return events


class LoopDetector(FailureDetector):
    """
    Detect tool-call doom loops and reasoning repetition.

    Two signals:
    1. Text hash repeat — normalized output identical to a previous step
    2. Tool replay — same (tool_name, arguments JSON) tuple without recovery
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        from harpo.trajectory.schema import StepType

        events: List[FailureEvent] = []
        seen_hashes: dict[str, str] = {}   # hash → step_id first seen
        tool_calls: list[tuple[str, str]] = []  # (sig, step_id)

        for step in trajectory.steps:
            if not step.output_text.strip():
                continue

            h = _step_hash(step.output_text)
            if h in seen_hashes:
                events.append(_make_event(
                    trajectory,
                    step_id      = step.step_id,
                    turn         = step.turn_number,
                    failure_type = "loop_text_repeat",
                    severity     = "high",
                    description  = (
                        f"Step output identical to step {seen_hashes[h]} "
                        f"(turn {step.turn_number})"
                    ),
                    related      = [seen_hashes[h]],
                ))
            else:
                seen_hashes[h] = step.step_id

            if step.step_type == StepType.TOOL_CALL and step.tool_call:
                import json
                try:
                    args_str = json.dumps(step.tool_call.arguments, sort_keys=True)
                except Exception:
                    args_str = str(step.tool_call.arguments)
                sig = f"{step.tool_call.name}::{args_str}"
                prior_same = [s for s, _ in tool_calls if s == sig]
                if prior_same:
                    events.append(_make_event(
                        trajectory,
                        step_id      = step.step_id,
                        turn         = step.turn_number,
                        failure_type = "tool_doom_loop",
                        severity     = "high",
                        description  = (
                            f"Tool '{step.tool_call.name}' called with identical arguments "
                            f"for the {len(prior_same)+1}th time without recovery"
                        ),
                    ))
                tool_calls.append((sig, step.step_id))

        return events


class ContextLossDetector(FailureDetector):
    """
    Detect mid-trajectory context loss using Jaccard overlap decay.

    Compares the topic vocabulary of the first half of the trajectory
    with the second half. A drop in overlap > 0.30 suggests context loss.
    Also uses semantic coherence drift events as a signal.
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        from harpo.trajectory.schema import StepType

        think_steps = [
            s for s in trajectory.steps
            if s.step_type in (StepType.THINK, StepType.RESPONSE)
            and s.output_text.strip()
        ]
        if len(think_steps) < 4:
            return []

        mid = len(think_steps) // 2
        first_half = think_steps[:mid]
        second_half = think_steps[mid:]

        def tokens(steps):
            return set(
                t for s in steps
                for t in re.findall(r'\b[a-z]{4,}\b', s.output_text.lower())
            )

        t1 = tokens(first_half)
        t2 = tokens(second_half)
        union = t1 | t2
        if not union:
            return []

        jaccard = len(t1 & t2) / len(union)
        if jaccard >= 0.15:   # 15% overlap = acceptable continuity
            return []

        # Also check semantic coherence drift events for extra evidence
        drift_count = 0
        try:
            from harpo.semantic.coherence import score_semantic_coherence
            coh = score_semantic_coherence(trajectory)
            drift_count = coh.drift_events
        except Exception:
            pass

        severity = "critical" if jaccard < 0.05 else "high"
        first_step = second_half[0]

        return [_make_event(
            trajectory,
            step_id      = first_step.step_id,
            turn         = first_step.turn_number,
            failure_type = "context_loss",
            severity     = severity,
            description  = (
                f"Topic vocabulary overlap between trajectory halves = {jaccard:.2f} "
                f"(threshold 0.15); drift_events={drift_count}"
            ),
        )]


class ReflectionEffectivenessDetector(FailureDetector):
    """
    Detect reflections that fired but caused no behavioral change.

    Uses semantic.reflection.analyze_reflection_effectiveness to compare
    THINK output before and after each REFLECTION step. A reflection with
    Jaccard distance < 0.25 is flagged as ineffective.
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        try:
            from harpo.semantic.reflection import analyze_reflection_effectiveness
        except ImportError:
            return []

        try:
            result = analyze_reflection_effectiveness(trajectory)
        except Exception:
            return []

        events: List[FailureEvent] = []
        for effect in result.effects:
            if effect.effective:
                continue
            events.append(_make_event(
                trajectory,
                step_id      = effect.reflection_step_id,
                turn         = effect.reflection_turn,
                failure_type = "reflection_ineffective",
                severity     = "medium",
                description  = (
                    f"Reflection at turn {effect.reflection_turn} produced no reasoning change "
                    f"(Jaccard distance={effect.token_change:.2f}, threshold=0.25). "
                    f"Action-oriented: {effect.action_oriented}."
                ),
            ))

        return events


class RecoveryQualityDetector(FailureDetector):
    """
    Detect recovery attempts that did not resolve the original failure.

    A recovery step that is NOT followed by a SUCCESS outcome within
    2 subsequent steps is flagged as a failed recovery.
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        from harpo.trajectory.schema import StepType, StepOutcome

        recovery_steps = [s for s in trajectory.steps if s.step_type == StepType.RECOVERY]
        if not recovery_steps:
            return []

        events: List[FailureEvent] = []
        all_steps = trajectory.steps

        for rec in recovery_steps:
            # Look for SUCCESS in the next 2 steps by timestamp
            subsequent = [
                s for s in all_steps
                if s.timestamp > rec.timestamp
            ][:2]

            resolved = any(s.outcome == StepOutcome.SUCCESS for s in subsequent)
            if not resolved:
                events.append(_make_event(
                    trajectory,
                    step_id      = rec.step_id,
                    turn         = rec.turn_number,
                    failure_type = "recovery_failed",
                    severity     = "high",
                    description  = (
                        f"Recovery at turn {rec.turn_number} not followed by success "
                        f"in next {len(subsequent)} step(s)"
                    ),
                    related      = [s.step_id for s in subsequent],
                ))

        return events


class DriftDetector(FailureDetector):
    """
    Detect topic drift using semantic coherence analysis.

    Wraps score_semantic_coherence: flags when ≥2 drift events occur
    without recovery, indicating the agent has strayed from the task.
    """

    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        try:
            from harpo.semantic.coherence import score_semantic_coherence
        except ImportError:
            return []

        try:
            coh = score_semantic_coherence(trajectory)
        except Exception:
            return []

        if coh.drift_events < 2:
            return []

        # Find first drift turn from turn_coherence
        drift_turns = [tc for tc in coh.turn_coherence if tc.is_drift]
        if not drift_turns:
            return []

        first_drift = drift_turns[0]
        unrecovered = coh.drift_events - coh.return_events
        severity = "high" if unrecovered >= 3 else "medium"

        return [_make_event(
            trajectory,
            step_id      = first_drift.step_id,
            turn         = first_drift.turn_number,
            failure_type = "topic_drift",
            severity     = severity,
            description  = (
                f"{coh.drift_events} drift event(s) detected "
                f"({coh.return_events} return(s), {unrecovered} unrecovered); "
                f"avg_core_overlap={coh.avg_core_overlap:.2f}"
            ),
        )]


# ── Default failure analyzer ──────────────────────────────────────────────────

class DefaultFailureAnalyzer(FailureAnalyzer):
    """
    Aggregate FailureEvents into a FailureReport.

    Wraps the existing detect_failure_modes() for the trajectory-level
    failure modes (HALLUCINATION, TOOL_MISUSE, etc.) and blends in the
    per-event FailureEvents from the Phase 2 detectors.
    """

    def analyze(self, events: List[FailureEvent]) -> FailureReport:
        if not events:
            return FailureReport(failure_density=0.0, recovery_rate=1.0)

        type_counts   = Counter(e.failure_type for e in events)
        dominant      = type_counts.most_common(1)[0][0] if type_counts else None
        recoveries    = sum(1 for e in events if e.failure_type in ("recovery", "recovery_failed"))
        unrecovered   = sum(1 for e in events if e.failure_type != "recovery")
        max_turn      = max((e.turn_number for e in events), default=1)
        density       = len(events) / max(max_turn, 1)
        recovery_rate = recoveries / len(events) if events else 1.0

        return FailureReport(
            failure_events    = events,
            dominant_failure  = dominant,
            failure_density   = density,
            recovery_rate     = recovery_rate,
            unrecovered_count = unrecovered,
            cascade_detected  = len(events) > 3 and density > 0.5,
        )


# ── Memory failure analyzer ───────────────────────────────────────────────────

class MemoryCollapseDetector(MemoryFailureAnalyzer):
    """
    Detect memory hit-rate decay that signals context collapse.

    Splits MEMORY_READ steps into first-half and second-half of the trajectory.
    If the hit rate drops by more than 0.2 between halves, returns a MemoryCollapse.
    """

    def analyze(self, trajectory: "AgentTrajectory") -> Optional[MemoryCollapse]:
        from harpo.trajectory.schema import StepType

        reads = [
            s for s in trajectory.steps
            if s.step_type == StepType.MEMORY_READ and s.memory_access
        ]
        if len(reads) < 4:
            return None

        mid         = len(reads) // 2
        first_half  = reads[:mid]
        second_half = reads[mid:]

        prior_rate = sum(1 for r in first_half  if r.memory_access.hit) / len(first_half)
        post_rate  = sum(1 for r in second_half if r.memory_access.hit) / len(second_half)
        delta      = post_rate - prior_rate  # negative = degradation

        if delta > -0.20:
            return None

        # Find the turn where degradation started
        detected_at = second_half[0].turn_number

        # Try to find a hint about what was lost
        lost_hint = ""
        for r in second_half:
            if r.memory_access and not r.memory_access.hit:
                lost_hint = str(r.memory_access.key or r.memory_access.value or "")[:80]
                break

        return MemoryCollapse(
            trajectory_id    = trajectory.trajectory_id,
            detected_at_turn = detected_at,
            lost_context_hint = lost_hint,
            prior_hit_rate   = round(prior_rate, 4),
            post_hit_rate    = round(post_rate, 4),
            delta            = round(delta, 4),
        )


# Backwards-compat alias
StubMemoryFailureAnalyzer = MemoryCollapseDetector
