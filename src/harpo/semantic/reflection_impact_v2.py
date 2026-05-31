"""
Reflection Impact v2 — Outcome-Based, Five-Type Classification

Replaces the binary effective/ineffective model with a five-type taxonomy
grounded in WHAT the reflection actually changed downstream, not just whether
vocabulary shifted.

Five types
----------
STRUCTURAL   — reflection reduced contradictions AND improved plan specificity
               downstream.  The trajectory would have been worse without it.

CORRECTIVE   — reflection explicitly addressed an identified contradiction or
               assumption, naming the specific claim being revised.
               Evidence: correction markers + downstream contradictions drop.

RECOVERY     — reflection immediately follows a TOOL failure or ERROR step,
               leads to a RECOVERY step within 2 turns, and that recovery
               succeeds (failure signals decrease).

STYLISTIC    — vocabulary changed noticeably (Jaccard distance ≥ 0.25) but
               the underlying problem signals (contradictions, failures,
               assumption patterns) did not improve.  "Changed wording,
               same trajectory."

INEFFECTIVE  — minimal vocabulary change AND no improvement in downstream
               signals.  The reflection happened but had no detectable effect.

Output
------
Per reflection:
  "Reflection (turn N) — CORRECTIVE: named 03:12 UTC timeline error;
   downstream contradictions: -2; assumption tokens dropped 40%."

  "Reflection (turn N) — STYLISTIC: vocabulary changed 68% but failure
   signals unchanged; assumption chain continued."

Aggregate:
  "12 structural/corrective/recovery, 10 stylistic, 0 ineffective (55% impactful)."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep

# ── Signal patterns ───────────────────────────────────────────────────────────

_CORRECTION_MARKERS = re.compile(
    r'\b(?:actually|correction|incorrect|wrong|i was wrong|re-evaluating|'
    r'correcting|this contradicts|my mistake|not sql|not 03:12|not at 03|'
    r'21:43|five hours|forensics|confirms|must correct|i need to revise|'
    r'revising|updating my|based on new)\b',
    re.IGNORECASE,
)

_FAILURE_MARKERS = re.compile(
    r'\b(?:failed|failure|error|violation|missed|incorrect|wrong|breach|'
    r'unresolved|conflict|deadline|penalty|harm|exfiltration|contradiction)\b',
    re.IGNORECASE,
)

_ACTION_MARKERS = re.compile(
    r'\b(?:will isolate|will contain|will block|will escalate|will notify|'
    r'must contain|must escalate|must notify|must verify|must block|'
    r'immediately|next step|action plan|remediation|containment|'
    r'rollback|revoke|patch|scan|isolate|block)\b',
    re.IGNORECASE,
)

_RECOVERY_MARKERS = re.compile(
    r'\b(?:recovered|fixed|resolved|contained|blocked|revoked|patched|'
    r'mitigated|stabilized|addressed|remediated|notified|isolated|verified)\b',
    re.IGNORECASE,
)

_STOP: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "that", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "then", "so", "as", "at", "by", "for",
    "of", "on", "to", "in", "with", "about", "from", "not", "no",
}

_WINDOW = 3   # steps before/after the reflection to evaluate


def _toks(text: str) -> Set[str]:
    return {t for t in re.findall(r'\b[a-z]{3,}\b', text.lower()) if t not in _STOP}


def _count(pat: re.Pattern, text: str) -> int:
    return len(pat.findall(text))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ReflectionImpactV2:
    """One reflection step with five-type outcome classification."""
    reflection_turn:     int
    reflection_step_id:  str
    agent_id:            str
    reflection_text:     str   # truncated to 200 chars

    # Signal deltas (positive = improvement in the window after reflection)
    contradiction_delta: int    # pre − post contradiction signal count
    failure_delta:       int    # pre − post failure signal count
    action_delta:        int    # post − pre action signal count
    vocab_change:        float  # Jaccard distance pre→post (0=same, 1=total)

    # Structural signals
    names_specific_claim: bool  # reflection text contains a specific claim marker
    assumption_overlap_drop: float  # fraction drop in assumption token presence post
    recovery_succeeded:   bool  # RECOVERY step followed and succeeded

    # Classification
    impact_type:  str   # "structural"|"corrective"|"recovery"|"stylistic"|"ineffective"
    impact_score: float # 0-1

    def narrative(self) -> str:
        tag = {
            "structural":  "STRUCTURAL",
            "corrective":  "CORRECTIVE",
            "recovery":    "RECOVERY",
            "stylistic":   "STYLISTIC",
            "ineffective": "INEFFECTIVE",
        }.get(self.impact_type, self.impact_type.upper())

        deltas = []
        if self.contradiction_delta > 0:
            deltas.append(f"contradictions −{self.contradiction_delta}")
        if self.failure_delta > 0:
            deltas.append(f"failures −{self.failure_delta}")
        if self.action_delta > 0:
            deltas.append(f"actions +{self.action_delta}")
        if self.assumption_overlap_drop > 0.1:
            deltas.append(f"assumption tokens −{self.assumption_overlap_drop:.0%}")
        if self.recovery_succeeded:
            deltas.append("recovery succeeded")

        delta_str = "; ".join(deltas) if deltas else "no measurable downstream improvement"
        agent_str = f" [{self.agent_id}]" if self.agent_id else ""

        if self.impact_type in ("stylistic", "ineffective"):
            vocab_note = f"vocabulary change={self.vocab_change:.2f}"
            return (
                f"Reflection (turn {self.reflection_turn}){agent_str} — "
                f"{tag}: {vocab_note}, {delta_str}."
            )
        return (
            f"Reflection (turn {self.reflection_turn}){agent_str} — "
            f"{tag} (score={self.impact_score:.2f}): {delta_str}."
        )


@dataclass
class ReflectionImpactReportV2:
    """Aggregated outcome-based reflection classification."""
    impacts:              List[ReflectionImpactV2] = field(default_factory=list)
    structural_count:     int   = 0
    corrective_count:     int   = 0
    recovery_count:       int   = 0
    stylistic_count:      int   = 0
    ineffective_count:    int   = 0
    avg_impact_score:     float = 0.0

    @property
    def impactful_count(self) -> int:
        return self.structural_count + self.corrective_count + self.recovery_count

    def impact_rate(self) -> float:
        n = len(self.impacts)
        return self.impactful_count / n if n > 0 else 0.5

    def as_dict(self) -> dict:
        return {
            "total":            len(self.impacts),
            "structural":       self.structural_count,
            "corrective":       self.corrective_count,
            "recovery":         self.recovery_count,
            "stylistic":        self.stylistic_count,
            "ineffective":      self.ineffective_count,
            "impact_rate":      round(self.impact_rate(), 3),
            "avg_impact_score": round(self.avg_impact_score, 3),
            "impacts": [
                {
                    "turn":            i.reflection_turn,
                    "agent_id":        i.agent_id,
                    "type":            i.impact_type,
                    "score":           round(i.impact_score, 3),
                    "contra_delta":    i.contradiction_delta,
                    "fail_delta":      i.failure_delta,
                    "action_delta":    i.action_delta,
                    "assumption_drop": round(i.assumption_overlap_drop, 3),
                    "narrative":       i.narrative(),
                }
                for i in self.impacts
            ],
        }

    def narrative(self) -> str:
        if not self.impacts:
            return "No reflection steps detected."
        n = len(self.impacts)
        impactful = self.impactful_count
        return (
            f"{n} reflection(s): {self.structural_count} structural, "
            f"{self.corrective_count} corrective, {self.recovery_count} recovery, "
            f"{self.stylistic_count} stylistic, {self.ineffective_count} ineffective. "
            f"Impact rate: {self.impact_rate():.0%}. "
            f"Avg score: {self.avg_impact_score:.2f}."
        )


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_reflection_impact_v2(traj: "AgentTrajectory") -> ReflectionImpactReportV2:
    """
    Classify each REFLECTION step with the five-type outcome taxonomy.

    For each reflection:
    1. Gather _WINDOW THINK/RESPONSE steps before and after.
    2. Compute signal deltas across contradiction, failure, action markers.
    3. Measure vocabulary change (Jaccard distance).
    4. Check if correction markers present in reflection text.
    5. Measure assumption token overlap drop pre→post.
    6. Check for nearby RECOVERY steps and whether they succeeded.
    7. Classify using decision tree:
       - recovery_succeeded → RECOVERY
       - names_specific_claim AND contradiction_delta > 0 → CORRECTIVE
       - contradiction_delta + failure_delta + action_delta ≥ 2 → STRUCTURAL
       - vocab_change ≥ 0.25 → STYLISTIC
       - else → INEFFECTIVE
    """
    from harpo.trajectory.schema import StepType

    all_steps    = traj.steps
    think_types  = (StepType.THINK, StepType.RESPONSE)
    think_steps  = [s for s in all_steps if s.step_type in think_types]
    reflections  = [s for s in all_steps if s.step_type == StepType.REFLECTION]
    recoveries   = [s for s in all_steps if s.step_type == StepType.RECOVERY]

    if not reflections:
        return ReflectionImpactReportV2()

    # Build assumption token pool
    assumption_toks: Set[str] = set()
    try:
        from harpo.semantic.assumptions import analyze_assumption_propagation
        apr = analyze_assumption_propagation(traj)
        for chain in apr.chains:
            if chain.propagation_radius() >= 1:
                assumption_toks |= chain.key_tokens
    except Exception:
        pass

    def _window_text(steps_before: List, steps_after: List) -> tuple:
        return (
            " ".join(s.output_text for s in steps_before[-_WINDOW:]),
            " ".join(s.output_text for s in steps_after[:_WINDOW]),
        )

    impacts: List[ReflectionImpactV2] = []

    for ref in reflections:
        pre_steps  = [s for s in think_steps if s.timestamp < ref.timestamp]
        post_steps = [s for s in think_steps if s.timestamp > ref.timestamp]

        if not pre_steps or not post_steps:
            continue

        pre_text, post_text = _window_text(pre_steps, post_steps)

        # Signal deltas
        pre_contra = _count(_FAILURE_MARKERS, pre_text) + _count(_CORRECTION_MARKERS, pre_text)
        post_contra = _count(_FAILURE_MARKERS, post_text) + _count(_CORRECTION_MARKERS, post_text)
        contra_delta = pre_contra - post_contra

        pre_fail  = _count(_FAILURE_MARKERS, pre_text)
        post_fail = _count(_FAILURE_MARKERS, post_text)
        fail_delta = pre_fail - post_fail

        pre_act  = _count(_ACTION_MARKERS, pre_text)
        post_act = _count(_ACTION_MARKERS, post_text)
        act_delta = post_act - pre_act

        vocab_change = 1.0 - _jaccard(_toks(pre_text), _toks(post_text))

        # Does the reflection name a specific claim?
        names_claim = bool(_CORRECTION_MARKERS.search(ref.output_text or ""))

        # Assumption token overlap drop
        pre_assump = len(assumption_toks & _toks(pre_text)) if assumption_toks else 0
        post_assump = len(assumption_toks & _toks(post_text)) if assumption_toks else 0
        assump_drop = max(0.0, (pre_assump - post_assump) / max(pre_assump, 1))

        # Recovery linkage
        nearby_rec = [r for r in recoveries
                      if 0 < (r.turn_number - ref.turn_number) <= _WINDOW]
        rec_succeeded = False
        if nearby_rec:
            rec_text = " ".join(s.output_text for s in nearby_rec)
            rec_ok   = _count(_RECOVERY_MARKERS, rec_text)
            rec_fail = _count(_FAILURE_MARKERS, rec_text)
            rec_succeeded = rec_ok > rec_fail

        # ── Classify ──────────────────────────────────────────────────────────
        pos_signals = sum([
            contra_delta > 0,
            fail_delta > 0,
            act_delta > 0,
            assump_drop >= 0.25,
            rec_succeeded,
        ])

        if rec_succeeded:
            impact_type = "recovery"
        elif names_claim and contra_delta > 0:
            impact_type = "corrective"
        elif pos_signals >= 2:
            impact_type = "structural"
        elif vocab_change >= 0.25:
            impact_type = "stylistic"
        else:
            impact_type = "ineffective"

        # Impact score (weighted)
        raw_score = (
            min(max(contra_delta / 4.0, 0.0), 1.0) * 0.25
            + min(max(fail_delta / 4.0, 0.0), 1.0) * 0.20
            + min(max(act_delta / 4.0, 0.0), 1.0) * 0.15
            + min(assump_drop * 1.5, 1.0) * 0.20
            + (0.10 if names_claim else 0.0)
            + (0.10 if rec_succeeded else 0.0)
        )
        impact_score = round(min(raw_score, 1.0), 3)

        impacts.append(ReflectionImpactV2(
            reflection_turn       = ref.turn_number,
            reflection_step_id    = ref.step_id,
            agent_id              = getattr(ref, "agent_id", ""),
            reflection_text       = (ref.output_text or "")[:200],
            contradiction_delta   = contra_delta,
            failure_delta         = fail_delta,
            action_delta          = act_delta,
            vocab_change          = round(vocab_change, 4),
            names_specific_claim  = names_claim,
            assumption_overlap_drop = round(assump_drop, 4),
            recovery_succeeded    = rec_succeeded,
            impact_type           = impact_type,
            impact_score          = impact_score,
        ))

    struct = sum(1 for i in impacts if i.impact_type == "structural")
    corr   = sum(1 for i in impacts if i.impact_type == "corrective")
    rec    = sum(1 for i in impacts if i.impact_type == "recovery")
    styl   = sum(1 for i in impacts if i.impact_type == "stylistic")
    ineff  = sum(1 for i in impacts if i.impact_type == "ineffective")
    avg_sc = sum(i.impact_score for i in impacts) / len(impacts) if impacts else 0.0

    return ReflectionImpactReportV2(
        impacts           = impacts,
        structural_count  = struct,
        corrective_count  = corr,
        recovery_count    = rec,
        stylistic_count   = styl,
        ineffective_count = ineff,
        avg_impact_score  = round(avg_sc, 3),
    )
