"""
Reflection Impact Analysis

Measures whether reflection steps produced *genuine downstream improvement*,
not just superficial behavioral change (the limitation of reflection.py).

reflection.py asks: "Did vocabulary change after this reflection?"
reflection_impact.py asks: "Did the PROBLEMS that triggered the reflection actually improve?"

A reflection is STRUCTURALLY IMPACTFUL when, in the steps following it:
  1. Contradiction density decreases (fewer contradiction markers)
  2. An active assumption propagation chain is broken (tokens stop propagating)
  3. Semantic coherence improves (objective token overlap rises)
  4. A recovery step follows and succeeds (no further failure signals)
  5. Plan specificity increases (concrete action tokens dominate)

A reflection is STYLISTICALLY IMPACTFUL only:
  • Vocabulary changes (high Jaccard distance) but problem signals persist

A reflection is NULL:
  • Same contradictions / assumption patterns appear in subsequent steps
  • No change in coherence, no recovery, no plan change

This yields a 3-tier classification (structural / stylistic / null) per
reflection event, replacing the binary effective/ineffective from reflection.py.

No external dependencies.  Heuristic token analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep

# ── Signal patterns ───────────────────────────────────────────────────────────

# Contradiction / conflict signals that should DECREASE after effective reflection
_CONTRADICTION_SIGNALS = re.compile(
    r'\b(?:but earlier|actually|i was wrong|incorrect|contradicts|'
    r'contrary to|however earlier|re-evaluating|my mistake|'
    r'this conflicts|not sql|not 03:12|the timeline)\b',
    re.IGNORECASE,
)

# Failure signals that should DECREASE after effective reflection
_FAILURE_SIGNALS = re.compile(
    r'\b(?:failed|failure|error|violation|missed|wrong|breach|'
    r'unresolved|conflict|deadline|penalty|harm|exfiltration)\b',
    re.IGNORECASE,
)

# Plan specificity signals — these should INCREASE after reflection
_ACTION_TOKENS = re.compile(
    r'\b(?:will isolate|will contain|will block|will escalate|will notify|'
    r'will patch|will revoke|will scan|will verify|will document|'
    r'must contain|must escalate|must notify|must verify|must block|'
    r'immediately|first step|next action|action plan|remediation|'
    r'containment|incident response|rollback|credential|reset)\b',
    re.IGNORECASE,
)

# Recovery signals in subsequent steps
_RECOVERY_SIGNALS = re.compile(
    r'\b(?:corrected|fixed|resolved|contained|blocked|revoked|patched|'
    r'mitigated|stabilized|addressed|recovered|remediated|notified|isolated)\b',
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
    "what", "which", "who", "when", "where", "how", "very", "just",
    "also", "can", "all", "any", "more", "into", "than", "here",
}

_WINDOW = 3   # steps to look back (pre) and forward (post) around reflection


def _sig_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r'\b[a-z]{3,}\b', text.lower()) if t not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _count_pattern(pattern: re.Pattern, text: str) -> int:
    return len(pattern.findall(text))


def _combine_text(steps: List["TrajectoryStep"]) -> str:
    return " ".join(s.output_text for s in steps)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class ReflectionImpact:
    """Per-reflection causal impact assessment."""
    reflection_step_id:        str
    reflection_turn:           int
    reflection_text:           str     # truncated to 200 chars

    # Pre / post window texts
    pre_window_text:           str
    post_window_text:          str

    # Signal deltas (positive = improvement)
    contradiction_delta:       int     # pre_count - post_count (positive = fewer after)
    failure_signal_delta:      int     # pre_count - post_count (positive = fewer after)
    action_specificity_delta:  int     # post_count - pre_count (positive = more action)
    coherence_delta:           float   # post_overlap - pre_overlap (positive = better)
    vocabulary_change:         float   # Jaccard distance pre→post (0=same, 1=total change)

    # Recovery linkage
    recovery_followed:         bool    # RECOVERY step within _WINDOW turns
    recovery_succeeded:        bool    # no failure signals in recovery step

    # Assumption chain breakage
    assumption_chain_broken:   bool    # propagating assumption tokens absent after reflection

    # Classification
    impact_type:               str     # "structural" | "stylistic" | "null"
    impact_score:              float   # 0-1 composite

    def narrative(self) -> str:
        gains = []
        if self.contradiction_delta > 0:
            gains.append(f"−{self.contradiction_delta} contradiction signal(s)")
        if self.failure_signal_delta > 0:
            gains.append(f"−{self.failure_signal_delta} failure signal(s)")
        if self.action_specificity_delta > 0:
            gains.append(f"+{self.action_specificity_delta} action token(s)")
        if self.coherence_delta > 0.02:
            gains.append(f"+{self.coherence_delta:.2f} coherence")
        if self.assumption_chain_broken:
            gains.append("assumption chain broken")
        if self.recovery_followed and self.recovery_succeeded:
            gains.append("recovery succeeded")
        gains_str = "; ".join(gains) if gains else "no measurable downstream improvement"
        return (
            f"Reflection (turn {self.reflection_turn}): "
            f"{self.impact_type.upper()} impact (score={self.impact_score:.2f}). "
            f"{gains_str}."
        )


@dataclass
class ReflectionImpactReport:
    """Aggregated reflection impact for one trajectory."""
    impacts:              List[ReflectionImpact] = field(default_factory=list)
    structural_count:     int   = 0
    stylistic_count:      int   = 0
    null_count:           int   = 0
    avg_impact_score:     float = 0.0
    most_impactful_turn:  Optional[int] = None
    least_impactful_turn: Optional[int] = None

    def impact_rate(self) -> float:
        n = len(self.impacts)
        return self.structural_count / n if n > 0 else 0.5

    def as_dict(self) -> dict:
        return {
            "total_reflections":  len(self.impacts),
            "structural_count":   self.structural_count,
            "stylistic_count":    self.stylistic_count,
            "null_count":         self.null_count,
            "avg_impact_score":   round(self.avg_impact_score, 3),
            "impact_rate":        round(self.impact_rate(), 3),
            "most_impactful_turn":  self.most_impactful_turn,
            "least_impactful_turn": self.least_impactful_turn,
            "impacts": [
                {
                    "turn":                  imp.reflection_turn,
                    "impact_type":           imp.impact_type,
                    "impact_score":          round(imp.impact_score, 3),
                    "contradiction_delta":   imp.contradiction_delta,
                    "failure_signal_delta":  imp.failure_signal_delta,
                    "action_specificity_delta": imp.action_specificity_delta,
                    "coherence_delta":       round(imp.coherence_delta, 3),
                    "vocabulary_change":     round(imp.vocabulary_change, 3),
                    "recovery_followed":     imp.recovery_followed,
                    "assumption_chain_broken": imp.assumption_chain_broken,
                    "narrative":             imp.narrative(),
                }
                for imp in self.impacts
            ],
        }

    def narrative(self) -> str:
        if not self.impacts:
            return "No reflection steps detected."
        n = len(self.impacts)
        return (
            f"{n} reflection(s): {self.structural_count} structural, "
            f"{self.stylistic_count} stylistic, {self.null_count} null. "
            f"Avg impact score: {self.avg_impact_score:.2f}. "
            f"Impact rate (structural): {self.impact_rate():.0%}."
        )


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_reflection_impact(traj: "AgentTrajectory") -> ReflectionImpactReport:
    """
    For each REFLECTION step in *traj*, compute a ReflectionImpact by comparing
    the _WINDOW steps before and after the reflection across 5 signal dimensions.

    Classification:
      structural — ≥2 of the 5 signals show positive delta
      stylistic  — vocabulary_change ≥ 0.25 but <2 positive signal deltas
      null       — <2 positive deltas AND vocabulary_change < 0.25
    """
    from harpo.trajectory.schema import StepType

    all_steps = traj.steps
    think_steps = [
        s for s in all_steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE, StepType.REFLECTION, StepType.RECOVERY)
    ]
    reflections = [s for s in all_steps if s.step_type == StepType.REFLECTION]
    recovery_steps = [s for s in all_steps if s.step_type == StepType.RECOVERY]

    if not reflections:
        return ReflectionImpactReport()

    # Build objective tokens from trajectory start (for coherence measurement)
    seed_steps = [s for s in think_steps if s.step_type in (StepType.THINK, StepType.RESPONSE)][:3]
    objective_tokens = set()
    for s in seed_steps:
        objective_tokens |= _sig_tokens(s.output_text)

    # Build active assumption token pool (simplified)
    assumption_token_pool: Set[str] = set()
    try:
        from harpo.semantic.assumptions import analyze_assumption_propagation
        apr = analyze_assumption_propagation(traj)
        for chain in apr.chains:
            if chain.propagation_radius() >= 1:
                assumption_token_pool |= chain.key_tokens
    except Exception:
        pass

    impacts: List[ReflectionImpact] = []

    for ref in reflections:
        # Collect pre/post windows
        pre_steps = [
            s for s in think_steps
            if s.timestamp < ref.timestamp and s.step_type in (StepType.THINK, StepType.RESPONSE)
        ][-_WINDOW:]

        post_steps = [
            s for s in think_steps
            if s.timestamp > ref.timestamp and s.step_type in (StepType.THINK, StepType.RESPONSE)
        ][:_WINDOW]

        if not pre_steps or not post_steps:
            continue

        pre_text  = _combine_text(pre_steps)
        post_text = _combine_text(post_steps)

        # ── Signal 1: contradiction density ──────────────────────────────────
        pre_contr  = _count_pattern(_CONTRADICTION_SIGNALS, pre_text)
        post_contr = _count_pattern(_CONTRADICTION_SIGNALS, post_text)
        contr_delta = pre_contr - post_contr   # positive = fewer contradictions after

        # ── Signal 2: failure signal density ─────────────────────────────────
        pre_fail  = _count_pattern(_FAILURE_SIGNALS, pre_text)
        post_fail = _count_pattern(_FAILURE_SIGNALS, post_text)
        fail_delta = pre_fail - post_fail      # positive = fewer failures after

        # ── Signal 3: action/plan specificity ────────────────────────────────
        pre_action  = _count_pattern(_ACTION_TOKENS, pre_text)
        post_action = _count_pattern(_ACTION_TOKENS, post_text)
        action_delta = post_action - pre_action  # positive = more specific plans after

        # ── Signal 4: coherence with objective ───────────────────────────────
        pre_toks  = _sig_tokens(pre_text)
        post_toks = _sig_tokens(post_text)
        pre_coh   = _jaccard(pre_toks, objective_tokens)
        post_coh  = _jaccard(post_toks, objective_tokens)
        coh_delta = post_coh - pre_coh         # positive = better aligned after

        # ── Signal 5: vocabulary change ───────────────────────────────────────
        vocab_change = 1.0 - _jaccard(pre_toks, post_toks)

        # ── Recovery linkage ─────────────────────────────────────────────────
        nearby_recoveries = [
            r for r in recovery_steps
            if 0 < (r.turn_number - ref.turn_number) <= _WINDOW
        ]
        recovery_followed = len(nearby_recoveries) > 0
        recovery_succeeded = False
        if recovery_followed:
            rec_text = _combine_text(nearby_recoveries)
            rec_ok = _count_pattern(_RECOVERY_SIGNALS, rec_text)
            rec_fail = _count_pattern(_FAILURE_SIGNALS, rec_text)
            recovery_succeeded = rec_ok > rec_fail

        # ── Assumption chain breakage ─────────────────────────────────────────
        assumption_chain_broken = False
        if assumption_token_pool:
            pre_assump_ov  = len(assumption_token_pool & _sig_tokens(pre_text))
            post_assump_ov = len(assumption_token_pool & _sig_tokens(post_text))
            assumption_chain_broken = (
                pre_assump_ov >= 3
                and post_assump_ov < pre_assump_ov * 0.5
            )

        # ── Classification ────────────────────────────────────────────────────
        positive_signals = sum([
            contr_delta > 0,
            fail_delta > 0,
            action_delta > 0,
            coh_delta > 0.02,
            assumption_chain_broken,
            recovery_followed and recovery_succeeded,
        ])

        if positive_signals >= 2:
            impact_type = "structural"
        elif vocab_change >= 0.25:
            impact_type = "stylistic"
        else:
            impact_type = "null"

        # Impact score: weighted combination
        raw_score = (
            min(max(contr_delta / 3.0, 0.0), 1.0) * 0.25
            + min(max(fail_delta / 3.0, 0.0), 1.0) * 0.20
            + min(max(action_delta / 3.0, 0.0), 1.0) * 0.15
            + min(max(coh_delta * 5.0, 0.0), 1.0) * 0.20
            + (0.10 if assumption_chain_broken else 0.0)
            + (0.10 if recovery_succeeded else 0.0)
        )
        impact_score = round(min(raw_score, 1.0), 3)

        impacts.append(ReflectionImpact(
            reflection_step_id        = ref.step_id,
            reflection_turn           = ref.turn_number,
            reflection_text           = ref.output_text[:200],
            pre_window_text           = pre_text[:300],
            post_window_text          = post_text[:300],
            contradiction_delta       = contr_delta,
            failure_signal_delta      = fail_delta,
            action_specificity_delta  = action_delta,
            coherence_delta           = round(coh_delta, 4),
            vocabulary_change         = round(vocab_change, 4),
            recovery_followed         = recovery_followed,
            recovery_succeeded        = recovery_succeeded,
            assumption_chain_broken   = assumption_chain_broken,
            impact_type               = impact_type,
            impact_score              = impact_score,
        ))

    structural = sum(1 for i in impacts if i.impact_type == "structural")
    stylistic  = sum(1 for i in impacts if i.impact_type == "stylistic")
    null       = sum(1 for i in impacts if i.impact_type == "null")
    avg_score  = sum(i.impact_score for i in impacts) / len(impacts) if impacts else 0.0

    most_imp  = max(impacts, key=lambda i: i.impact_score, default=None)
    least_imp = min(impacts, key=lambda i: i.impact_score, default=None)

    return ReflectionImpactReport(
        impacts              = impacts,
        structural_count     = structural,
        stylistic_count      = stylistic,
        null_count           = null,
        avg_impact_score     = round(avg_score, 3),
        most_impactful_turn  = most_imp.reflection_turn if most_imp else None,
        least_impactful_turn = least_imp.reflection_turn if least_imp else None,
    )
