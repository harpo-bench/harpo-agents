"""
Reflection Effectiveness Analysis

Determines whether reflection steps actually changed downstream reasoning.

A reflection is effective when the THINK step immediately after it differs
meaningfully from the THINK step immediately before it (Jaccard distance ≥ 0.25).

Covers two patterns:
- REFLECT steps (JudgeVerdict RETRY → mapped to REFLECTION step type)
- RECOVERY steps (explicit retry after failure)

No external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Set

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep

_ACTION_PATTERNS = [
    r"\bI will\b", r"\bI should\b", r"\bnext\b", r"\bwill try\b",
    r"\bcorrect\b", r"\binstead\b", r"\blet me\b", r"\bI need to\b",
    r"\bI must\b", r"\bmy approach\b", r"\ba better\b", r"\bchange\b",
    r"\bre-think\b", r"\breconsider\b", r"\badjust\b", r"\bmodify\b",
]

# Minimum behavior change to count a reflection as "effective"
_CHANGE_THRESHOLD = 0.25


@dataclass
class ReflectionEffect:
    reflection_step_id: str
    reflection_turn:    int
    reflection_text:    str    # truncated to 200 chars
    pre_think_text:     str    # THINK before reflection
    post_think_text:    str    # THINK after reflection
    token_change:       float  # Jaccard distance pre→post (0=identical, 1=completely different)
    action_oriented:    bool   # reflection text contains actionable language
    effective:          bool   # token_change ≥ CHANGE_THRESHOLD

    @property
    def jaccard_similarity(self) -> float:
        return 1.0 - self.token_change


@dataclass
class ReflectionResult:
    effects:               List[ReflectionEffect] = field(default_factory=list)
    effective_count:       int = 0
    ineffective_count:     int = 0
    action_oriented_count: int = 0
    avg_behavior_change:   float = 0.0

    @property
    def total(self) -> int:
        return len(self.effects)

    def effectiveness_rate(self) -> float:
        """Fraction of reflections that produced meaningful behavior change."""
        if self.total == 0:
            return 0.5  # no reflections → neutral
        return self.effective_count / self.total


def _token_set(text: str) -> Set[str]:
    return set(re.findall(r'\b[a-z]{3,}\b', text.lower()))


def _jaccard_distance(text_a: str, text_b: str) -> float:
    """Jaccard distance: 0 = identical token sets, 1 = completely disjoint."""
    if not text_a and not text_b:
        return 0.0
    a = _token_set(text_a)
    b = _token_set(text_b)
    if not a and not b:
        return 0.0
    union = len(a | b)
    return 1.0 - (len(a & b) / union) if union > 0 else 0.0


def _is_action_oriented(text: str) -> bool:
    for pat in _ACTION_PATTERNS:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


def analyze_reflection_effectiveness(traj: "AgentTrajectory") -> ReflectionResult:
    """
    For each REFLECTION step, compare the THINK output before and after.

    If the post-reflection THINK differs by Jaccard distance ≥ 0.25 from
    the pre-reflection THINK, the reflection is considered effective.
    """
    from harpo.trajectory.schema import StepType

    steps          = traj.steps
    reflections    = [s for s in steps if s.step_type == StepType.REFLECTION]
    think_steps    = [s for s in steps if s.step_type in (StepType.THINK, StepType.RESPONSE)]

    if not reflections:
        return ReflectionResult()

    effects: List[ReflectionEffect] = []

    for ref in reflections:
        pre_thinks  = [s for s in think_steps if s.timestamp < ref.timestamp]
        post_thinks = [s for s in think_steps if s.timestamp > ref.timestamp]

        pre_text  = pre_thinks[-1].output_text  if pre_thinks  else ""
        post_text = post_thinks[0].output_text  if post_thinks else ""

        token_change  = _jaccard_distance(pre_text, post_text)
        action_orient = _is_action_oriented(ref.output_text)
        effective     = token_change >= _CHANGE_THRESHOLD

        effects.append(ReflectionEffect(
            reflection_step_id = ref.step_id,
            reflection_turn    = ref.turn_number,
            reflection_text    = ref.output_text[:200],
            pre_think_text     = pre_text[:200],
            post_think_text    = post_text[:200],
            token_change       = round(token_change, 4),
            action_oriented    = action_orient,
            effective          = effective,
        ))

    effective_count       = sum(1 for e in effects if e.effective)
    ineffective_count     = sum(1 for e in effects if not e.effective)
    action_oriented_count = sum(1 for e in effects if e.action_oriented)
    avg_change            = sum(e.token_change for e in effects) / len(effects) if effects else 0.0

    return ReflectionResult(
        effects=effects,
        effective_count=effective_count,
        ineffective_count=ineffective_count,
        action_oriented_count=action_oriented_count,
        avg_behavior_change=round(avg_change, 4),
    )
