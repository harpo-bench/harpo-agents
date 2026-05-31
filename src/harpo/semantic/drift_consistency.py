"""
Drift Consistency Layer — Single Source of Truth

Resolves the contradiction where the system can simultaneously report:
  "10 objective drift events"   (from drift_analysis.py v1)
  "No objective drift detected" (from objective_drift_v2.py)

Root cause: v1 fires on ANY step with low Jaccard overlap against the
incident brief, even when it's just a specialist agent using domain
vocabulary.  v2 requires sustained low overlap AND pressure token presence.

This module provides a single authoritative DriftSummary that:
  1. Always uses v2 as the ground truth for objective drift
  2. Preserves v1's attention collapse events (they're separately calibrated in v2)
  3. Documents the false-positive count clearly so analysts understand the delta
  4. Ensures the report never contradicts itself

Used by the forensics report to get a single consistent drift picture.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional


@dataclass
class DriftSummary:
    """Single source of truth for drift across the trajectory."""
    objective_drift_count:    int            # from v2 (calibrated)
    attention_collapse_count: int            # from v2 (calibrated)
    topic_evolution_count:    int            # benign specialization (v2)
    drift_agents:             List[str]      # agents with real objective drift
    recovery_rate:            float          # fraction of drift events that recovered
    overall_drift_score:      float          # 0-1
    false_positive_suppressed: int           # v1 events that v2 filtered
    authoritative_source:     str            # "v2" always

    # Narrative
    summary:                  str

    def has_harmful_drift(self) -> bool:
        return self.objective_drift_count > 0

    def render(self) -> str:
        if not self.has_harmful_drift() and self.attention_collapse_count == 0:
            benign_note = (f" ({self.topic_evolution_count} benign topic evolutions "
                           f"suppressed as healthy role specialization.)"
                           if self.topic_evolution_count else "")
            return f"  No objective drift detected.{benign_note}"

        lines = [f"  Drift Score: {self.overall_drift_score:.2f}"]
        if self.objective_drift_count:
            agents = ", ".join(self.drift_agents[:3])
            lines.append(f"  Objective Drift: {self.objective_drift_count} event(s) — {agents}")
        if self.attention_collapse_count:
            lines.append(f"  Attention Collapse: {self.attention_collapse_count} entity collapse(s)")
        if self.false_positive_suppressed:
            lines.append(f"  [{self.false_positive_suppressed} v1 false positives suppressed: "
                         f"healthy role specialization, not drift]")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "objective_drift_count":    self.objective_drift_count,
            "attention_collapse_count": self.attention_collapse_count,
            "topic_evolution_count":    self.topic_evolution_count,
            "drift_agents":             self.drift_agents,
            "recovery_rate":            round(self.recovery_rate, 3),
            "overall_drift_score":      round(self.overall_drift_score, 3),
            "false_positive_suppressed": self.false_positive_suppressed,
            "summary":                  self.summary,
        }


def get_authoritative_drift(analysis: Any) -> DriftSummary:
    """
    Extract the authoritative drift picture from a SemanticAnalysis object.

    Always uses v2 if available; falls back to v1 with a correction note.
    """
    dr2 = getattr(analysis, "drift_v2", None)

    if dr2 is not None:
        # v2 is available — use it as ground truth
        obj_count  = getattr(dr2, "objective_drift_count", 0)
        ac_count   = getattr(dr2, "attention_collapse_count", 0)
        ev_count   = getattr(dr2, "topic_evolution_count", 0)
        fp         = getattr(dr2, "false_positive_filter", 0)
        score      = getattr(dr2, "overall_drift_score", 0.0)
        rec_rate   = getattr(dr2, "recovery_rate", 0.0)
        agents     = getattr(dr2, "drift_agents", [])
        summary    = dr2.narrative() if hasattr(dr2, "narrative") else ""

        return DriftSummary(
            objective_drift_count    = obj_count,
            attention_collapse_count = ac_count,
            topic_evolution_count    = ev_count,
            drift_agents             = agents,
            recovery_rate            = rec_rate,
            overall_drift_score      = score,
            false_positive_suppressed = fp,
            authoritative_source     = "v2",
            summary                  = summary,
        )

    # Fall back to v1 — report its numbers but flag that they may include
    # false positives from healthy role specialization
    dr1 = getattr(analysis, "drift", None)
    if dr1 is None:
        return DriftSummary(
            objective_drift_count=0, attention_collapse_count=0,
            topic_evolution_count=0, drift_agents=[], recovery_rate=0.0,
            overall_drift_score=0.0, false_positive_suppressed=0,
            authoritative_source="none",
            summary="No drift data available.",
        )

    obj_count = sum(1 for e in getattr(dr1, "events", [])
                    if getattr(e, "drift_type", "").endswith("drift"))
    ac_count  = len(getattr(dr1, "attention_collapse_turns", []))
    score     = getattr(dr1, "overall_drift_score", 0.0)
    rec_rate  = getattr(dr1, "recovery_rate", 0.0)
    summary   = (
        f"[v1 drift — may include false positives from role specialization] "
        f"{dr1.narrative() if hasattr(dr1, 'narrative') else ''}"
    )
    return DriftSummary(
        objective_drift_count    = obj_count,
        attention_collapse_count = ac_count,
        topic_evolution_count    = 0,
        drift_agents             = [],
        recovery_rate            = rec_rate,
        overall_drift_score      = score,
        false_positive_suppressed = 0,
        authoritative_source     = "v1_fallback",
        summary                  = summary,
    )
