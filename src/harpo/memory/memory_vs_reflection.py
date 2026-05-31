"""
Memory vs Reflection Recovery Attribution

For every recovery event: what caused the recovery?

A recovery event (agent revised its plan) can be caused by:

  MEMORY_UPDATE     — the agent re-read the shared memory store and got the
                       corrected value. Signal: corrective read event exists
                       for this agent+key after the stale read.
                       Confidence boost: high if re-read is explicit; lower
                       if inferred from vocabulary shift.

  REFLECTION        — the agent reflected on its own reasoning and self-corrected.
                       Signal: a REFLECTION step exists after the stale read
                       AND the reflection content shows the corrected value.

  EXTERNAL_INFO     — a subsequent agent's report injected corrected values
                       (context injection pattern in HARPO). This covers the
                       common case where later agents correct earlier ones by
                       mentioning the right values in their feedback.

  HUMAN             — judge feedback explicitly told the agent the correct value.
                       Signal: the corrective judge phase text contains the
                       corrected key value.

  MIXED             — multiple signals are present. Attribution is fractional.

This module produces per-recovery attribution with confidence scores and
fractional credit assignment when signals overlap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport
    from harpo.memory.correction_vs_recovery import CorrectionRecoveryReport, RecoveryEvent

_AGENT_DISPLAY: Dict[str, str] = {
    "product-manager":  "Product Manager",
    "engineering-lead": "Engineering Lead",
    "finance-lead":     "Finance Lead",
    "legal-lead":       "Legal Lead",
    "marketing-lead":   "Marketing Lead",
    "operations-lead":  "Operations Lead",
}

def _dn(a: str) -> str:
    return _AGENT_DISPLAY.get(a or "", (a or "").replace("-", " ").title())


# ── Recovery attribution for a single recovery event ─────────────────────────

@dataclass
class RecoveryAttribution:
    """
    Attribution of a single recovery event to one or more causes.

    The four contribution floats sum to 1.0.
    primary_cause is the dominant cause (highest contribution).
    """
    key:                    str
    recovering_agent:       str

    memory_contribution:    float   # 0-1
    reflection_contribution: float  # 0-1
    external_contribution:  float   # 0-1
    human_contribution:     float   # 0-1

    primary_cause:          str     # "memory_update" | "reflection" | "external" | "human" | "mixed"
    confidence:             float   # overall confidence in this attribution
    narrative:              str     # "Recovery was 70% memory-driven, 30% reflection"

    # Evidence that drove the attribution
    found_corrective_read:  bool = False
    found_reflection:       bool = False
    found_external_info:    bool = False
    found_human_feedback:   bool = False

    def render(self) -> str:
        contribs = []
        if self.memory_contribution > 0.05:
            contribs.append(f"memory {self.memory_contribution * 100:.0f}%")
        if self.reflection_contribution > 0.05:
            contribs.append(f"reflection {self.reflection_contribution * 100:.0f}%")
        if self.external_contribution > 0.05:
            contribs.append(f"external {self.external_contribution * 100:.0f}%")
        if self.human_contribution > 0.05:
            contribs.append(f"human {self.human_contribution * 100:.0f}%")
        contrib_str = "  +  ".join(contribs)
        return (
            f"  Recovery [{self.key}]  {_dn(self.recovering_agent)}\n"
            f"    Cause:       {contrib_str}\n"
            f"    Primary:     {self.primary_cause}\n"
            f"    Confidence:  {self.confidence:.2f}\n"
            f"    Narrative:   {self.narrative}"
        )


# ── Aggregate report ──────────────────────────────────────────────────────────

@dataclass
class MemoryVsReflectionReport:
    """
    All recovery events with their causal attribution.
    Summarises: was recovery primarily memory-driven or reflection-driven?
    """
    attributions:              List[RecoveryAttribution] = field(default_factory=list)
    avg_memory_contribution:   float = 0.0
    avg_reflection_contribution: float = 0.0
    dominant_recovery_mode:    str   = "unknown"   # "memory" | "reflection" | "external" | "mixed"
    unattributed_recoveries:   int   = 0
    summary:                   str   = ""

    def as_dict(self) -> dict:
        return {
            "avg_memory_contribution":   round(self.avg_memory_contribution, 3),
            "avg_reflection_contribution": round(self.avg_reflection_contribution, 3),
            "dominant_recovery_mode":    self.dominant_recovery_mode,
            "unattributed_recoveries":   self.unattributed_recoveries,
            "summary":                   self.summary,
            "attributions": [
                {
                    "key":                    a.key,
                    "recovering_agent":       a.recovering_agent,
                    "memory_pct":             round(a.memory_contribution * 100, 1),
                    "reflection_pct":         round(a.reflection_contribution * 100, 1),
                    "external_pct":           round(a.external_contribution * 100, 1),
                    "human_pct":              round(a.human_contribution * 100, 1),
                    "primary_cause":          a.primary_cause,
                    "confidence":             round(a.confidence, 3),
                    "narrative":              a.narrative,
                }
                for a in self.attributions
            ],
        }

    def render(self) -> str:
        lines = []
        if not self.attributions:
            lines.append("  No recoveries to attribute.")
            return "\n".join(lines)
        lines += [
            f"  {len(self.attributions)} recovery event(s) attributed.",
            f"  Dominant recovery mode:    {self.dominant_recovery_mode.upper()}",
            f"  Avg memory contribution:   {self.avg_memory_contribution * 100:.0f}%",
            f"  Avg reflection contribution: {self.avg_reflection_contribution * 100:.0f}%",
            f"  Summary: {self.summary}",
            "",
        ]
        for a in self.attributions:
            lines.append(a.render())
            lines.append("")
        return "\n".join(lines)


# ── Evidence signals ──────────────────────────────────────────────────────────

def _check_corrective_read(
    store: "SharedMemoryStore",
    key: str,
    agent: str,
) -> bool:
    """True if this agent has a non-stale read of key after the initial stale read."""
    reads = [
        r for r in store.all_reads()
        if r.key == key and r.reader_agent == agent and not r.is_stale
    ]
    return len(reads) > 0


def _check_reflection_signal(
    traj: Optional[object],
    key: str,
    corrected_value: str,
) -> bool:
    """
    True if there is a REFLECTION step whose text contains tokens from the
    corrected value.
    """
    if traj is None:
        return False
    try:
        from harpo.trajectory.schema import StepType
        import re
        corr_tokens = set(re.findall(r'[a-z0-9]+', corrected_value.lower()))
        corr_tokens -= {"to", "the", "a", "is", "of", "by", "in", "at", "on"}
        for step in traj.steps:
            if step.step_type == StepType.REFLECTION and step.output_text:
                step_tokens = set(re.findall(r'[a-z0-9]+', step.output_text.lower()))
                if len(corr_tokens & step_tokens) >= max(1, len(corr_tokens) // 3):
                    return True
    except Exception:
        pass
    return False


def _check_external_info(
    stale_agents: List[str],
    recovering_agent: str,
    correction_agent: str,
) -> bool:
    """
    True if recovering_agent receives output from another agent who has the
    corrected value — i.e., the correction propagated through social context
    rather than direct memory re-read.
    """
    # If the correction came from a different agent than the stale reader,
    # and that correction agent's output is consumed by the recovering agent,
    # this is external_info.
    return (
        recovering_agent != correction_agent
        and correction_agent not in stale_agents
    )


# ── Attribution logic ─────────────────────────────────────────────────────────

def _attribute_one(
    key:              str,
    recovering_agent: str,
    corrected_value:  str,
    correction_agent: str,
    stale_agents:     List[str],
    store:            "SharedMemoryStore",
    traj:             Optional[object],
    recovery_type:    str,    # from RecoveryEvent.recovery_type
) -> RecoveryAttribution:
    """Attribute a single recovery event."""

    found_read     = _check_corrective_read(store, key, recovering_agent)
    found_refl     = _check_reflection_signal(traj, key, corrected_value)
    found_external = _check_external_info(stale_agents, recovering_agent, correction_agent)
    found_human    = (recovery_type == "human")

    # Build raw signal weights
    mem_raw  = 0.8 if found_read  else (0.3 if recovery_type == "memory_update" else 0.0)
    refl_raw = 0.6 if found_refl  else 0.0
    ext_raw  = 0.5 if found_external else 0.0
    hum_raw  = 0.7 if found_human else 0.0

    total = mem_raw + refl_raw + ext_raw + hum_raw
    if total < 0.01:
        # No signal at all — default to external (judge context is always present)
        ext_raw = 0.5
        hum_raw = 0.5
        total   = 1.0

    mem_c  = mem_raw  / total
    refl_c = refl_raw / total
    ext_c  = ext_raw  / total
    hum_c  = hum_raw  / total

    # Primary cause
    contributions = {
        "memory_update": mem_c,
        "reflection":    refl_c,
        "external":      ext_c,
        "human":         hum_c,
    }
    primary = max(contributions, key=contributions.get)
    # "mixed" if top two are within 15% of each other
    sorted_vals = sorted(contributions.values(), reverse=True)
    if len(sorted_vals) >= 2 and sorted_vals[0] - sorted_vals[1] < 0.15:
        primary = "mixed"

    # Confidence: higher when we have explicit signals, lower when inferred
    n_signals = sum([found_read, found_refl, found_external, found_human])
    confidence = 0.60 + n_signals * 0.10

    # Narrative
    parts = []
    if mem_c > 0.05:
        parts.append(f"{mem_c * 100:.0f}% memory correction")
    if refl_c > 0.05:
        parts.append(f"{refl_c * 100:.0f}% reflection")
    if ext_c > 0.05:
        parts.append(f"{ext_c * 100:.0f}% external information")
    if hum_c > 0.05:
        parts.append(f"{hum_c * 100:.0f}% human feedback")
    narrative = (
        f"{_dn(recovering_agent)} recovered via: {' + '.join(parts)}. "
        f"Primary driver: {primary.replace('_', ' ')}."
    )

    return RecoveryAttribution(
        key                     = key,
        recovering_agent        = recovering_agent,
        memory_contribution     = round(mem_c,  3),
        reflection_contribution = round(refl_c, 3),
        external_contribution   = round(ext_c,  3),
        human_contribution      = round(hum_c,  3),
        primary_cause           = primary,
        confidence              = round(confidence, 3),
        narrative               = narrative,
        found_corrective_read   = found_read,
        found_reflection        = found_refl,
        found_external_info     = found_external,
        found_human_feedback    = found_human,
    )


# ── Public builder ────────────────────────────────────────────────────────────

def build_memory_vs_reflection_report(
    store:           "SharedMemoryStore",
    cr_report:       "CorrectionRecoveryReport",
    stale_report:    "StaleMemoryReport",
    traj:            Optional[object] = None,
) -> "MemoryVsReflectionReport":
    """
    Build a MemoryVsReflectionReport from correction/recovery events and
    the memory store.

    Each RecoveryEvent in cr_report gets one RecoveryAttribution.
    """
    attributions: List[RecoveryAttribution] = []

    # Build stale agents map per key
    stale_agents_by_key: Dict[str, List[str]] = {}
    for rec in stale_report.records:
        stale_agents_by_key.setdefault(rec.key, []).append(rec.reader_agent)

    # Build correction agent map per key
    correction_agent_by_key: Dict[str, str] = {}
    for c in cr_report.corrections:
        correction_agent_by_key[c.key] = c.correction_agent

    for rec_ev in cr_report.recoveries:
        correction_agent = correction_agent_by_key.get(rec_ev.key, "unknown")
        stale_agents     = stale_agents_by_key.get(rec_ev.key, [])

        attr = _attribute_one(
            key              = rec_ev.key,
            recovering_agent = rec_ev.recovering_agent,
            corrected_value  = rec_ev.corrected_value,
            correction_agent = correction_agent,
            stale_agents     = stale_agents,
            store            = store,
            traj             = traj,
            recovery_type    = rec_ev.recovery_type,
        )
        attributions.append(attr)

    if not attributions:
        return MemoryVsReflectionReport(
            unattributed_recoveries = len(cr_report.corrections),
            summary = "No recovery events detected. Corrections occurred without behavioral change.",
            dominant_recovery_mode = "none",
        )

    avg_mem  = sum(a.memory_contribution     for a in attributions) / len(attributions)
    avg_refl = sum(a.reflection_contribution for a in attributions) / len(attributions)
    avg_ext  = sum(a.external_contribution   for a in attributions) / len(attributions)
    avg_hum  = sum(a.human_contribution      for a in attributions) / len(attributions)

    # Dominant mode
    mode_scores = {
        "memory":    avg_mem,
        "reflection": avg_refl,
        "external":  avg_ext,
        "human":     avg_hum,
    }
    dominant = max(mode_scores, key=mode_scores.get)
    sorted_modes = sorted(mode_scores.values(), reverse=True)
    if len(sorted_modes) >= 2 and sorted_modes[0] - sorted_modes[1] < 0.10:
        dominant = "mixed"

    summary = (
        f"{len(attributions)} recovery event(s). "
        f"Primary recovery driver: {dominant.replace('_', ' ')} "
        f"(avg memory={avg_mem * 100:.0f}%, reflection={avg_refl * 100:.0f}%, "
        f"external={avg_ext * 100:.0f}%, human={avg_hum * 100:.0f}%)."
    )

    return MemoryVsReflectionReport(
        attributions               = attributions,
        avg_memory_contribution    = round(avg_mem,  3),
        avg_reflection_contribution = round(avg_refl, 3),
        dominant_recovery_mode     = dominant,
        unattributed_recoveries    = 0,
        summary                    = summary,
    )
