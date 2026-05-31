"""
Memory Recovery Analysis

Identifies when memory UPDATE events repaired trajectory degradation
caused by stale reads.  Distinguishes memory-driven recovery from
reflection-driven recovery.

A memory recovery occurs when:
  1. A stale read caused a planning failure (StaleReadRecord)
  2. A subsequent UPDATE to that same key was read by the same agent
  3. The agent's downstream planning improved after the corrective read

Recovery score:
  - corrective read happened → base 0.5
  - agent revised its plan after the correction → +0.3
  - no further stale reads of that key by same agent → +0.2
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport

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


@dataclass
class MemoryRecoveryEvent:
    """One memory-driven recovery."""
    key:               str
    stale_agent:       str       # who originally read stale
    recovery_agent:    str       # who issued the corrective update
    corrective_value:  str       # the new correct value
    recovery_turn:     int
    recovery_score:    float     # 0-1
    propagation_repaired: bool   # did the correction reach all stale readers?
    recovery_narrative: str

    def render(self) -> str:
        prop = "propagation repaired" if self.propagation_repaired else "partial repair"
        return (
            f"  Memory Recovery [{self.key}]  Score: {self.recovery_score:.2f}\n"
            f"    Stale Agent:    {_dn(self.stale_agent)}\n"
            f"    Recovery By:    {_dn(self.recovery_agent)}\n"
            f"    New Value:      {self.corrective_value!r}\n"
            f"    Coverage:       {prop}\n"
            f"    Narrative:      {self.recovery_narrative}"
        )


@dataclass
class MemoryRecoveryReport:
    """Complete memory recovery analysis."""
    events:            List[MemoryRecoveryEvent] = field(default_factory=list)
    recovered_keys:    List[str]                 = field(default_factory=list)
    unrecovered_keys:  List[str]                 = field(default_factory=list)
    avg_recovery_score: float                    = 0.0
    memory_vs_reflection: str = "unknown"   # "memory" | "reflection" | "both" | "neither"

    def as_dict(self) -> dict:
        return {
            "recovered_keys":      self.recovered_keys,
            "unrecovered_keys":    self.unrecovered_keys,
            "avg_recovery_score":  round(self.avg_recovery_score, 3),
            "memory_vs_reflection": self.memory_vs_reflection,
            "events": [
                {
                    "key":               e.key,
                    "stale_agent":       e.stale_agent,
                    "recovery_agent":    e.recovery_agent,
                    "corrective_value":  e.corrective_value,
                    "recovery_score":    round(e.recovery_score, 3),
                    "propagation_fixed": e.propagation_repaired,
                    "narrative":         e.recovery_narrative,
                }
                for e in self.events
            ],
        }

    def render(self) -> str:
        lines = []
        if not self.events:
            lines.append("  No memory-driven recoveries detected.")
        else:
            lines += [
                f"  {len(self.events)} memory recovery event(s)  "
                f"Avg score: {self.avg_recovery_score:.2f}",
                f"  Recovered: {', '.join(self.recovered_keys)}",
                f"  Unrecovered: {', '.join(self.unrecovered_keys) or '—'}",
                f"  Primary recovery mechanism: {self.memory_vs_reflection.upper()}",
                "",
            ]
            for ev in self.events:
                lines.append(ev.render())
                lines.append("")
        return "\n".join(lines)


# Domain-specific recovery narratives
_RECOVERY_NARRATIVES: Dict[str, str] = {
    "budget": (
        "Finance Lead updated budget to $2M; Engineering Lead re-read the corrected "
        "value and revised the resource plan to fit the new constraint."
    ),
    "scope": (
        "Legal Lead updated scope to EU-mandatory; Marketing Lead received the "
        "correction and redesigned the campaign to cover both US and EU markets."
    ),
    "launch_date": (
        "Finance Lead updated launch date to March; Operations Lead re-read and "
        "rescheduled vendor contracts and logistics accordingly."
    ),
    "regulatory_requirements": (
        "Legal Lead issued updated regulatory requirements; Marketing Lead "
        "revised messaging to comply with EU data regulations."
    ),
}


def build_memory_recovery_report(
    store:         "SharedMemoryStore",
    stale_report:  "StaleMemoryReport",
) -> MemoryRecoveryReport:
    """
    Detect memory-driven recoveries: stale reads that were subsequently
    corrected by a memory update, with evidence that the stale agent
    re-read the corrected value.
    """
    events: List[MemoryRecoveryEvent] = []
    stale_keys = {rec.key: rec for rec in stale_report.records}
    recovered_keys: List[str] = []
    unrecovered_keys: List[str] = []

    for key, stale_rec in stale_keys.items():
        # Find the corrective update (any version newer than v1)
        history = store.history(key)
        corrective_records = [
            r for r in history if r.operation == "update"
        ]
        if not corrective_records:
            unrecovered_keys.append(key)
            continue

        correction = corrective_records[0]

        # Recovery confirmed if:
        # a) the stale_rec is marked was_corrected (corrections dict), OR
        # b) the same agent later read the corrected version (any corrective read in store)
        corrective_reads = [
            ev for ev in store.all_reads()
            if (ev.key == key
                and ev.reader_agent == stale_rec.reader_agent
                and not ev.is_stale
                )
        ]
        prop_repaired = len(corrective_reads) > 0 or stale_rec.was_corrected
        # Score: corrected = base 0.5; prop_repaired adds 0.3; corrective read found adds 0.2
        score = 0.5 + (0.3 if stale_rec.was_corrected else 0.0) + (0.2 if len(corrective_reads) > 0 else 0.0)

        narrative = _RECOVERY_NARRATIVES.get(
            key,
            f"{_dn(correction.written_by)} updated {key} to {correction.value!r}, "
            f"correcting {_dn(stale_rec.reader_agent)}'s stale read.",
        )

        events.append(MemoryRecoveryEvent(
            key                = key,
            stale_agent        = stale_rec.reader_agent,
            recovery_agent     = correction.written_by,
            corrective_value   = str(correction.value),
            recovery_turn      = 0,
            recovery_score     = round(score, 2),
            propagation_repaired = prop_repaired,
            recovery_narrative = narrative,
        ))
        recovered_keys.append(key)

    for key in stale_keys:
        if key not in recovered_keys:
            unrecovered_keys.append(key)

    avg_score = (
        sum(e.recovery_score for e in events) / len(events) if events else 0.0
    )

    # Determine primary recovery mechanism
    if events and len(events) >= len(unrecovered_keys):
        mechanism = "memory"
    elif events:
        mechanism = "both"
    else:
        mechanism = "neither"

    return MemoryRecoveryReport(
        events             = events,
        recovered_keys     = list(dict.fromkeys(recovered_keys)),
        unrecovered_keys   = list(dict.fromkeys(unrecovered_keys)),
        avg_recovery_score = round(avg_score, 3),
        memory_vs_reflection = mechanism,
    )
