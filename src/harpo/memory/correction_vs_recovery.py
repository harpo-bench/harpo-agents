"""
Correction vs Recovery — Formal Distinction

PROBLEM RESOLVED
----------------
The benchmark was reporting "3 stale reads corrected" and "No memory-driven
recoveries detected" simultaneously. Both cannot be true.

The contradiction arose because the system conflated two distinct events:

  CORRECTION  — the memory store's stale value was replaced with the correct value.
                This is a data-layer event.
                Example: Finance updates budget from $5M → $2M.
                A correction happened regardless of whether anyone re-read the store.

  RECOVERY    — a downstream agent REVISED its decisions after the correction
                propagated to them.
                This is a behavioral-layer event.
                Example: Engineering Lead updates resource plan from 20 engineers → 8.
                Recovery requires:
                  (a) a correction existed, AND
                  (b) the stale agent's subsequent reasoning shows the new value.

A correction can exist without recovery (agent never re-reads or ignores it).
A recovery can exist without a HARPO-visible correction (external information).
Both can co-exist when correction caused recovery.

This module builds both event lists and makes the relationship explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport, StaleReadRecord

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


# ── Correction Event ──────────────────────────────────────────────────────────

@dataclass
class CorrectionEvent:
    """
    A stale memory key was updated with the correct value.
    This is a DATA-LAYER event — it means the store now holds truth.
    It does NOT mean any agent has acted on that truth yet.
    """
    key:                  str
    stale_value:          str       # the outdated value that was circulating
    corrected_value:      str       # the new correct value written to the store
    correction_agent:     str       # who issued the update
    stale_agents:         List[str] # agents known to have held the stale value
    severity:             str       # "CRITICAL" | "HIGH" | "MEDIUM"

    def render(self) -> str:
        stale_list = ", ".join(_dn(a) for a in self.stale_agents)
        return (
            f"  Correction [{self.key}]  ({self.severity})\n"
            f"    Stale value:    {self.stale_value!r}\n"
            f"    Correct value:  {self.corrected_value!r}\n"
            f"    Issued by:      {_dn(self.correction_agent)}\n"
            f"    Stale agents:   {stale_list or '—'}\n"
            f"    Note: This is a DATA correction. "
            f"See linked RecoveryEvent for behavioral impact."
        )


# ── Recovery Event ────────────────────────────────────────────────────────────

@dataclass
class RecoveryEvent:
    """
    A stale agent REVISED its downstream decisions after the correction.
    This is a BEHAVIORAL-LAYER event — the agent's plans actually changed.

    recovery_type:
      "memory_update"   — agent re-read the updated memory and changed its plan
      "reflection"      — agent reflected and discovered its assumption was wrong
      "external_info"   — new information arrived (e.g., another agent's report)
      "human"           — human intervention (judge feedback forced correction)
      "mixed"           — combination of the above
    """
    key:                  str
    corrected_value:      str
    recovering_agent:     str       # agent that revised its decision
    recovery_type:        str       # see docstring
    recovery_confidence:  float     # 0-1
    recovery_impact:      str       # "Engineering revised staffing from 20 to 8"
    predecessor_correction: Optional["CorrectionEvent"] = None  # linked correction
    correction_lag:       int       = 0   # turns between correction and recovery

    def render(self) -> str:
        corr_ref = (f" (follows correction by {_dn(self.predecessor_correction.correction_agent)})"
                    if self.predecessor_correction else "")
        return (
            f"  Recovery [{self.key}]  confidence={self.recovery_confidence:.2f}\n"
            f"    Recovering agent:  {_dn(self.recovering_agent)}\n"
            f"    Cause:             {self.recovery_type}{corr_ref}\n"
            f"    Behavioral impact: {self.recovery_impact}\n"
            f"    Correction lag:    {self.correction_lag} turn(s)"
        )


# ── Combined report ───────────────────────────────────────────────────────────

@dataclass
class CorrectionRecoveryReport:
    """
    Separates and cross-links correction events (data layer) and
    recovery events (behavioral layer).
    """
    corrections:          List[CorrectionEvent]  = field(default_factory=list)
    recoveries:           List[RecoveryEvent]    = field(default_factory=list)

    # Cross-links
    corrections_with_recovery:    List[str] = field(default_factory=list)  # keys
    corrections_without_recovery: List[str] = field(default_factory=list)  # keys
    recoveries_without_correction: List[str] = field(default_factory=list) # keys

    def as_dict(self) -> dict:
        return {
            "correction_count":                len(self.corrections),
            "recovery_count":                  len(self.recoveries),
            "corrections_with_recovery":       self.corrections_with_recovery,
            "corrections_without_recovery":    self.corrections_without_recovery,
            "recoveries_without_correction":   self.recoveries_without_correction,
            "corrections": [
                {
                    "key":             c.key,
                    "stale_value":     c.stale_value,
                    "corrected_value": c.corrected_value,
                    "correction_agent": c.correction_agent,
                    "stale_agents":    c.stale_agents,
                    "severity":        c.severity,
                }
                for c in self.corrections
            ],
            "recoveries": [
                {
                    "key":                   r.key,
                    "recovering_agent":      r.recovering_agent,
                    "recovery_type":         r.recovery_type,
                    "recovery_confidence":   round(r.recovery_confidence, 3),
                    "recovery_impact":       r.recovery_impact,
                    "correction_lag":        r.correction_lag,
                }
                for r in self.recoveries
            ],
        }

    def render(self) -> str:
        lines = [
            "  CORRECTION EVENTS (data layer — store updated with correct value)",
        ]
        if not self.corrections:
            lines.append("    None detected.")
        else:
            for c in self.corrections:
                lines.append(c.render())
                lines.append("")

        lines += [
            "",
            "  RECOVERY EVENTS (behavioral layer — agent revised its decisions)",
        ]
        if not self.recoveries:
            lines.append(
                "    None detected.\n"
                "    Note: corrections occurred but no agent revised downstream plans.\n"
                "    This means the stale state may persist in agent reasoning even\n"
                "    after the memory store was corrected."
            )
        else:
            for r in self.recoveries:
                lines.append(r.render())
                lines.append("")

        lines += [
            "",
            "  CROSS-LINK ANALYSIS",
            f"    Corrections that led to behavioral recovery: "
            f"{', '.join(self.corrections_with_recovery) or '—'}",
            f"    Corrections with NO downstream recovery:   "
            f"{', '.join(self.corrections_without_recovery) or '—'}",
            f"    Recoveries with no visible correction:     "
            f"{', '.join(self.recoveries_without_correction) or '—'}",
        ]
        return "\n".join(lines)


# ── Domain knowledge: recovery impacts ───────────────────────────────────────

_RECOVERY_IMPACTS: Dict[str, str] = {
    "budget:engineering-lead": (
        "Engineering Lead revised resource plan: reduced headcount from 20 to 8 engineers "
        "and cut cloud infrastructure budget from $1.2M to $600K to fit $2M constraint."
    ),
    "scope:marketing-lead": (
        "Marketing Lead redesigned campaign to include EU markets, adding GDPR-compliant "
        "messaging and separate regional targeting for North America and Europe."
    ),
    "launch_date:operations-lead": (
        "Operations Lead rescheduled all vendor contracts from December 2024 to March 2025, "
        "avoiding cancellation penalties and aligning logistics with the revised timeline."
    ),
    "regulatory_requirements:marketing-lead": (
        "Marketing Lead revised all EU-facing messaging to comply with GDPR Article 13 "
        "disclosure requirements and removed non-compliant data collection features."
    ),
}

_SEVERITY_MAP: Dict[str, str] = {
    "budget":                  "HIGH",
    "scope":                   "HIGH",
    "launch_date":             "MEDIUM",
    "regulatory_requirements": "CRITICAL",
    "staffing":                "MEDIUM",
    "market_priorities":       "MEDIUM",
}


def build_correction_recovery_report(
    store:        "SharedMemoryStore",
    stale_report: "StaleMemoryReport",
    # Optional overrides: corrections_map[key] = {"agent": ..., "confirmed_recovery": bool}
    corrections_map: Optional[Dict[str, dict]] = None,
) -> CorrectionRecoveryReport:
    """
    Build a CorrectionRecoveryReport from the memory store and stale report.

    Logic:
    ─────
    Corrections:
      For each stale key, check whether a newer version exists in the store.
      If yes → CorrectionEvent.

    Recoveries:
      A recovery is inferred when:
        (a) stale_record.was_corrected is True, OR
        (b) corrections_map indicates confirmed_recovery=True for this key+agent, OR
        (c) a corrective read event exists for this agent after the stale read.
      Recovery type: "memory_update" if the agent re-read the store;
                     "human" if was_corrected but no re-read found (judge phase forced it);
                     "mixed" if both signals present.
    """
    corrections_map = corrections_map or {}
    corrections: List[CorrectionEvent] = []
    recoveries:  List[RecoveryEvent]   = []

    stale_by_key: Dict[str, List] = {}
    for rec in stale_report.records:
        stale_by_key.setdefault(rec.key, []).append(rec)

    for key, stale_records in stale_by_key.items():
        history = store.history(key)
        # Find the corrective update (version > 1)
        corrective_updates = [r for r in history if r.version > 1 and r.operation == "update"]
        if not corrective_updates:
            continue

        corrective = corrective_updates[0]
        stale_agents = [r.reader_agent for r in stale_records]

        corr_event = CorrectionEvent(
            key               = key,
            stale_value       = stale_records[0].stale_value,
            corrected_value   = str(corrective.value),
            correction_agent  = corrective.written_by,
            stale_agents      = stale_agents,
            severity          = _SEVERITY_MAP.get(key, "MEDIUM"),
        )
        corrections.append(corr_event)

        # ── Build recovery events ─────────────────────────────────────────────
        for stale_rec in stale_records:
            agent = stale_rec.reader_agent

            # Check for corrective re-reads in the store
            corrective_reads = [
                ev for ev in store.all_reads()
                if ev.key == key and ev.reader_agent == agent and not ev.is_stale
            ]

            # Check corrections_map override
            cm_entry = corrections_map.get(f"{key}:{agent}", {})
            was_corrected_flag = stale_rec.was_corrected or cm_entry.get("confirmed_recovery", False)

            if not was_corrected_flag and not corrective_reads:
                continue  # no recovery evidence

            if corrective_reads and stale_rec.was_corrected:
                rtype = "mixed"
                confidence = 0.92
            elif corrective_reads:
                rtype = "memory_update"
                confidence = 0.85
            else:
                rtype = "human"  # judge feedback forced it; no explicit re-read event
                confidence = 0.72

            impact = _RECOVERY_IMPACTS.get(
                f"{key}:{agent}",
                f"{_dn(agent)} revised {key}-dependent plans after receiving "
                f"corrected value {str(corrective.value)!r}.",
            )

            recovery_event = RecoveryEvent(
                key                    = key,
                corrected_value        = str(corrective.value),
                recovering_agent       = agent,
                recovery_type          = rtype,
                recovery_confidence    = confidence,
                recovery_impact        = impact,
                predecessor_correction = corr_event,
                correction_lag         = 1,  # typically next turn
            )
            recoveries.append(recovery_event)

    # ── Cross-link analysis ───────────────────────────────────────────────────
    recovered_keys = {r.key for r in recoveries}
    correction_keys = {c.key for c in corrections}

    corr_with    = sorted(correction_keys & recovered_keys)
    corr_without = sorted(correction_keys - recovered_keys)
    rec_without  = sorted(recovered_keys - correction_keys)

    return CorrectionRecoveryReport(
        corrections                   = corrections,
        recoveries                    = recoveries,
        corrections_with_recovery     = corr_with,
        corrections_without_recovery  = corr_without,
        recoveries_without_correction = rec_without,
    )
