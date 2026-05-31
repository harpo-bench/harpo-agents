"""
Stale Memory Detector

Identifies which agents used outdated memory values and what planning
decisions resulted from those stale reads.

For each stale read, reports:
  - which agent read it
  - what stale value they got vs. what the correct value was
  - how many versions behind they were
  - what planning decision they likely made based on the stale value

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore, MemoryReadEvent

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


# Domain-specific consequences of stale reads
_STALE_CONSEQUENCES: Dict[str, Dict] = {
    "budget": {
        "stale_decision": "Agent planned resources based on outdated budget allocation.",
        "failure_type":   "resource_overestimation",
        "severity":       "HIGH",
    },
    "scope": {
        "stale_decision": "Agent designed campaign/plan for wrong geographic scope.",
        "failure_type":   "scope_mismatch",
        "severity":       "HIGH",
    },
    "launch_date": {
        "stale_decision": "Agent scheduled work against incorrect launch timeline.",
        "failure_type":   "timeline_conflict",
        "severity":       "MEDIUM",
    },
    "regulatory_requirements": {
        "stale_decision": "Agent ignored updated regulatory constraints.",
        "failure_type":   "compliance_gap",
        "severity":       "CRITICAL",
    },
    "staffing": {
        "stale_decision": "Agent made hiring/allocation decisions on outdated headcount.",
        "failure_type":   "resource_mismatch",
        "severity":       "MEDIUM",
    },
    "market_priorities": {
        "stale_decision": "Agent prioritized wrong market segments.",
        "failure_type":   "misaligned_priorities",
        "severity":       "MEDIUM",
    },
}


@dataclass
class StaleReadRecord:
    """One detected stale memory read with its planning consequence."""
    reader_agent:      str
    key:               str
    stale_value:       str    # what the agent read
    current_value:     str    # what they should have read
    version_lag:       int    # how many versions behind
    consequence:       str    # one sentence on what went wrong
    failure_type:      str    # "resource_overestimation" | "scope_mismatch" | ...
    severity:          str    # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    was_corrected:     bool   = False
    correction_agent:  str    = ""

    def render(self) -> str:
        cor = f"Corrected by {_dn(self.correction_agent)}" if self.was_corrected else "NOT corrected"
        return (
            f"  [{self.severity}]  {_dn(self.reader_agent)} read stale {self.key}\n"
            f"     Read:    {self.stale_value!r} (v{self.version_lag + 1} behind)\n"
            f"     Actual:  {self.current_value!r}\n"
            f"     Impact:  {self.consequence}\n"
            f"     Status:  {cor}"
        )


@dataclass
class StaleMemoryReport:
    """All detected stale reads and their consequences."""
    records:          List[StaleReadRecord] = field(default_factory=list)
    affected_agents:  List[str]            = field(default_factory=list)
    total_stale:      int                  = 0
    corrected_count:  int                  = 0
    uncorrected_count: int                 = 0

    def as_dict(self) -> dict:
        return {
            "total_stale":     self.total_stale,
            "corrected":       self.corrected_count,
            "uncorrected":     self.uncorrected_count,
            "affected_agents": self.affected_agents,
            "records": [
                {
                    "agent":            r.reader_agent,
                    "key":              r.key,
                    "stale_value":      r.stale_value,
                    "current_value":    r.current_value,
                    "version_lag":      r.version_lag,
                    "failure_type":     r.failure_type,
                    "severity":         r.severity,
                    "corrected":        r.was_corrected,
                    "correction_agent": r.correction_agent,
                    "consequence":      r.consequence,
                }
                for r in self.records
            ],
        }

    def render(self) -> str:
        if not self.records:
            return "  No stale memory reads detected."
        lines = [
            f"  {self.total_stale} stale read(s): "
            f"{self.corrected_count} corrected, {self.uncorrected_count} unresolved",
            "",
        ]
        for rec in self.records:
            lines.append(rec.render())
            lines.append("")
        return "\n".join(lines)


def build_stale_memory_report(
    store: "SharedMemoryStore",
    # Optional: map of key → agent that corrected the stale read
    corrections: Dict[str, str] = None,
) -> StaleMemoryReport:
    """Detect stale reads from the store and annotate their consequences."""
    stale_reads = store.stale_reads()
    if not stale_reads:
        return StaleMemoryReport()

    corrections = corrections or {}
    records: List[StaleReadRecord] = []

    for read_ev in stale_reads:
        key = read_ev.key

        # Get current value at time of read
        current_record = store.current(key)
        current_value  = str(current_record.value) if current_record else "unknown"
        stale_value    = str(read_ev.value_read)

        # Look up domain consequence
        domain_info = _STALE_CONSEQUENCES.get(key, {
            "stale_decision": f"Agent used outdated {key} value.",
            "failure_type":   "stale_read",
            "severity":       "MEDIUM",
        })

        corrector = corrections.get(f"{key}:{read_ev.reader_agent}", "")
        records.append(StaleReadRecord(
            reader_agent     = read_ev.reader_agent,
            key              = key,
            stale_value      = stale_value,
            current_value    = current_value,
            version_lag      = read_ev.version_lag,
            consequence      = domain_info["stale_decision"],
            failure_type     = domain_info["failure_type"],
            severity         = domain_info["severity"],
            was_corrected    = bool(corrector),
            correction_agent = corrector,
        ))

    affected = list(dict.fromkeys(r.reader_agent for r in records))
    corrected = sum(1 for r in records if r.was_corrected)

    return StaleMemoryReport(
        records           = records,
        affected_agents   = affected,
        total_stale       = len(records),
        corrected_count   = corrected,
        uncorrected_count = len(records) - corrected,
    )
