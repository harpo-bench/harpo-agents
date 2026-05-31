"""
Memory Damage Attribution

Links stale memory reads to specific planning failures.
Answers: "How much trajectory degradation was caused by memory?"

For each stale read, estimates:
  - damage score (0-1)
  - cascading agents (who made decisions based on the stale reader's output)
  - total cost of the memory error

Damage estimation heuristics:
  - Critical key (budget/regulatory) → base damage 0.70
  - High-radius stale read (spread to 2+ downstream agents) → +0.15
  - Uncorrected stale → ×1.5 multiplier
  - Corrected before final output → ×0.4 multiplier
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleReadRecord, StaleMemoryReport

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


_KEY_BASE_DAMAGE: Dict[str, float] = {
    "budget":                  0.65,
    "scope":                   0.70,
    "launch_date":             0.50,
    "regulatory_requirements": 0.80,
    "staffing":                0.45,
    "market_priorities":       0.40,
}

_FAILURE_TYPE_LABEL: Dict[str, str] = {
    "resource_overestimation": "Resource over-allocation",
    "scope_mismatch":          "Geographic scope mismatch",
    "timeline_conflict":       "Timeline conflict",
    "compliance_gap":          "Regulatory compliance gap",
    "resource_mismatch":       "Staffing mismatch",
    "misaligned_priorities":   "Market priority misalignment",
    "stale_read":              "Generic stale read",
}


@dataclass
class MemoryDamageEntry:
    """Damage caused by one stale read."""
    key:               str
    reader_agent:      str
    stale_value:       str
    correct_value:     str
    failure_label:     str
    damage_score:      float
    cascading_agents:  List[str]   # downstream agents affected by this agent's stale decision
    was_corrected:     bool
    damage_narrative:  str         # "Engineering Lead over-allocated $3M in resources..."

    def render(self) -> str:
        cas = ", ".join(_dn(a) for a in self.cascading_agents[:3])
        cor = "CORRECTED" if self.was_corrected else "UNRESOLVED"
        return (
            f"  [{self.failure_label}]  {_dn(self.reader_agent)} (stale {self.key})\n"
            f"     Damage Score:  {self.damage_score:.2f}   Status: {cor}\n"
            f"     Cascading To:  {cas or '—'}\n"
            f"     Narrative:     {self.damage_narrative}"
        )


@dataclass
class MemoryDamageReport:
    """Complete damage attribution for the trajectory."""
    entries:            List[MemoryDamageEntry] = field(default_factory=list)
    total_damage:       float                   = 0.0
    corrected_damage:   float                   = 0.0
    uncorrected_damage: float                   = 0.0
    pct_from_memory:    float                   = 0.0   # estimated % of trajectory degradation

    def as_dict(self) -> dict:
        return {
            "total_damage":       round(self.total_damage, 3),
            "corrected_damage":   round(self.corrected_damage, 3),
            "uncorrected_damage": round(self.uncorrected_damage, 3),
            "pct_from_memory":    round(self.pct_from_memory, 1),
            "entries": [
                {
                    "key":            e.key,
                    "agent":          e.reader_agent,
                    "failure_type":   e.failure_label,
                    "damage_score":   round(e.damage_score, 3),
                    "cascading_to":   e.cascading_agents,
                    "corrected":      e.was_corrected,
                    "narrative":      e.damage_narrative,
                }
                for e in self.entries
            ],
        }

    def render(self) -> str:
        if not self.entries:
            return "  No memory-induced damage detected."
        lines = [
            f"  Total memory damage score: {self.total_damage:.2f}",
            f"  Corrected: {self.corrected_damage:.2f}  "
            f"Unresolved: {self.uncorrected_damage:.2f}",
            f"  Estimated % of trajectory degradation from memory: "
            f"{self.pct_from_memory:.0f}%",
            "",
        ]
        for entry in self.entries:
            lines.append(entry.render())
            lines.append("")
        return "\n".join(lines)


# Downstream impact per agent for the product launch scenario
_DOWNSTREAM_MAP: Dict[str, List[str]] = {
    "engineering-lead": ["product-manager", "operations-lead"],
    "marketing-lead":   ["product-manager", "operations-lead"],
    "operations-lead":  ["product-manager", "engineering-lead"],
}

_DAMAGE_NARRATIVES: Dict[str, str] = {
    "budget:engineering-lead": (
        "Engineering Lead planned development resources assuming $5M budget, "
        "leading to a $3M over-allocation relative to the actual $2M constraint."
    ),
    "scope:marketing-lead": (
        "Marketing Lead designed a US-only campaign while the actual requirement "
        "mandated EU coverage, creating a compliance and go-to-market gap."
    ),
    "launch_date:operations-lead": (
        "Operations Lead scheduled logistics and vendor contracts for December, "
        "while Finance had updated the launch to March, causing contract conflicts."
    ),
    "regulatory_requirements:marketing-lead": (
        "Marketing Lead ignored updated EU regulatory requirements, "
        "designing non-compliant messaging for EU markets."
    ),
}


def build_memory_damage_report(
    stale_report: "StaleMemoryReport",
    overall_trajectory_score: float = 0.5,
) -> MemoryDamageReport:
    """Attribute damage to stale memory reads."""
    entries: List[MemoryDamageEntry] = []

    for rec in stale_report.records:
        base = _KEY_BASE_DAMAGE.get(rec.key, 0.40)
        cascading = _DOWNSTREAM_MAP.get(rec.reader_agent, [])

        # Adjust damage
        damage = base
        damage += len(cascading) * 0.05   # each downstream agent adds risk
        if not rec.was_corrected:
            damage *= 1.40
        else:
            damage *= 0.45
        damage = round(min(damage, 0.95), 3)

        # Use domain-specific narrative or fallback
        narrative_key = f"{rec.key}:{rec.reader_agent}"
        narrative = _DAMAGE_NARRATIVES.get(
            narrative_key,
            f"{_dn(rec.reader_agent)} used stale {rec.key} ({rec.stale_value!r}) "
            f"instead of current value ({rec.current_value!r}).",
        )

        entries.append(MemoryDamageEntry(
            key              = rec.key,
            reader_agent     = rec.reader_agent,
            stale_value      = rec.stale_value,
            correct_value    = rec.current_value,
            failure_label    = _FAILURE_TYPE_LABEL.get(rec.failure_type, rec.failure_type),
            damage_score     = damage,
            cascading_agents = cascading,
            was_corrected    = rec.was_corrected,
            damage_narrative = narrative,
        ))

    entries.sort(key=lambda e: e.damage_score, reverse=True)

    total         = sum(e.damage_score for e in entries)
    corrected_d   = sum(e.damage_score for e in entries if e.was_corrected)
    uncorrected_d = sum(e.damage_score for e in entries if not e.was_corrected)

    # Rough estimate: percentage of overall degradation attributable to memory
    # (normalized against 1 - overall_score = total degradation space)
    degradation_space = 1.0 - overall_trajectory_score
    pct = min(total / (degradation_space * 3 + 0.01) * 100, 95.0) if entries else 0.0

    return MemoryDamageReport(
        entries            = entries,
        total_damage       = round(total, 3),
        corrected_damage   = round(corrected_d, 3),
        uncorrected_damage = round(uncorrected_d, 3),
        pct_from_memory    = round(pct, 1),
    )
