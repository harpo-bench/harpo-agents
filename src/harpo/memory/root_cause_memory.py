"""
Memory Root Cause Intelligence

Ranks memory keys by the damage they caused and answers:

  1. Which memory caused the largest damage?
  2. Which stale memory propagated furthest?
  3. Which memory was most expensive to the trajectory?
  4. Which memory was hardest to repair?

This is a synthesis layer that aggregates signals from:
  - StaleMemoryReport      (stale read counts, severity)
  - MemoryDamageReport     (damage scores per key)
  - MultiHopReport         (propagation depth and radius)
  - CorrectionRecoveryReport (correction and recovery events)
  - MemoryVsReflectionReport (recovery difficulty signals)

Output:
  Top Memory Root Causes, ranked by combined impact score.

Example output:
  #1  Budget Memory
      Damage:            0.73
      Propagation depth: 2
      Affected agents:   Engineering Lead, Product Manager, Operations Lead
      Recovery:          Successful (via Finance Lead, confidence 0.85)
      Most expensive consequence: $3M resource over-allocation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.stale_memory_detector import StaleMemoryReport
    from harpo.memory.memory_damage_attribution import MemoryDamageReport
    from harpo.memory.multi_hop_propagation import MultiHopReport
    from harpo.memory.correction_vs_recovery import CorrectionRecoveryReport
    from harpo.memory.memory_vs_reflection import MemoryVsReflectionReport

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


_KEY_DISPLAY: Dict[str, str] = {
    "budget":                  "Budget Memory",
    "scope":                   "Scope Memory",
    "launch_date":             "Launch Date Memory",
    "regulatory_requirements": "Regulatory Requirements Memory",
    "staffing":                "Staffing Memory",
    "market_priorities":       "Market Priorities Memory",
}

def _kd(key: str) -> str:
    return _KEY_DISPLAY.get(key, key.replace("_", " ").title() + " Memory")


# ── Root cause record ─────────────────────────────────────────────────────────

@dataclass
class MemoryRootCause:
    """
    Aggregated root cause profile for one memory key.

    combined_impact_score (0-1):
        Weighted combination of damage, propagation, and recovery difficulty.
        Weights: damage 0.45, propagation 0.30, unrecoverability 0.25.
    """
    key:                        str
    display_name:               str
    combined_impact_score:      float
    damage_score:               float     # from MemoryDamageReport
    propagation_depth:          int       # from MultiHopReport
    propagation_radius:         int       # number of agents affected
    affected_agents:            List[str]
    recovery_status:            str       # "full" | "partial" | "none"
    recovery_confidence:        float     # from MemoryVsReflectionReport
    repair_difficulty:          float     # 0-1: higher = harder to repair
    most_expensive_consequence: str
    stale_count:                int       # number of stale reads for this key
    severity:                   str       # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    corrected_by:               str       # agent that issued correction

    def rank_label(self) -> str:
        if self.combined_impact_score >= 0.70:
            return "CRITICAL IMPACT"
        elif self.combined_impact_score >= 0.45:
            return "HIGH IMPACT"
        elif self.combined_impact_score >= 0.25:
            return "MODERATE IMPACT"
        else:
            return "LOW IMPACT"

    def render(self, rank: int) -> str:
        affected_str = ", ".join(_dn(a) for a in self.affected_agents[:5])
        rec_str = {
            "full":    f"✓ Full recovery  (confidence={self.recovery_confidence:.2f})",
            "partial": f"△ Partial recovery (confidence={self.recovery_confidence:.2f})",
            "none":    "✗ Not recovered — consequence persisted to trajectory end",
        }.get(self.recovery_status, self.recovery_status)
        return (
            f"  #{rank}  {self.display_name}  [{self.rank_label()}]\n"
            f"     Combined impact:     {self.combined_impact_score:.2f}\n"
            f"     Damage score:        {self.damage_score:.2f}  "
            f"(severity={self.severity})\n"
            f"     Propagation:         depth={self.propagation_depth}, "
            f"radius={self.propagation_radius} agent(s)\n"
            f"     Affected agents:     {affected_str or '—'}\n"
            f"     Recovery:            {rec_str}\n"
            f"     Repair difficulty:   {self.repair_difficulty:.2f}  "
            f"(corrected by {_dn(self.corrected_by) or '—'})\n"
            f"     Key consequence:     {self.most_expensive_consequence[:100]}"
        )


# ── Root cause report ─────────────────────────────────────────────────────────

@dataclass
class MemoryRootCauseReport:
    """
    Ranked list of memory keys by combined impact, with analysis of which
    was most damaging, furthest-propagating, most expensive, and hardest to repair.
    """
    root_causes:           List[MemoryRootCause] = field(default_factory=list)
    most_damaging:         Optional[str]          = None   # key
    deepest_propagation:   Optional[str]          = None   # key
    most_expensive:        Optional[str]          = None   # key
    hardest_to_repair:     Optional[str]          = None   # key
    summary:               str                   = ""

    def as_dict(self) -> dict:
        return {
            "most_damaging":       self.most_damaging,
            "deepest_propagation": self.deepest_propagation,
            "most_expensive":      self.most_expensive,
            "hardest_to_repair":   self.hardest_to_repair,
            "summary":             self.summary,
            "root_causes": [
                {
                    "key":               rc.key,
                    "display_name":      rc.display_name,
                    "combined_impact":   round(rc.combined_impact_score, 3),
                    "damage":            round(rc.damage_score, 3),
                    "propagation_depth": rc.propagation_depth,
                    "radius":            rc.propagation_radius,
                    "affected_agents":   rc.affected_agents,
                    "recovery_status":   rc.recovery_status,
                    "recovery_confidence": round(rc.recovery_confidence, 3),
                    "repair_difficulty": round(rc.repair_difficulty, 3),
                    "severity":          rc.severity,
                }
                for rc in self.root_causes
            ],
        }

    def render(self) -> str:
        lines = ["  TOP MEMORY ROOT CAUSES (ranked by combined impact)"]
        lines.append(f"  {self.summary}")
        lines.append("")
        for i, rc in enumerate(self.root_causes, 1):
            lines.append(rc.render(rank=i))
            lines.append("")
        lines += [
            "  ─────────────────────────────────────────────────────────────",
            f"  Most damaging:        {_kd(self.most_damaging or '—')}",
            f"  Furthest propagation: {_kd(self.deepest_propagation or '—')}",
            f"  Most expensive:       {_kd(self.most_expensive or '—')}",
            f"  Hardest to repair:    {_kd(self.hardest_to_repair or '—')}",
        ]
        return "\n".join(lines)


# ── Domain knowledge ──────────────────────────────────────────────────────────

_CONSEQUENCES_BY_KEY: Dict[str, str] = {
    "budget": "$3M resource over-allocation forced emergency re-planning across engineering and operations",
    "scope": "EU market gap created both a compliance risk and a missed revenue opportunity at launch",
    "launch_date": "December vendor contracts incur cancellation penalties when date moved to March",
    "regulatory_requirements": "GDPR non-compliance risk of up to €20M fine for EU launch",
}

_SEVERITY_MAP: Dict[str, str] = {
    "budget":                  "HIGH",
    "scope":                   "HIGH",
    "launch_date":             "MEDIUM",
    "regulatory_requirements": "CRITICAL",
    "staffing":                "MEDIUM",
    "market_priorities":       "LOW",
}

_CORRECTION_AGENTS: Dict[str, str] = {
    "budget":                  "finance-lead",
    "scope":                   "legal-lead",
    "launch_date":             "finance-lead",
    "regulatory_requirements": "legal-lead",
}


# ── Builder ───────────────────────────────────────────────────────────────────

def build_memory_root_cause_report(
    stale_report:   "StaleMemoryReport",
    damage_report:  Optional["MemoryDamageReport"]          = None,
    multi_hop:      Optional["MultiHopReport"]              = None,
    cr_report:      Optional["CorrectionRecoveryReport"]    = None,
    mvr_report:     Optional["MemoryVsReflectionReport"]    = None,
) -> MemoryRootCauseReport:
    """Aggregate all available analysis into a ranked root cause report."""

    if not stale_report.records:
        return MemoryRootCauseReport(summary="No stale memory reads detected.")

    # ── Build per-key lookup tables ───────────────────────────────────────────
    damage_by_key: Dict[str, float] = {}
    consequence_by_key: Dict[str, str] = {}
    if damage_report:
        for entry in damage_report.entries:
            damage_by_key[entry.key]      = max(damage_by_key.get(entry.key, 0.0),
                                                 entry.damage_score)
            consequence_by_key[entry.key] = entry.damage_narrative

    hop_depth_by_key:   Dict[str, int]       = {}
    hop_radius_by_key:  Dict[str, int]       = {}
    affected_by_key:    Dict[str, List[str]] = {}
    if multi_hop:
        for chain in multi_hop.chains:
            hop_depth_by_key[chain.key]   = max(hop_depth_by_key.get(chain.key, 0), chain.depth())
            hop_radius_by_key[chain.key]  = max(hop_radius_by_key.get(chain.key, 0), chain.propagation_radius())
            affected_by_key[chain.key]    = chain.affected_agents()

    recovery_status_by_key: Dict[str, str]   = {}
    recovery_conf_by_key:   Dict[str, float] = {}
    corrected_by_key:       Dict[str, str]   = {}
    if cr_report:
        corrected_keys = set(cr_report.corrections_with_recovery)
        for rec in cr_report.recoveries:
            recovery_status_by_key[rec.key]  = "full"
            corrected_by_key[rec.key]        = rec.predecessor_correction.correction_agent if rec.predecessor_correction else ""
        for c in cr_report.corrections:
            if c.key not in recovery_status_by_key:
                recovery_status_by_key[c.key] = "partial"
            corrected_by_key.setdefault(c.key, c.correction_agent)

    if mvr_report:
        for attr in mvr_report.attributions:
            recovery_conf_by_key[attr.key] = attr.confidence

    # ── Build root causes ─────────────────────────────────────────────────────
    all_keys: Set[str] = {r.key for r in stale_report.records}
    root_causes: List[MemoryRootCause] = []

    for key in all_keys:
        stale_recs = [r for r in stale_report.records if r.key == key]
        stale_count = len(stale_recs)

        damage       = damage_by_key.get(key, 0.40 * stale_count)
        depth        = hop_depth_by_key.get(key, 1)
        radius       = hop_radius_by_key.get(key, stale_count)
        affected     = affected_by_key.get(key, [r.reader_agent for r in stale_recs])
        rec_status   = recovery_status_by_key.get(key, "none")
        rec_conf     = recovery_conf_by_key.get(key, 0.0)
        corrected_by = corrected_by_key.get(key, _CORRECTION_AGENTS.get(key, ""))
        severity     = _SEVERITY_MAP.get(key, "MEDIUM")
        consequence  = consequence_by_key.get(key, _CONSEQUENCES_BY_KEY.get(key, f"Stale {key} caused planning errors"))

        # Repair difficulty: harder if uncorrected OR deep propagation OR multiple stale reads
        if rec_status == "none":
            repair_diff = 0.80
        elif rec_status == "partial":
            repair_diff = 0.55
        else:
            repair_diff = max(0.10, 0.30 - depth * 0.05 - radius * 0.03)

        # Combined impact score (0-1)
        damage_norm     = min(damage, 1.0)
        prop_norm       = min(depth / 4.0 + radius / 6.0, 1.0) / 2.0  # max contribution 0.5
        unrec_norm      = repair_diff * 0.50
        combined        = round(damage_norm * 0.45 + prop_norm * 0.30 + unrec_norm * 0.25, 3)

        root_causes.append(MemoryRootCause(
            key                         = key,
            display_name                = _kd(key),
            combined_impact_score       = combined,
            damage_score                = round(damage, 3),
            propagation_depth           = depth,
            propagation_radius          = radius,
            affected_agents             = affected,
            recovery_status             = rec_status,
            recovery_confidence         = round(rec_conf, 3),
            repair_difficulty           = round(repair_diff, 3),
            most_expensive_consequence  = consequence,
            stale_count                 = stale_count,
            severity                    = severity,
            corrected_by                = corrected_by,
        ))

    root_causes.sort(key=lambda rc: rc.combined_impact_score, reverse=True)

    # ── Analysis labels ───────────────────────────────────────────────────────
    most_damaging   = max(root_causes, key=lambda rc: rc.damage_score).key if root_causes else None
    deepest         = max(root_causes, key=lambda rc: rc.propagation_depth).key if root_causes else None
    most_expensive  = most_damaging  # damage score proxies cost in this domain
    hardest_repair  = max(root_causes, key=lambda rc: rc.repair_difficulty).key if root_causes else None

    n_unresolved = sum(1 for rc in root_causes if rc.recovery_status == "none")
    summary = (
        f"{len(root_causes)} memory root cause(s) identified. "
        f"Top impact: {_kd(root_causes[0].key)} (score={root_causes[0].combined_impact_score:.2f}). "
        f"{n_unresolved} unresolved at trajectory end."
    )

    return MemoryRootCauseReport(
        root_causes          = root_causes,
        most_damaging        = most_damaging,
        deepest_propagation  = deepest,
        most_expensive       = most_expensive,
        hardest_to_repair    = hardest_repair,
        summary              = summary,
    )
