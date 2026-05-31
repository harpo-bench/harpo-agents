"""
Executive Memory Forensics Report

Generates a professional postmortem of memory-related trajectory failures.
Designed to read like a "Postmortem of a failed planning process", not an event log.

REPORT STRUCTURE
----------------
  1. Executive Summary
  2. Memory Root Causes
  3. Stale Memory Events
  4. Propagation Chains
  5. Recovery Events
  6. Memory Damage Attribution
  7. Remaining Risks
  8. Final Assessment

DESIGN PRINCIPLES
-----------------
  - Human-readable narrative: "Engineering Lead operated on a $5M budget
    assumption while Finance had already revised the constraint to $2M."
  - Causal language: "caused", "propagated to", "prevented by", "corrected by"
  - Verdict-first: each section leads with the finding, evidence second
  - No raw event logs: all data is synthesised into sentences
  - Actionable remaining risks: not "stale read detected" but
    "the regulatory compliance gap remains unresolved and creates GDPR liability"
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.stale_memory_detector import StaleMemoryReport
    from harpo.memory.memory_damage_attribution import MemoryDamageReport
    from harpo.memory.correction_vs_recovery import CorrectionRecoveryReport
    from harpo.memory.multi_hop_propagation import MultiHopReport
    from harpo.memory.memory_vs_reflection import MemoryVsReflectionReport
    from harpo.memory.root_cause_memory import MemoryRootCauseReport
    from harpo.memory.contribution_analysis import ContributionAttribution

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

_W = 70  # text wrap width


def _wrap(text: str, indent: str = "  ") -> str:
    return textwrap.fill(text, width=_W, initial_indent=indent, subsequent_indent=indent)


def _sep(title: str = "", char: str = "─", width: int = 70) -> str:
    if not title:
        return char * width
    pad = max(0, width - len(title) - 2)
    return f"{char * (pad // 2)} {title} {char * (pad - pad // 2)}"


# ── Report dataclass ──────────────────────────────────────────────────────────

@dataclass
class MemoryForensicsReport:
    """
    Complete executive memory forensics report.
    Call .render() to get the full formatted string.
    """
    # Source data
    stale_report:     Optional[object] = None
    damage_report:    Optional[object] = None
    cr_report:        Optional[object] = None
    multi_hop:        Optional[object] = None
    mvr_report:       Optional[object] = None
    root_cause_report: Optional[object] = None
    contribution:     Optional[object] = None

    scenario_name:    str = "Multi-Agent Planning Trajectory"
    agent_count:      int = 0
    total_steps:      int = 0
    generated_at:     str = field(default_factory=lambda: datetime.now().strftime("%Y-%m-%d %H:%M UTC"))

    def render(self) -> str:
        sections = [
            self._header(),
            self._section_1_executive_summary(),
            self._section_2_root_causes(),
            self._section_3_stale_events(),
            self._section_4_propagation(),
            self._section_5_recovery(),
            self._section_6_attribution(),
            self._section_7_remaining_risks(),
            self._section_8_verdict(),
            self._footer(),
        ]
        return "\n\n".join(s for s in sections if s)

    # ── Header ────────────────────────────────────────────────────────────────

    def _header(self) -> str:
        return (
            "═" * _W + "\n"
            f"  HARPO MEMORY FORENSICS REPORT\n"
            f"  {self.scenario_name}\n"
            f"  Generated: {self.generated_at}\n"
            + "═" * _W
        )

    # ── Section 1: Executive Summary ─────────────────────────────────────────

    def _section_1_executive_summary(self) -> str:
        lines = [_sep("1. EXECUTIVE SUMMARY")]

        sr = self.stale_report
        cr = self.cr_report
        rc = self.root_cause_report
        da = self.damage_report

        if sr is None or sr.total_stale == 0:
            lines.append("  No memory failures were detected in this trajectory.")
            return "\n".join(lines)

        n_stale     = sr.total_stale
        n_corrected = sr.corrected_count
        n_unres     = sr.uncorrected_count
        n_agents    = len(sr.affected_agents)

        # Opening finding
        lines.append(_wrap(
            f"This trajectory experienced {n_stale} stale memory read(s) affecting "
            f"{n_agents} agent(s). "
            f"{n_corrected} were corrected during execution; "
            f"{n_unres} remained unresolved at trajectory end.",
        ))

        # Root cause lead
        if rc and rc.root_causes:
            top = rc.root_causes[0]
            lines.append("")
            lines.append(_wrap(
                f"The highest-impact memory failure involved {top.display_name}: "
                f"{top.most_expensive_consequence}",
            ))

        # Correction / recovery distinction
        if cr:
            n_corrections = len(cr.corrections)
            n_recoveries  = len(cr.recoveries)
            if n_corrections > 0 and n_recoveries == 0:
                lines.append("")
                lines.append(_wrap(
                    f"Notably: {n_corrections} correction(s) occurred at the data layer "
                    f"(the memory store was updated with correct values), but NO corresponding "
                    f"behavioral recovery was detected — agents did not revise their downstream "
                    f"plans after the corrections were issued. This is the most common and most "
                    f"dangerous failure mode: the truth existed in memory, but no agent used it.",
                ))
            elif n_recoveries > 0:
                lines.append("")
                lines.append(_wrap(
                    f"{n_corrections} memory correction(s) produced {n_recoveries} "
                    f"behavioral recovery event(s). Agents revised their plans "
                    f"after receiving corrected values.",
                ))

        # Damage summary
        if da and da.entries:
            lines.append("")
            total_d = da.total_damage
            lines.append(_wrap(
                f"Estimated memory-induced damage: {total_d:.2f} "
                f"(corrected: {da.corrected_damage:.2f}, unresolved: {da.uncorrected_damage:.2f}).",
            ))

        return "\n".join(lines)

    # ── Section 2: Root Causes ────────────────────────────────────────────────

    def _section_2_root_causes(self) -> str:
        lines = [_sep("2. MEMORY ROOT CAUSES")]

        rc = self.root_cause_report
        if rc is None or not rc.root_causes:
            lines.append("  No memory root causes identified.")
            return "\n".join(lines)

        for i, cause in enumerate(rc.root_causes, 1):
            rec_str = {
                "full":    "✓ Corrected and recovered",
                "partial": "△ Corrected but only partially recovered",
                "none":    "✗ Not corrected — persisted to trajectory end",
            }.get(cause.recovery_status, cause.recovery_status)

            affected_str = ", ".join(_dn(a) for a in cause.affected_agents[:4])
            if len(cause.affected_agents) > 4:
                affected_str += f" (+{len(cause.affected_agents) - 4} more)"

            lines += [
                "",
                f"  #{i}  {cause.display_name}  [{cause.severity}]  "
                f"impact={cause.combined_impact_score:.2f}",
                _wrap(f"Propagated to {cause.propagation_radius} agent(s) across "
                      f"{cause.propagation_depth} hop(s): {affected_str}.", "     "),
                f"     {rec_str}",
                _wrap(f"Key consequence: {cause.most_expensive_consequence}", "     "),
            ]

        lines += [
            "",
            f"  Highest damage:       {rc.most_damaging or '—'}",
            f"  Deepest propagation:  {rc.deepest_propagation or '—'}",
            f"  Hardest to repair:    {rc.hardest_to_repair or '—'}",
        ]
        return "\n".join(lines)

    # ── Section 3: Stale Memory Events ───────────────────────────────────────

    def _section_3_stale_events(self) -> str:
        lines = [_sep("3. STALE MEMORY EVENTS")]

        sr = self.stale_report
        if sr is None or not sr.records:
            lines.append("  No stale memory reads detected.")
            return "\n".join(lines)

        for rec in sr.records:
            status = "✓ corrected" if rec.was_corrected else "✗ NOT corrected"
            lines += [
                "",
                f"  [{rec.severity}]  {_dn(rec.reader_agent)}  ·  key: {rec.key}",
                f"     Read:    {rec.stale_value!r}  (should have been: {rec.current_value!r})",
                f"     Impact:  {rec.consequence}",
                f"     Status:  {status}"
                + (f" by {_dn(rec.correction_agent)}" if rec.was_corrected else ""),
            ]

        return "\n".join(lines)

    # ── Section 4: Propagation Chains ─────────────────────────────────────────

    def _section_4_propagation(self) -> str:
        lines = [_sep("4. PROPAGATION CHAINS")]

        mh = self.multi_hop
        if mh is None or not mh.chains:
            lines.append("  No multi-hop propagation detected.")
            return "\n".join(lines)

        lines += [
            f"  Max propagation depth: {mh.max_depth}",
            f"  Total agents contaminated: {mh.total_agents_affected}",
        ]

        for chain in mh.chains:
            lines.append("")
            lines.append(f"  {chain.key.upper()}  (stale value: {chain.stale_value!r})")
            lines.append(f"     Correct value was: {chain.correct_value!r}")
            for hop in chain.hops:
                depth_label = {0: "Direct stale reader", 1: "1st indirect", 2: "2nd indirect"}.get(
                    hop.hop_depth, f"depth-{hop.hop_depth}"
                )
                arrow = "  " * hop.hop_depth + "→"
                lines.append(
                    f"     {arrow} {_dn(hop.agent_id)} [{depth_label}]  "
                    f"via {hop.how_inherited}"
                )

        return "\n".join(lines)

    # ── Section 5: Recovery Events ────────────────────────────────────────────

    def _section_5_recovery(self) -> str:
        lines = [_sep("5. RECOVERY EVENTS")]

        cr  = self.cr_report
        mvr = self.mvr_report

        if cr is None:
            lines.append("  Recovery analysis not available.")
            return "\n".join(lines)

        if not cr.corrections and not cr.recoveries:
            lines.append("  No corrections or recoveries detected.")
            return "\n".join(lines)

        # Corrections without recovery (most dangerous)
        if cr.corrections_without_recovery:
            lines += [
                "",
                "  CORRECTIONS WITHOUT BEHAVIORAL RECOVERY (data fixed; agents didn't act):",
            ]
            for key in cr.corrections_without_recovery:
                corr = next((c for c in cr.corrections if c.key == key), None)
                if corr:
                    lines.append(_wrap(
                        f"  • {key}: {_dn(corr.correction_agent)} updated the value to "
                        f"{corr.corrected_value!r}, but the following agents never revised "
                        f"their plans: {', '.join(_dn(a) for a in corr.stale_agents)}.",
                        "    "
                    ))

        # Successful recoveries
        if cr.recoveries:
            lines += [
                "",
                "  SUCCESSFUL RECOVERIES (data corrected AND agent revised its plans):",
            ]
            for rec in cr.recoveries:
                lines += [
                    "",
                    f"  [{rec.key}]  {_dn(rec.recovering_agent)}",
                    _wrap(rec.recovery_impact, "     "),
                ]
                # Attribution
                if mvr:
                    attr = next(
                        (a for a in mvr.attributions if a.key == rec.key
                         and a.recovering_agent == rec.recovering_agent),
                        None,
                    )
                    if attr:
                        lines.append(
                            f"     Recovery driver: {attr.primary_cause.replace('_', ' ')}  "
                            f"(confidence {attr.confidence:.2f})"
                        )

        # Cross-link summary
        lines += [
            "",
            f"  Corrections that led to recovery:     {', '.join(cr.corrections_with_recovery) or '—'}",
            f"  Corrections without recovery:          {', '.join(cr.corrections_without_recovery) or '—'}",
        ]

        return "\n".join(lines)

    # ── Section 6: Memory Damage Attribution ─────────────────────────────────

    def _section_6_attribution(self) -> str:
        lines = [_sep("6. MEMORY DAMAGE ATTRIBUTION")]

        da = self.damage_report
        ca = self.contribution

        if da and da.entries:
            lines += [
                "",
                "  DAMAGE BY STALE READ:",
            ]
            for entry in da.entries:
                cas = ", ".join(_dn(a) for a in entry.cascading_agents[:3])
                lines += [
                    f"  [{entry.failure_label}]  {_dn(entry.reader_agent)}  "
                    f"(key={entry.key})",
                    f"     Damage: {entry.damage_score:.2f}  "
                    + ("✓ corrected" if entry.was_corrected else "✗ unresolved"),
                    _wrap(entry.damage_narrative, "     "),
                    (f"     Cascaded to: {cas}" if cas else ""),
                    "",
                ]

        if ca:
            lines += [
                "",
                "  TRAJECTORY DEGRADATION ATTRIBUTION ACROSS FAILURE CATEGORIES:",
                "",
                ca.render(),
            ]

        return "\n".join(lines)

    # ── Section 7: Remaining Risks ────────────────────────────────────────────

    def _section_7_remaining_risks(self) -> str:
        lines = [_sep("7. REMAINING RISKS")]

        sr = self.stale_report
        cr = self.cr_report
        rc = self.root_cause_report

        risks_found = False

        if sr:
            for rec in sr.records:
                if not rec.was_corrected:
                    risks_found = True
                    lines += [
                        "",
                        f"  [UNRESOLVED — {rec.severity}]  {rec.key}  "
                        f"held by {_dn(rec.reader_agent)}",
                        _wrap(
                            f"The stale value {rec.stale_value!r} was never corrected. "
                            f"{rec.consequence} This risk persisted to the end of the "
                            f"planning session.",
                            "     "
                        ),
                    ]

        if cr and cr.corrections_without_recovery:
            for key in cr.corrections_without_recovery:
                corr = next((c for c in cr.corrections if c.key == key), None)
                if corr:
                    risks_found = True
                    lines += [
                        "",
                        f"  [CORRECTION WITHOUT RECOVERY — {corr.severity}]  {key}",
                        _wrap(
                            f"The memory store was corrected to {corr.corrected_value!r} "
                            f"by {_dn(corr.correction_agent)}, but agents "
                            f"({', '.join(_dn(a) for a in corr.stale_agents)}) "
                            f"never revised their plans. Their outputs may still reflect "
                            f"the outdated value.",
                            "     "
                        ),
                    ]

        if not risks_found:
            lines.append("  No unresolved memory risks detected at trajectory end.")

        return "\n".join(lines)

    # ── Section 8: Final Verdict ──────────────────────────────────────────────

    def _section_8_verdict(self) -> str:
        lines = [_sep("8. FINAL ASSESSMENT")]
        lines.append("")

        sr = self.stale_report
        cr = self.cr_report
        rc = self.root_cause_report

        if sr is None or sr.total_stale == 0:
            lines.append(_wrap(
                "No memory failures were detected. Memory did not contribute to "
                "trajectory degradation in this run."
            ))
            return "\n".join(lines)

        n_stale   = sr.total_stale
        n_unres   = sr.uncorrected_count
        n_rec     = len(cr.recoveries) if cr else 0
        n_corr    = len(cr.corrections) if cr else 0
        n_corr_wo = len(cr.corrections_without_recovery) if cr else 0

        # Determine overall verdict
        if n_unres == 0 and n_rec == n_corr:
            verdict = "RESOLVED"
            verdict_detail = (
                "All stale memory reads were corrected and all agents revised their "
                "downstream plans. Memory failures were detected and fully repaired "
                "within this trajectory."
            )
        elif n_unres == 0 and n_corr_wo > 0:
            verdict = "PARTIALLY RESOLVED"
            verdict_detail = (
                f"All stale values were corrected at the data layer, but {n_corr_wo} "
                f"correction(s) did not produce behavioral recovery — the affected agents "
                f"continued operating on outdated assumptions despite the memory update. "
                f"The planning process was corrected on paper but not in practice."
            )
        elif n_unres > 0 and n_rec > 0:
            verdict = "PARTIALLY RESOLVED"
            verdict_detail = (
                f"{n_rec} recovery event(s) occurred, but {n_unres} stale read(s) "
                f"remained unresolved. The trajectory was partially stabilized but "
                f"carries forward unresolved memory risks."
            )
        else:
            verdict = "UNRESOLVED"
            verdict_detail = (
                f"All {n_stale} stale read(s) remained uncorrected. No behavioral "
                f"recovery was detected. The trajectory completed with incorrect "
                f"assumptions embedded throughout agent reasoning."
            )

        lines += [
            f"  VERDICT:  {verdict}",
            "",
            _wrap(verdict_detail),
        ]

        # Top root cause reminder
        if rc and rc.root_causes:
            top = rc.root_causes[0]
            lines += [
                "",
                _wrap(
                    f"Primary root cause: {top.display_name} — "
                    f"{top.most_expensive_consequence}",
                ),
            ]

        return "\n".join(lines)

    def _footer(self) -> str:
        return "═" * _W


# ── Public builder ────────────────────────────────────────────────────────────

def build_memory_forensics_report(
    stale_report:      Optional[object] = None,
    damage_report:     Optional[object] = None,
    cr_report:         Optional[object] = None,
    multi_hop:         Optional[object] = None,
    mvr_report:        Optional[object] = None,
    root_cause_report: Optional[object] = None,
    contribution:      Optional[object] = None,
    scenario_name:     str = "Multi-Agent Planning Trajectory",
    agent_count:       int = 0,
    total_steps:       int = 0,
) -> MemoryForensicsReport:
    """
    Assemble all available memory analysis into an executive forensics report.

    All inputs are optional. Provide all available reports for the richest output.
    """
    return MemoryForensicsReport(
        stale_report      = stale_report,
        damage_report     = damage_report,
        cr_report         = cr_report,
        multi_hop         = multi_hop,
        mvr_report        = mvr_report,
        root_cause_report = root_cause_report,
        contribution      = contribution,
        scenario_name     = scenario_name,
        agent_count       = agent_count,
        total_steps       = total_steps,
    )
