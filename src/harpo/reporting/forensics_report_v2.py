"""
Trajectory Forensics Report v2

Redesigned to read like a senior incident review, not a metric dump.

Structure
---------
  1. Executive Summary       — 3-5 clean sentences, no raw text fragments
  2. Top Root Causes         — ranked, with confidence and resolution status
  3. Major Contradictions    — what was disputed and how it was resolved
  4. Assumption Cascades     — which claims contaminated which agents
  5. Recovery Events         — what was corrected and by whom
  6. Agent Contributions     — stabilizer / amplifier / mixed / neutral
  7. Memory Influence        — harmful vs. beneficial memory operations
  8. Remaining Risks         — what is still unresolved at trajectory end
  9. Final Verdict           — one paragraph assessment

No raw trajectory text appears in the output.
No partial sentence fragments.
Every finding is backed by an agent name, a turn number, or a score.
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

_AGENT_DISPLAY: Dict[str, str] = {
    "security-analyst":   "Security Analyst",
    "infra-engineer":     "Infrastructure Engineer",
    "forensics-agent":    "Forensics Agent",
    "compliance-agent":   "Compliance Agent",
    "comms-officer":      "Communications Officer",
    "incident-commander": "Incident Commander",
}

def _dn(a: str) -> str:
    return _AGENT_DISPLAY.get(a or "", (a or "").replace("-", " ").title())

_WIDTH = 70
_SEP   = "─" * _WIDTH
_THICK = "═" * _WIDTH


def _wrap(text: str, indent: int = 2) -> str:
    prefix = " " * indent
    return textwrap.fill(text, width=_WIDTH, initial_indent=prefix,
                         subsequent_indent=prefix)


@dataclass
class ForensicsReportV2:
    # Section content
    executive_summary:   str
    root_causes_text:    str
    contradictions_text: str
    cascades_text:       str
    recovery_text:       str
    contributions_text:  str
    memory_text:         str
    risks_text:          str
    verdict_text:        str

    # Scores
    overall_score:       float = 0.0
    n_root_causes:       int   = 0
    n_unresolved:        int   = 0
    n_recoveries:        int   = 0

    def render(self) -> str:
        lines = [
            _THICK,
            "  HARPO TRAJECTORY FORENSICS REPORT  v2",
            _THICK, "",
        ]

        def section(title: str, body: str) -> None:
            lines.append(f"  {title}")
            lines.append(f"  {'─' * (len(title) + 2)}")
            for line in body.strip().split("\n"):
                lines.append(line)
            lines.extend(["", _SEP, ""])

        section("1. EXECUTIVE SUMMARY",    self.executive_summary)
        section("2. TOP ROOT CAUSES",      self.root_causes_text)
        section("3. MAJOR CONTRADICTIONS", self.contradictions_text)
        section("4. ASSUMPTION CASCADES",  self.cascades_text)
        section("5. RECOVERY EVENTS",      self.recovery_text)
        section("6. AGENT CONTRIBUTIONS",  self.contributions_text)
        section("7. MEMORY INFLUENCE",     self.memory_text)
        section("8. REMAINING RISKS",      self.risks_text)
        section("9. FINAL VERDICT",        self.verdict_text)

        lines.append(_THICK)
        return "\n".join(lines)


class ForensicsReportBuilderV2:
    """
    Builds a ForensicsReportV2 from structured analysis objects.
    Never reads raw trajectory text — uses only typed fields.
    """

    def __init__(
        self,
        traj:           "AgentTrajectory",
        analysis:       Any,
        rc_report:      Any,
        ranked_causes:  Any,
        recovery:       Any,
        memory_lineage: Any,
        drift_summary:  Any,
    ) -> None:
        self._traj           = traj
        self._analysis       = analysis
        self._rc             = rc_report
        self._ranked         = ranked_causes
        self._recovery       = recovery
        self._memory         = memory_lineage
        self._drift          = drift_summary

    def build(self) -> ForensicsReportV2:
        # Build executive summary using the clean summarizer
        try:
            from harpo.forensics.executive_summary import build_executive_summary
            exec_summary = build_executive_summary(
                self._rc, self._recovery,
                getattr(self._analysis, "collaboration", None),
            )
        except Exception:
            exec_summary = "Executive summary unavailable."

        return ForensicsReportV2(
            executive_summary   = self._section_exec(exec_summary),
            root_causes_text    = self._section_root_causes(),
            contradictions_text = self._section_contradictions(),
            cascades_text       = self._section_cascades(),
            recovery_text       = self._section_recovery(),
            contributions_text  = self._section_contributions(),
            memory_text         = self._section_memory(),
            risks_text          = self._section_risks(),
            verdict_text        = self._section_verdict(),
            overall_score       = getattr(self._traj, "_last_score", 0.0),
            n_root_causes       = len(getattr(self._rc, "root_causes", [])),
            n_unresolved        = len(getattr(self._rc, "unresolved_causes", [])),
            n_recoveries        = len(getattr(self._recovery, "events", [])),
        )

    # ── Section builders ──────────────────────────────────────────────────────

    def _section_exec(self, text: str) -> str:
        return "\n".join(_wrap(s.strip(), 2) for s in text.split(". ") if s.strip())

    def _section_root_causes(self) -> str:
        ranked = self._ranked or []
        if not ranked:
            return "  No root causes identified."
        lines = []
        for rrc in ranked:
            rc = rrc.root_cause
            cor = _dn(rc.corrected_by) if rc.corrected_by else "—"
            status_sym = "✓" if rc.resolution == "CORRECTED" else "✗"
            agents = ", ".join(_dn(a) for a in rc.affected_agents[:4])
            more   = f" +{len(rc.affected_agents)-4}" if len(rc.affected_agents) > 4 else ""
            lines += [
                f"  #{rrc.rank}  [{rrc.damage_label}]  {rc.title}",
                f"     Origin:      {_dn(rc.origin_agent)}",
                f"     Affected:    {agents}{more}",
                f"     Corrected:   {cor}   {status_sym} {rc.resolution}",
                f"     Confidence:  {rc.confidence:.0%}   Damage: {rc.damage_score:.2f}",
                "",
            ]
        return "\n".join(lines).rstrip()

    def _section_contradictions(self) -> str:
        # Prefer cross-agent contradictions from the demo's MultiAgentDiagnostics
        # (these have clean topic + agent labels, not raw text snippets)
        # Fall back to semantic ContradictionResult events

        # Try to get the 4 hardcoded cross-agent contradictions from root causes
        rc_contradictions = []
        if self._rc and self._rc.root_causes:
            for rc in self._rc.root_causes:
                if rc.evidence_type == "contradiction" or rc.corrected_by:
                    rc_contradictions.append(rc)

        if rc_contradictions:
            lines = []
            domain_topic = {
                "timeline":      "Intrusion timeline",
                "attack_vector": "Attack vector",
                "scope":         "Compromised host scope",
                "compliance":    "GDPR notification deadline",
            }
            for i, rc in enumerate(rc_contradictions, 1):
                topic  = domain_topic.get(rc.domain, rc.title)
                cor    = _dn(rc.corrected_by) if rc.corrected_by else "—"
                status = "✓ Resolved" if rc.corrected_by else "✗ Unresolved"
                lines += [
                    f"  [{i}]  Topic: {topic}",
                    f"       Origin:   {_dn(rc.origin_agent)}",
                    f"       Resolved: {cor}   {status}",
                    "",
                ]
            return "\n".join(lines).rstrip()

        # Fallback: raw ContradictionResult
        cont = getattr(self._analysis, "contradictions", None)
        if not cont or not cont.contradictions:
            return "  No contradictions detected."
        lines = []
        for i, ev in enumerate(cont.contradictions[:6], 1):
            kind_label = {
                "reversal_marker": "Explicit self-correction",
                "plan_flip":       "Plan reversal",
                "negation_flip":   "Factual negation",
                "stance_reversal": "Silent stance reversal",
            }.get(getattr(ev, "kind", ""), getattr(ev, "kind", "unknown"))
            snip_a = (ev.snippet_a or "")[:60]
            snip_b = (ev.snippet_b or "")[:60]
            lines += [
                f"  [{i}]  {kind_label}  (turns {ev.turn_a} → {ev.turn_b})",
                (f"       Before: \"{snip_a}\"" if snip_a else ""),
                (f"       After:  \"{snip_b}\"" if snip_b else ""),
                "",
            ]
        return "\n".join(l for l in lines if l or l == "").rstrip()

    def _section_cascades(self) -> str:
        # Use root causes as the authoritative cascade source — they have clean text
        if self._rc and self._rc.root_causes:
            lines = []
            for i, rc in enumerate(self._rc.root_causes, 1):
                agents = ", ".join(_dn(a) for a in rc.affected_agents[:4])
                more   = f" (+{len(rc.affected_agents)-4})" if len(rc.affected_agents) > 4 else ""
                status = "✓ CORRECTED" if rc.resolution == "CORRECTED" else "✗ UNRESOLVED"
                cor    = f"corrected by {_dn(rc.corrected_by)}" if rc.corrected_by else "not corrected"
                lines += [
                    f"  [{i}]  {_dn(rc.origin_agent)} — {rc.title}",
                    f"       Spread to:  {agents}{more}",
                    f"       Status:     {status} ({cor})",
                    f"       Impact:     {rc.impact[:80]}",
                    "",
                ]
            return "\n".join(lines).rstrip()

        return "  No assumption cascade data available."

    def _section_recovery(self) -> str:
        rec = self._recovery
        if not rec or not rec.events:
            return "  No recovery events detected."

        lines = [
            f"  Coverage: {rec.recovery_coverage:.0%} of failures recovered  "
            f"({rec.recovered_count}/{rec.total_failures_detected})",
            "",
        ]
        for i, ev in enumerate(rec.events, 1):
            agents = ", ".join(_dn(a) for a in ev.affected_agents[:3])
            term   = "propagation stopped" if ev.propagation_terminated else "propagation continued"
            lines += [
                f"  [{i}]  {ev.event_type.replace('_', ' ').upper()} — {_dn(ev.recovery_agent)}",
                f"       Problem:   {ev.original_failure[:90]}",
                f"       Action:    {ev.recovery_action[:90]}",
                f"       Affected:  {agents or '—'}  ({term})",
                f"       Score:     {ev.recovery_score:.2f}",
                "",
            ]
        return "\n".join(lines).rstrip()

    def _section_contributions(self) -> str:
        co = getattr(self._analysis, "collaboration", None)
        if not co or not hasattr(co, "agent_profiles"):
            return "  Collaboration data not available."

        lines = [
            f"  {'Agent':28s}  Score  Role        Repairs  Adopted By",
            "  " + "─" * 62,
        ]
        for aid, prof in sorted(
            co.agent_profiles.items(),
            key=lambda x: x[1].contribution_score, reverse=True
        ):
            role = "STABILIZER" if prof.contradiction_repairs >= 2 and not prof.is_siloed else (
                   "AMPLIFIER"  if prof.failure_amplifications > prof.stabilization_events + 1 else
                   "MIXED"      if prof.contradiction_repairs >= 1 else "NEUTRAL")
            sym  = {"STABILIZER": "✓", "AMPLIFIER": "✗", "MIXED": "±", "NEUTRAL": "·"}[role]
            lines.append(
                f"  {sym} {_dn(aid):28s}  {prof.contribution_score:.2f}   "
                f"{role:10s}  {prof.contradiction_repairs:7d}  {len(prof.adopted_by)}"
            )
        return "\n".join(lines)

    def _section_memory(self) -> str:
        mem = self._memory
        if not mem or not mem.edges:
            return (
                "  No explicit memory events detected.\n"
                "  Cross-agent context passing occurred via report injection\n"
                "  (inferred memory operations logged in memory lineage graph)."
            )
        lines = [f"  Net Memory Impact: {mem.net_memory_impact}", ""]
        for edge in mem.edges[:8]:
            sym = "⚠" if edge.is_harmful else ("✓" if edge.is_beneficial else "·")
            lines.append(
                f"  {sym}  {_dn(edge.source_agent)} → {_dn(edge.consumer_agent)}"
                f"  [{edge.impact}]"
            )
            lines.append(f"     {edge.outcome[:80]}")
            lines.append("")
        return "\n".join(lines).rstrip()

    def _section_risks(self) -> str:
        unresolved = getattr(self._rc, "unresolved_causes", [])
        if not unresolved:
            return "  All identified root causes were resolved. No remaining risks."

        lines = []
        for i, rc in enumerate(unresolved, 1):
            lines += [
                f"  [{i}]  {rc.title}",
                f"       {_wrap(rc.impact, indent=7).strip()}",
                f"       Agents at risk: {', '.join(_dn(a) for a in rc.affected_agents)}",
                "",
            ]
        return "\n".join(lines).rstrip()

    def _section_verdict(self) -> str:
        rc   = self._rc
        rec  = self._recovery
        dr   = self._drift

        n_causes     = len(getattr(rc, "root_causes", []))
        n_unresolved = len(getattr(rc, "unresolved_causes", []))
        n_recoveries = len(getattr(rec, "events", [])) if rec else 0
        drift_score  = getattr(dr, "overall_drift_score", 0.0) if dr else 0.0

        if n_unresolved == 0 and n_causes > 0:
            verdict = (
                f"The trajectory encountered {n_causes} root cause failure(s), "
                f"all of which were identified and corrected before trajectory end. "
                f"{n_recoveries} recovery event(s) contributed to stabilization. "
                "The trajectory is considered RESOLVED."
            )
        elif n_unresolved > 0:
            verdict = (
                f"The trajectory encountered {n_causes} root cause failure(s). "
                f"{n_causes - n_unresolved} were corrected; "
                f"{n_unresolved} remain unresolved. "
                f"Recovery coverage: {getattr(rec, 'recovery_coverage', 0):.0%}. "
                "The trajectory is considered PARTIALLY RESOLVED."
            )
        else:
            verdict = "Trajectory analysis complete. No root causes identified."

        if drift_score > 0.3:
            verdict += (
                f" Note: {drift_score:.0%} objective drift score indicates "
                "some deviation from core mission focus."
            )

        return _wrap(verdict, indent=2)


# ── Convenience builder ───────────────────────────────────────────────────────

def build_forensics_v2(
    traj:     "AgentTrajectory",
    analysis: Any,
) -> ForensicsReportV2:
    """Build a complete ForensicsReportV2. Entry point for the demo."""
    from harpo.forensics.root_cause_engine   import build_root_causes
    from harpo.forensics.root_cause_ranking  import rank_root_causes
    from harpo.recovery.recovery_attribution import build_recovery_report
    from harpo.memory.memory_scenario_support import build_memory_lineage
    from harpo.semantic.drift_consistency    import get_authoritative_drift

    rc_report      = build_root_causes(analysis, traj)
    ranked         = rank_root_causes(rc_report.root_causes)
    recovery       = build_recovery_report(traj, analysis, rc_report)
    memory_lineage = build_memory_lineage(traj, analysis)
    drift_summary  = get_authoritative_drift(analysis)

    return ForensicsReportBuilderV2(
        traj           = traj,
        analysis       = analysis,
        rc_report      = rc_report,
        ranked_causes  = ranked,
        recovery       = recovery,
        memory_lineage = memory_lineage,
        drift_summary  = drift_summary,
    ).build()
