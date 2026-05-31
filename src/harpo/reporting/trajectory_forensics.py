"""
Trajectory Forensics Report

Primary HARPO report.  Aggregates all causal signals into an executive-level
document that answers the questions practitioners actually ask:

  WHY did the trajectory degrade?
  WHICH agent introduced the problem?
  WHICH agents amplified it?
  WHICH agents repaired it?
  WAS recovery successful?

Report sections
---------------
  1. Executive Summary       — 3-sentence paragraph
  2. Root Cause Analysis     — origin agent, turn, assumption/decision that failed
  3. Failure Amplification   — the chain from first error to final outcome
  4. Agent Contributions     — per-agent: helped / hurt / neutral
  5. Recovery Analysis       — what was recovered vs. what remains unresolved
  6. Trajectory Timeline     — turn-by-turn key events (compressed)

Designed for practitioners, not researchers.  Every finding includes the
evidence that supports it (turn numbers, agent names, quoted snippets).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

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

def _dn(agent_id: str) -> str:
    return _AGENT_DISPLAY.get(agent_id, agent_id.replace("-", " ").title())


# ── Output structures ─────────────────────────────────────────────────────────

@dataclass
class RootCauseEntry:
    origin_agent:    str
    turn_number:     int
    assumption_text: str   # clean, ≤ 100 chars
    trigger_type:    str   # "incomplete_data" | "uncertainty" | "inference" | "delegation"
    damage_score:    float
    propagated_to:   List[str]

    def render(self) -> str:
        agents = ", ".join(_dn(a) for a in self.propagated_to[:4])
        more   = f" (+{len(self.propagated_to)-4})" if len(self.propagated_to) > 4 else ""
        return (
            f"  Agent:     {_dn(self.origin_agent)} (turn {self.turn_number})\n"
            f"  Claim:     \"{self.assumption_text}\"\n"
            f"  Trigger:   {self.trigger_type}\n"
            f"  Spread to: {agents}{more}\n"
            f"  Damage:    {self.damage_score:.2f}"
        )


@dataclass
class AgentContribution:
    agent_id:        str
    role:            str   # "stabilizer" | "amplifier" | "neutral" | "mixed"
    introduced:      int   # assumptions introduced
    repairs_made:    int   # corrections to other agents
    adopted_by:      int   # other agents that used this agent's output
    key_action:      str   # most significant single action (one sentence)

    def render(self) -> str:
        role_sym = {"stabilizer": "✓", "amplifier": "✗", "mixed": "±", "neutral": "·"}
        sym = role_sym.get(self.role, " ")
        return (
            f"  {sym} {_dn(self.agent_id):28s}  "
            f"introduced={self.introduced}  repairs={self.repairs_made}  "
            f"adopted_by={self.adopted_by}  [{self.role.upper()}]\n"
            f"      {self.key_action}"
        )


@dataclass
class TimelineEvent:
    turn:        int
    agent_id:    str
    event_type:  str   # "assumption" | "contradiction" | "correction" | "drift" | "recovery" | "tool_failure"
    description: str   # one sentence

    def render(self) -> str:
        sym = {
            "assumption":    "⚑",
            "contradiction": "✗",
            "correction":    "✓",
            "drift":         "↻",
            "recovery":      "↺",
            "tool_failure":  "⚠",
        }.get(self.event_type, "·")
        agent_str = f"[{_dn(self.agent_id)}]" if self.agent_id else ""
        return f"  Turn {self.turn:2d}  {sym}  {agent_str} {self.description}"


@dataclass
class ForensicsReport:
    """Complete trajectory forensics report."""
    # Content
    executive_summary:      str
    root_causes:            List[RootCauseEntry]
    failure_chains:         List[str]          # amplification chain sentences
    agent_contributions:    List[AgentContribution]
    recovery_summary:       str
    unresolved_issues:      List[str]
    timeline:               List[TimelineEvent]

    # Scores (mirror from trajectory evaluation)
    overall_score:          float = 0.0
    assumption_score:       float = 0.0
    collaboration_score:    float = 0.0

    def render(self, width: int = 72) -> str:
        sep   = "─" * width
        thick = "═" * width

        lines = [
            thick,
            "  HARPO TRAJECTORY FORENSICS REPORT",
            thick,
            "",
        ]

        # 1. Executive Summary
        lines += [f"  {'EXECUTIVE SUMMARY':}", ""]
        for sent in self.executive_summary.split(". "):
            sent = sent.strip()
            if sent:
                lines.append(f"  {sent}.")
        lines += ["", sep, ""]

        # 2. Root Cause Analysis
        lines += ["  ROOT CAUSE ANALYSIS", ""]
        if self.root_causes:
            for i, rc in enumerate(self.root_causes, 1):
                lines.append(f"  [{i}]")
                lines.append(rc.render())
                lines.append("")
        else:
            lines.append("  No root cause assumptions identified.")
            lines.append("")
        lines += [sep, ""]

        # 3. Failure Amplification Chains
        lines += ["  FAILURE AMPLIFICATION CHAINS", ""]
        if self.failure_chains:
            for i, chain in enumerate(self.failure_chains, 1):
                lines.append(f"  {i}. {chain}")
        else:
            lines.append("  No amplification chains detected.")
        lines += ["", sep, ""]

        # 4. Agent Contributions
        lines += ["  AGENT CONTRIBUTIONS", ""]
        lines.append(f"  {'Agent':28s}  Introduced  Repairs  Adopted  Role")
        lines.append("  " + "─" * 64)
        for ac in self.agent_contributions:
            lines.append(ac.render())
            lines.append("")
        lines += [sep, ""]

        # 5. Recovery Analysis
        lines += ["  RECOVERY ANALYSIS", ""]
        lines.append(f"  {self.recovery_summary}")
        if self.unresolved_issues:
            lines.append("")
            lines.append("  Unresolved at end of trajectory:")
            for issue in self.unresolved_issues:
                lines.append(f"    • {issue}")
        lines += ["", sep, ""]

        # 6. Timeline
        lines += ["  TRAJECTORY TIMELINE (key events)", ""]
        for ev in self.timeline:
            lines.append(ev.render())
        lines += ["", thick]

        return "\n".join(lines)


# ── Builder ───────────────────────────────────────────────────────────────────

class TrajectoryForensics:
    """
    Builds a ForensicsReport from a trajectory + its semantic analysis results.

    Usage
    -----
        from harpo.reporting.trajectory_forensics import TrajectoryForensics
        from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer

        analysis = SemanticTrajectoryAnalyzer().analyze(traj)
        report   = TrajectoryForensics(traj, analysis).build()
        print(report.render())
    """

    def __init__(self, traj: "AgentTrajectory", analysis) -> None:
        self._traj     = traj
        self._analysis = analysis   # SemanticAnalysis

    def build(self) -> ForensicsReport:
        a = self._analysis

        root_causes      = self._build_root_causes()
        failure_chains   = self._build_failure_chains()
        contributions    = self._build_contributions()
        recovery_summary, unresolved = self._build_recovery()
        timeline         = self._build_timeline()
        exec_summary     = self._build_executive_summary(
            root_causes, failure_chains, contributions, unresolved
        )

        # Pull scores from trajectory evaluation if available
        overall_score = getattr(self._traj, "_last_scores_overall", 0.0)

        return ForensicsReport(
            executive_summary   = exec_summary,
            root_causes         = root_causes,
            failure_chains      = failure_chains,
            agent_contributions = contributions,
            recovery_summary    = recovery_summary,
            unresolved_issues   = unresolved,
            timeline            = timeline,
            overall_score       = overall_score,
        )

    # ── Root Cause Analysis ───────────────────────────────────────────────────

    def _build_root_causes(self) -> List[RootCauseEntry]:
        a   = self._analysis
        out = []

        # Source 1: causal chain summarizer (clean text)
        if getattr(a, "causal_chain_summary", None):
            for s in a.causal_chain_summary.summaries[:5]:
                out.append(RootCauseEntry(
                    origin_agent    = s.origin_agent,
                    turn_number     = 0,   # not in summary; placeholder
                    assumption_text = s.assumption[:100],
                    trigger_type    = "unknown",
                    damage_score    = s.damage_score,
                    propagated_to   = s.affected_agents,
                ))
            return out

        # Source 2: causal_propagation (raw)
        cp = getattr(a, "causal_propagation", None)
        if cp and cp.chains:
            for chain in sorted(cp.chains, key=lambda c: c.damage_score, reverse=True)[:5]:
                # Skip noise chains
                text = chain.assumption_text.strip()
                if not text or text[0] == '#' or '**' in text[:30]:
                    continue
                # Clean text
                text = re.sub(r'[#*`\[\]|>]+', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 15:
                    continue
                out.append(RootCauseEntry(
                    origin_agent    = chain.origin_agent_id,
                    turn_number     = chain.origin_turn,
                    assumption_text = text[:100],
                    trigger_type    = chain.trigger_type,
                    damage_score    = chain.damage_score,
                    propagated_to   = chain.contaminated_agents(),
                ))
        return out

    # ── Failure Chains ────────────────────────────────────────────────────────

    def _build_failure_chains(self) -> List[str]:
        a   = self._analysis
        out = []

        # From causal_chain_summary unresolved items
        css = getattr(a, "causal_chain_summary", None)
        if css and css.unresolved:
            for s in css.unresolved[:3]:
                chain_str = (
                    f"{_dn(s.origin_agent)} introduced \"{s.assumption[:50]}...\" "
                    f"→ spread to {', '.join(_dn(x) for x in s.affected_agents[:3])} "
                    f"→ {s.outcome}"
                )
                out.append(chain_str)

        # From collaboration intelligence
        co = getattr(a, "collaboration", None)
        if co and hasattr(co, "agent_profiles"):
            for aid, prof in co.agent_profiles.items():
                if prof.failure_amplifications > prof.stabilization_events + 1:
                    out.append(
                        f"{_dn(aid)} amplified trajectory failures "
                        f"({prof.failure_amplifications} amplification events, "
                        f"{prof.stabilization_events} stabilizations)."
                    )

        # From drift
        dr = getattr(a, "drift_v2", None) or getattr(a, "drift", None)
        if dr and hasattr(dr, "harmful_events"):
            for ev in dr.harmful_events()[:2]:
                if hasattr(ev, "pressure_tokens") and ev.pressure_tokens:
                    out.append(
                        f"{_dn(ev.agent_id)} drifted from mission objective at turn "
                        f"{ev.turn_detected} (pressure: "
                        f"{', '.join(sorted(ev.pressure_tokens)[:3])})."
                    )

        return out[:5]

    # ── Agent Contributions ───────────────────────────────────────────────────

    def _build_contributions(self) -> List[AgentContribution]:
        a   = self._analysis
        out = []

        co = getattr(a, "collaboration", None)
        cp = getattr(a, "causal_propagation", None)

        # Count assumptions introduced per agent from causal propagation
        agent_introduced: Dict[str, int] = {}
        if cp and cp.chains:
            for chain in cp.chains:
                aid = chain.origin_agent_id
                if aid:
                    agent_introduced[aid] = agent_introduced.get(aid, 0) + 1

        if co and hasattr(co, "agent_profiles"):
            for aid, prof in sorted(co.agent_profiles.items(),
                                    key=lambda x: x[1].contribution_score, reverse=True):
                introduced = agent_introduced.get(aid, 0)
                repairs    = prof.contradiction_repairs
                adopted_by = len(prof.adopted_by)
                stab       = prof.stabilization_events
                amp        = prof.failure_amplifications

                # Determine role
                if repairs >= 2 and stab >= amp:
                    role = "stabilizer"
                elif amp > stab + 1 and repairs == 0:
                    role = "amplifier"
                elif repairs >= 1 and amp >= 1:
                    role = "mixed"
                else:
                    role = "neutral"

                # Key action (one sentence summary)
                key_action = self._key_action_for(aid, role, introduced, repairs, adopted_by)

                out.append(AgentContribution(
                    agent_id     = aid,
                    role         = role,
                    introduced   = introduced,
                    repairs_made = repairs,
                    adopted_by   = adopted_by,
                    key_action   = key_action,
                ))
        else:
            # No collaboration data — build minimal entries from trajectory
            agent_ids = {getattr(s, "agent_id", "") for s in self._traj.steps} - {""}
            for aid in sorted(agent_ids):
                introduced = agent_introduced.get(aid, 0)
                out.append(AgentContribution(
                    agent_id     = aid,
                    role         = "neutral",
                    introduced   = introduced,
                    repairs_made = 0,
                    adopted_by   = 0,
                    key_action   = f"{_dn(aid)} contributed to the investigation.",
                ))
        return out

    def _key_action_for(self, aid: str, role: str,
                         introduced: int, repairs: int, adopted_by: int) -> str:
        dn = _dn(aid)
        if role == "stabilizer":
            return (f"{dn} stabilized the trajectory with {repairs} correction(s); "
                    f"output adopted by {adopted_by} other agent(s).")
        if role == "amplifier":
            return (f"{dn} introduced {introduced} assumption(s) that propagated "
                    f"without correction.")
        if role == "mixed":
            return (f"{dn} both introduced {introduced} assumption(s) and made "
                    f"{repairs} correction(s).")
        return f"{dn} participated without notable amplification or stabilization."

    # ── Recovery Analysis ─────────────────────────────────────────────────────

    def _build_recovery(self) -> Tuple[str, List[str]]:
        a = self._analysis

        resolved   = []
        unresolved = []

        # From causal chain summaries
        css = getattr(a, "causal_chain_summary", None)
        if css:
            for s in css.summaries:
                if s.was_corrected:
                    resolved.append(s.assumption[:60])
                else:
                    unresolved.append(s.assumption[:60])

        # From reflection impact v2
        ri = getattr(a, "reflection_impact_v2", None) or getattr(a, "reflection_impact", None)
        if ri and hasattr(ri, "recovery_count"):
            rec_count = getattr(ri, "recovery_count", 0)
        else:
            rec_count = 0

        # From drift
        dr = getattr(a, "drift_v2", None) or getattr(a, "drift", None)
        if dr and hasattr(dr, "harmful_events"):
            drift_recovered  = sum(1 for e in dr.harmful_events() if e.recovery_detected)
            drift_unresolved = sum(1 for e in dr.harmful_events() if not e.recovery_detected)
            if drift_unresolved:
                unresolved.append(f"{drift_unresolved} objective drift event(s) unrecovered")

        if not resolved and not unresolved and rec_count == 0:
            return "No recovery events detected.", []

        n_resolved   = len(resolved)
        n_unresolved = len(unresolved)
        summary = (
            f"{n_resolved} assumption chain(s) resolved, "
            f"{n_unresolved} unresolved. "
            + (f"{rec_count} recovery reflection(s) contributed to stabilization. "
               if rec_count else "")
        )
        return summary.strip(), unresolved[:5]

    # ── Timeline ──────────────────────────────────────────────────────────────

    def _build_timeline(self) -> List[TimelineEvent]:
        a      = self._analysis
        events = []
        seen   = set()

        def _add(turn: int, agent_id: str, etype: str, desc: str) -> None:
            key = (turn, etype, desc[:30])
            if key not in seen:
                seen.add(key)
                events.append(TimelineEvent(turn, agent_id, etype, desc))

        # Assumptions from causal chain summary or raw propagation
        css = getattr(a, "causal_chain_summary", None)
        cp  = getattr(a, "causal_propagation", None)

        if css and css.summaries:
            for s in css.summaries[:6]:
                _add(0, s.origin_agent, "assumption",
                     f"{_dn(s.origin_agent)}: \"{s.assumption[:60]}\"")
                if s.correction_agent:
                    _add(0, s.correction_agent, "correction",
                         f"{_dn(s.correction_agent)} corrected {_dn(s.origin_agent)}'s claim.")
        elif cp and cp.chains:
            for chain in sorted(cp.chains, key=lambda c: c.origin_turn)[:6]:
                text = re.sub(r'[#*`\[\]|>]+', ' ', chain.assumption_text)
                text = re.sub(r'\s+', ' ', text).strip()[:60]
                if len(text) >= 15:
                    _add(chain.origin_turn, chain.origin_agent_id, "assumption",
                         f"{_dn(chain.origin_agent_id)}: \"{text}\"")

        # Contradictions
        cont = getattr(a, "contradictions", None)
        if cont and cont.contradictions:
            for ev in cont.contradictions[:4]:
                _add(ev.turn_b, "", "contradiction",
                     f"Contradiction detected (turn {ev.turn_a} vs {ev.turn_b}, kind={ev.kind})")

        # Drift events
        dr = getattr(a, "drift_v2", None) or getattr(a, "drift", None)
        if dr and hasattr(dr, "harmful_events"):
            for ev in dr.harmful_events()[:3]:
                _add(ev.turn_detected, ev.agent_id, "drift",
                     f"{_dn(ev.agent_id)} drifted from mission objective "
                     f"(pressure: {', '.join(sorted(ev.pressure_tokens)[:2]) or 'N/A'})")

        # Reflection impact events
        ri = getattr(a, "reflection_impact_v2", None) or getattr(a, "reflection_impact", None)
        if ri and hasattr(ri, "impacts"):
            for imp in ri.impacts[:5]:
                if getattr(imp, "impact_type", "") in ("corrective", "recovery"):
                    aid = getattr(imp, "agent_id", "")
                    _add(imp.reflection_turn, aid, "correction",
                         f"{_dn(aid) if aid else 'Agent'} made {imp.impact_type} reflection "
                         f"(score={getattr(imp, 'impact_score', 0.0):.2f})")

        # Tool failures
        from harpo.trajectory.schema import StepType, StepOutcome
        for step in self._traj.steps:
            if (step.step_type == StepType.TOOL_CALL and
                    step.outcome == StepOutcome.FAILURE):
                tool_name = step.tool_call.name if step.tool_call else "unknown"
                _add(step.turn_number, getattr(step, "agent_id", ""),
                     "tool_failure", f"Tool failure: {tool_name}")

        events.sort(key=lambda e: e.turn)
        return events

    # ── Executive Summary ─────────────────────────────────────────────────────

    def _build_executive_summary(
        self,
        root_causes: List[RootCauseEntry],
        failure_chains: List[str],
        contributions: List[AgentContribution],
        unresolved: List[str],
    ) -> str:
        parts = []

        # Sentence 1: what happened (root cause)
        if root_causes:
            rc = root_causes[0]
            agents_str = (", ".join(_dn(a) for a in rc.propagated_to[:2])
                          + (f" and {len(rc.propagated_to)-2} others"
                             if len(rc.propagated_to) > 2 else ""))
            parts.append(
                f"The primary failure originated with {_dn(rc.origin_agent)}, "
                f"whose assumption \"{rc.assumption_text[:60]}\" "
                f"propagated to {agents_str}."
            )
        else:
            parts.append("No single root cause identified; trajectory shows distributed degradation.")

        # Sentence 2: amplification and stabilization
        amplifiers  = [c for c in contributions if c.role == "amplifier"]
        stabilizers = [c for c in contributions if c.role == "stabilizer"]
        if amplifiers and stabilizers:
            amp_str  = ", ".join(_dn(c.agent_id) for c in amplifiers[:2])
            stab_str = ", ".join(_dn(c.agent_id) for c in stabilizers[:2])
            parts.append(
                f"Failures were amplified by {amp_str} and partially "
                f"repaired by {stab_str}."
            )
        elif stabilizers:
            stab_str = ", ".join(_dn(c.agent_id) for c in stabilizers[:2])
            parts.append(f"Trajectory was stabilized by {stab_str}.")
        elif amplifiers:
            amp_str = ", ".join(_dn(c.agent_id) for c in amplifiers[:2])
            parts.append(f"No stabilizing agents detected; {amp_str} amplified failures without correction.")

        # Sentence 3: outcome
        if unresolved:
            n = len(unresolved)
            parts.append(
                f"{n} issue(s) remain unresolved at end of trajectory: "
                f"{unresolved[0][:60]}."
            )
        else:
            parts.append("All identified assumption chains were corrected before trajectory end.")

        return " ".join(parts)


# ── Convenience function ──────────────────────────────────────────────────────

def generate_forensics_report(traj: "AgentTrajectory",
                               analysis=None) -> ForensicsReport:
    """
    Generate a ForensicsReport for *traj*.

    If *analysis* is not provided, runs SemanticTrajectoryAnalyzer.
    """
    if analysis is None:
        try:
            from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
            analysis = SemanticTrajectoryAnalyzer(run_causal=True).analyze(traj)
        except Exception as exc:
            raise RuntimeError(f"Semantic analysis failed: {exc}") from exc
    return TrajectoryForensics(traj, analysis).build()
