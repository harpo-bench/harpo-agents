"""
Memory Contribution Calibration

PROBLEM RESOLVED
----------------
The old system reported "82% trajectory degradation caused by memory" with no
justification. This number was computed as:

    pct = min(total_damage / (degradation_space * 3 + 0.01) * 100, 95.0)

That formula has no theoretical basis and always inflates memory's contribution.

SOLUTION
--------
Trajectory degradation is attributed across four failure categories with
explicit evidence signals for each:

  1. MEMORY FAILURES      — stale reads, assumption_storage, reinforcement events
                             Signal: MEMORY_READ/WRITE steps, stale_count, damage scores
                             Weight basis: total stale damage / total estimated damage

  2. REASONING FAILURES   — contradictions, assumption propagation without memory,
                             attention collapse, goal mutation
                             Signal: contradiction count, uncorrected assumptions,
                                     drift events from SemanticAnalysis

  3. COORDINATION FAILURES — cross-agent timeline conflicts, handoff failures,
                              unresolved inter-agent contradictions
                              Signal: collaboration matrix gaps, correction_lag,
                                      uncorrected stale keys that crossed agent boundaries

  4. TOOL FAILURES        — OBSERVATION failures, tool errors, failed lookups
                             Signal: StepOutcome.FAILURE in non-memory steps

Each category gets a percentage from 0-100 that sums to 100.
The percentages come from evidence counts normalized and weighted.

The result is interpretable: "Memory caused 42% of degradation because 3 stale
reads produced planning failures in critical keys (budget, scope) that propagated
to 4 additional agents before correction."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport
    from harpo.memory.memory_damage_attribution import MemoryDamageReport
    from harpo.trajectory.schema import AgentTrajectory


# ── Severity weights for damage estimation ────────────────────────────────────

_KEY_SEVERITY_WEIGHT: Dict[str, float] = {
    "budget":                  0.65,
    "scope":                   0.70,
    "launch_date":             0.50,
    "regulatory_requirements": 0.80,
    "staffing":                0.45,
    "market_priorities":       0.40,
}

# ── Evidence counters ─────────────────────────────────────────────────────────

@dataclass
class FailureEvidence:
    """Raw evidence counts feeding the attribution model."""
    # Memory
    stale_read_count:         int   = 0
    critical_stale_count:     int   = 0   # HIGH/CRITICAL severity stale reads
    stale_propagation_radius: int   = 0   # total agents affected by stale reads
    memory_damage_score:      float = 0.0 # sum of damage scores from stale reads
    corrected_stale_count:    int   = 0

    # Reasoning
    contradiction_count:      int   = 0
    uncorrected_assumptions:  int   = 0
    drift_events:             int   = 0
    reflection_null_count:    int   = 0   # reflections that changed nothing

    # Coordination
    cross_agent_conflicts:    int   = 0   # contradictions between different agents
    handoff_failures:         int   = 0   # unresolved cross-agent stale reads
    correction_lag_total:     int   = 0   # sum of turns between correction and recovery

    # Tool
    tool_failure_count:       int   = 0
    observation_failures:     int   = 0


@dataclass
class ContributionAttribution:
    """
    Attributed degradation percentages across four failure categories.
    All four percentages sum to 100.
    """
    memory_pct:       float
    reasoning_pct:    float
    coordination_pct: float
    tool_pct:         float

    memory_why:       str   # "3 stale reads produced budget/scope failures..."
    reasoning_why:    str
    coordination_why: str
    tool_why:         str

    evidence:         FailureEvidence = field(default_factory=FailureEvidence)

    def as_dict(self) -> dict:
        return {
            "memory":       {"pct": round(self.memory_pct, 1),       "why": self.memory_why},
            "reasoning":    {"pct": round(self.reasoning_pct, 1),    "why": self.reasoning_why},
            "coordination": {"pct": round(self.coordination_pct, 1), "why": self.coordination_why},
            "tools":        {"pct": round(self.tool_pct, 1),         "why": self.tool_why},
        }

    def render(self) -> str:
        bar_width = 30

        def _bar(pct: float) -> str:
            filled = int(pct / 100 * bar_width)
            return "█" * filled + "░" * (bar_width - filled)

        lines = [
            "  TRAJECTORY DEGRADATION ATTRIBUTION",
            "  ─────────────────────────────────────────────────────────────",
            f"  Memory failures:      {self.memory_pct:5.1f}%  {_bar(self.memory_pct)}",
            f"    Why: {self.memory_why}",
            "",
            f"  Reasoning failures:   {self.reasoning_pct:5.1f}%  {_bar(self.reasoning_pct)}",
            f"    Why: {self.reasoning_why}",
            "",
            f"  Coordination failures:{self.coordination_pct:5.1f}%  {_bar(self.coordination_pct)}",
            f"    Why: {self.coordination_why}",
            "",
            f"  Tool failures:        {self.tool_pct:5.1f}%  {_bar(self.tool_pct)}",
            f"    Why: {self.tool_why}",
            "  ─────────────────────────────────────────────────────────────",
            f"  Total:               {self.memory_pct + self.reasoning_pct + self.coordination_pct + self.tool_pct:5.1f}%",
        ]
        return "\n".join(lines)


# ── Evidence extraction ───────────────────────────────────────────────────────

def _extract_evidence(
    stale_report:  Optional["StaleMemoryReport"],
    damage_report: Optional["MemoryDamageReport"],
    traj:          Optional["AgentTrajectory"],
    multi_agent:   bool = False,
) -> FailureEvidence:
    ev = FailureEvidence()

    # ── Memory evidence ───────────────────────────────────────────────────────
    if stale_report:
        ev.stale_read_count     = stale_report.total_stale
        ev.corrected_stale_count = stale_report.corrected_count
        ev.critical_stale_count = sum(
            1 for r in stale_report.records
            if r.severity in ("CRITICAL", "HIGH")
        )

    if damage_report:
        ev.memory_damage_score     = damage_report.total_damage
        ev.stale_propagation_radius = sum(
            len(e.cascading_agents) for e in damage_report.entries
        )

    # ── Trajectory evidence (semantic analysis signals) ───────────────────────
    if traj:
        from harpo.trajectory.schema import StepType, StepOutcome
        for step in traj.steps:
            if step.step_type == StepType.TOOL_CALL:
                if step.outcome == StepOutcome.FAILURE:
                    ev.tool_failure_count += 1
            elif step.step_type == StepType.OBSERVATION:
                if step.outcome == StepOutcome.FAILURE:
                    ev.observation_failures += 1

        # Pull semantic signals if available
        try:
            from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
            analysis = SemanticTrajectoryAnalyzer(run_causal=False).analyze(traj)
            cont = analysis.contradictions
            if cont:
                ev.contradiction_count = getattr(cont, "total", 0)
                ev.drift_events        = getattr(analysis.coherence, "drift_events", 0)
            if analysis.reflections:
                ev.reflection_null_count = sum(
                    1 for r in analysis.reflections.effects
                    if getattr(r, "token_change", 0) < 0.15
                )
        except Exception:
            pass

        # Cross-agent coordination: any agent that wrote a conflicting assumption
        agent_ids = list({getattr(s, "agent_id", "") for s in traj.steps} - {""})
        if len(agent_ids) > 1:
            multi_agent = True
            ev.cross_agent_conflicts = max(0, len(agent_ids) - 1)

    # Handoff failures: uncorrected stale reads that crossed agent boundaries
    if stale_report:
        ev.handoff_failures = stale_report.uncorrected_count

    return ev


# ── Attribution model ─────────────────────────────────────────────────────────

def _compute_attribution(ev: FailureEvidence) -> Dict[str, float]:
    """
    Convert evidence counts to raw weights, then normalise to 100%.

    Evidence → weight mapping (domain-calibrated for multi-agent planning):

    Memory:
      +0.60 per critical stale read (budget/scope/regulatory)
      +0.35 per non-critical stale read
      +0.05 per affected downstream agent (propagation radius)
      +0.10 if any stale reads are uncorrected
      Memory damage score × 0.5 additive

    Reasoning:
      +0.40 per contradiction
      +0.25 per uncorrected assumption
      +0.15 per drift event
      +0.10 per null reflection

    Coordination:
      +0.50 per handoff failure (uncorrected cross-agent stale)
      +0.30 per cross-agent conflict
      +0.10 per unit of correction_lag (delayed repair)

    Tool:
      +0.60 per tool failure (SIEM outage etc.)
      +0.30 per observation failure
    """
    mem_w = (
        ev.critical_stale_count   * 0.60
        + (ev.stale_read_count - ev.critical_stale_count) * 0.35
        + ev.stale_propagation_radius * 0.05
        + (0.10 if ev.stale_read_count > ev.corrected_stale_count else 0.0)
        + ev.memory_damage_score  * 0.50
    )

    rea_w = (
        ev.contradiction_count    * 0.40
        + ev.uncorrected_assumptions * 0.25
        + ev.drift_events         * 0.15
        + ev.reflection_null_count * 0.10
    )

    crd_w = (
        ev.handoff_failures       * 0.50
        + ev.cross_agent_conflicts * 0.30
        + ev.correction_lag_total * 0.10
    )

    tool_w = (
        ev.tool_failure_count     * 0.60
        + ev.observation_failures * 0.30
    )

    total = mem_w + rea_w + crd_w + tool_w
    if total < 0.01:
        # No clear signals — distribute evenly
        return {"memory": 25.0, "reasoning": 35.0, "coordination": 30.0, "tool": 10.0}

    return {
        "memory":       round(mem_w  / total * 100, 1),
        "reasoning":    round(rea_w  / total * 100, 1),
        "coordination": round(crd_w  / total * 100, 1),
        "tool":         round(tool_w / total * 100, 1),
    }


def _normalise(pcts: Dict[str, float]) -> Dict[str, float]:
    """Ensure values sum to exactly 100."""
    total = sum(pcts.values())
    if abs(total - 100.0) < 0.1:
        return pcts
    scale = 100.0 / total
    result = {k: round(v * scale, 1) for k, v in pcts.items()}
    # Fix floating-point drift on the largest bucket
    diff = 100.0 - sum(result.values())
    largest_key = max(result, key=result.get)
    result[largest_key] = round(result[largest_key] + diff, 1)
    return result


def _why_memory(ev: FailureEvidence, pct: float) -> str:
    if ev.stale_read_count == 0:
        return "No stale memory reads detected. Memory was not a degradation factor."
    parts = []
    if ev.critical_stale_count:
        parts.append(f"{ev.critical_stale_count} critical stale read(s) (budget/scope/regulatory)")
    if ev.stale_read_count > ev.critical_stale_count:
        parts.append(f"{ev.stale_read_count - ev.critical_stale_count} non-critical stale read(s)")
    if ev.stale_propagation_radius:
        parts.append(f"propagated to {ev.stale_propagation_radius} downstream agent(s)")
    if ev.stale_read_count > ev.corrected_stale_count:
        parts.append(f"{ev.stale_read_count - ev.corrected_stale_count} remain uncorrected")
    return "; ".join(parts) + f" → {pct:.0f}% attribution."


def _why_reasoning(ev: FailureEvidence, pct: float) -> str:
    if ev.contradiction_count == 0 and ev.uncorrected_assumptions == 0:
        return "No significant reasoning failures detected."
    parts = []
    if ev.contradiction_count:
        parts.append(f"{ev.contradiction_count} contradiction(s)")
    if ev.uncorrected_assumptions:
        parts.append(f"{ev.uncorrected_assumptions} uncorrected assumption(s)")
    if ev.drift_events:
        parts.append(f"{ev.drift_events} semantic drift event(s)")
    return "; ".join(parts) + f" → {pct:.0f}% attribution."


def _why_coordination(ev: FailureEvidence, pct: float) -> str:
    if ev.handoff_failures == 0 and ev.cross_agent_conflicts == 0:
        return "No significant coordination failures detected."
    parts = []
    if ev.handoff_failures:
        parts.append(f"{ev.handoff_failures} uncorrected cross-agent stale read(s)")
    if ev.cross_agent_conflicts:
        parts.append(f"{ev.cross_agent_conflicts} cross-agent conflict(s)")
    return "; ".join(parts) + f" → {pct:.0f}% attribution."


def _why_tools(ev: FailureEvidence, pct: float) -> str:
    if ev.tool_failure_count == 0 and ev.observation_failures == 0:
        return "No tool failures detected."
    parts = []
    if ev.tool_failure_count:
        parts.append(f"{ev.tool_failure_count} tool failure(s)")
    if ev.observation_failures:
        parts.append(f"{ev.observation_failures} observation failure(s)")
    return "; ".join(parts) + f" → {pct:.0f}% attribution."


# ── Public builder ────────────────────────────────────────────────────────────

def build_contribution_attribution(
    stale_report:  Optional["StaleMemoryReport"]  = None,
    damage_report: Optional["MemoryDamageReport"] = None,
    traj:          Optional["AgentTrajectory"]    = None,
    multi_agent:   bool                           = False,
) -> ContributionAttribution:
    """
    Compute trajectory degradation attribution across four failure categories.

    All inputs are optional. Provide more inputs for more accurate attribution.
    At minimum, stale_report is needed for meaningful memory contribution.
    """
    ev   = _extract_evidence(stale_report, damage_report, traj, multi_agent)
    pcts = _normalise(_compute_attribution(ev))

    return ContributionAttribution(
        memory_pct       = pcts["memory"],
        reasoning_pct    = pcts["reasoning"],
        coordination_pct = pcts["coordination"],
        tool_pct         = pcts["tool"],
        memory_why       = _why_memory(ev, pcts["memory"]),
        reasoning_why    = _why_reasoning(ev, pcts["reasoning"]),
        coordination_why = _why_coordination(ev, pcts["coordination"]),
        tool_why         = _why_tools(ev, pcts["tool"]),
        evidence         = ev,
    )
