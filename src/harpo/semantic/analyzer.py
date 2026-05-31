"""
SemanticTrajectoryAnalyzer — main entry point for semantic intelligence.

Runs all analyzers in one call and returns a SemanticAnalysis with a
structured summary. Safe to call from HookRegistry post-trajectory hooks.

Analyzer layers
---------------
  Core (P1 baseline):
    contradictions  — detect_contradictions
    assumptions     — analyze_assumption_propagation
    reflections     — analyze_reflection_effectiveness
    coherence       — score_semantic_coherence

  Causal intelligence (P1-P4):
    causal_propagation   — analyze_causal_propagation   (P1)
    drift                — analyze_drift                 (P2)
    memory_causality     — analyze_memory_causality      (P3)
    reflection_impact    — analyze_reflection_impact     (P4)

  Collaboration (P5):
    collaboration        — analyze_collaboration

No external dependencies; all heuristic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from .contradiction  import ContradictionResult,           detect_contradictions
from .assumptions    import AssumptionPropagationResult,   analyze_assumption_propagation
from .reflection     import ReflectionResult,              analyze_reflection_effectiveness
from .coherence      import CoherenceResult,               score_semantic_coherence

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory


def _safe(fn, *args, **kwargs):
    """Call fn(*args) and return None on any exception."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        return None


@dataclass
class SemanticAnalysis:
    """All semantic analysis results for one trajectory."""
    contradictions:      ContradictionResult
    assumptions:         AssumptionPropagationResult
    reflections:         ReflectionResult
    coherence:           CoherenceResult

    # Causal intelligence layer (may be None if import fails)
    causal_propagation:  object = None   # CausalPropagationReport
    drift:               object = None   # DriftReport
    memory_causality:    object = None   # MemoryCausalReport
    reflection_impact:   object = None   # ReflectionImpactReport
    collaboration:       object = None   # CollaborationIntelligenceReport

    # ── Legacy summary ─────────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Flat dict suitable for logging or display."""
        base = {
            "contradictions": {
                "total":          self.contradictions.total,
                "reversal_count": self.contradictions.reversal_count,
                "flip_count":     self.contradictions.flip_count,
                "severity":       round(self.contradictions.severity(), 4),
            },
            "assumptions": {
                "total":               self.assumptions.total_assumptions,
                "propagating":         self.assumptions.propagating_count,
                "reinforced":          self.assumptions.reinforced_count,
                "max_radius_turns":    self.assumptions.max_radius,
                "propagation_density": round(self.assumptions.propagation_density(), 4),
            },
            "reflections": {
                "total":               self.reflections.total,
                "effective":           self.reflections.effective_count,
                "action_oriented":     self.reflections.action_oriented_count,
                "effectiveness_rate":  round(self.reflections.effectiveness_rate(), 4),
                "avg_behavior_change": self.reflections.avg_behavior_change,
            },
            "coherence": {
                "overall":           self.coherence.overall_coherence,
                "avg_core_overlap":  self.coherence.avg_core_overlap,
                "drift_events":      self.coherence.drift_events,
                "return_events":     self.coherence.return_events,
            },
        }

        # Append causal layer summaries when available
        if self.causal_propagation is not None:
            cp = self.causal_propagation
            base["causal_propagation"] = {
                "total_assumptions":  cp.total_assumptions,
                "uncorrected":        cp.uncorrected_count,
                "high_damage":        cp.high_damage_count,
                "cascade_detected":   cp.cascade_detected,
                "summary":            cp.summary_narrative,
            }

        if self.drift is not None:
            dr = self.drift
            base["drift"] = {
                "total_events":      len(dr.events),
                "objective_drift":   dr.objective_drift_detected,
                "onset_turn":        dr.drift_onset_turn,
                "recovery_rate":     dr.recovery_rate,
                "overall_score":     dr.overall_drift_score,
                "summary":           dr.narrative(),
            }

        if self.memory_causality is not None:
            mc = self.memory_causality
            base["memory_causality"] = mc.as_dict() if hasattr(mc, "as_dict") else {}

        if self.reflection_impact is not None:
            ri = self.reflection_impact
            base["reflection_impact"] = ri.as_dict() if hasattr(ri, "as_dict") else {}

        if self.collaboration is not None:
            co = self.collaboration
            base["collaboration"] = co.as_dict() if hasattr(co, "as_dict") else {}

        return base

    # ── Flags ──────────────────────────────────────────────────────────────────

    def flags(self) -> list[str]:
        """Human-readable diagnostic flags for this trajectory."""
        out = []
        c = self.contradictions
        a = self.assumptions
        r = self.reflections
        s = self.coherence

        if c.flip_count > 0:
            out.append(f"CONTRADICTION: {c.flip_count} silent plan/fact flip(s) detected")
        if c.reversal_count > 1:
            out.append(f"SELF-CORRECTION: {c.reversal_count} explicit reversal(s) in reasoning")
        if a.propagating_count > 0:
            out.append(
                f"ASSUMPTION_PROPAGATION: {a.propagating_count}/{a.total_assumptions} "
                f"assumption(s) spread into later turns (max radius {a.max_radius} turns)"
            )
        if a.reinforced_count > 0:
            out.append(f"ASSUMPTION_REINFORCEMENT: {a.reinforced_count} assumption(s) echoed/restated")
        if r.total > 0 and r.ineffective_count > r.effective_count:
            out.append(
                f"REFLECTION_IGNORED: {r.ineffective_count}/{r.total} reflection(s) "
                f"produced no measurable reasoning change"
            )
        if s.drift_events > 0:
            ret_note = f", {s.return_events} return(s)" if s.return_events else ", no returns"
            out.append(f"TOPIC_DRIFT: {s.drift_events} drift event(s){ret_note}")

        # Causal flags
        cp = self.causal_propagation
        if cp is not None:
            if cp.cascade_detected:
                out.append("ASSUMPTION_CASCADE: mutually-reinforcing uncorrected assumptions linked to failures")
            if cp.high_damage_count > 0:
                out.append(f"HIGH_DAMAGE_ASSUMPTIONS: {cp.high_damage_count} assumption(s) with damage_score > 0.5")

        # Prefer v2 drift flags (calibrated, fewer false positives)
        dr2 = getattr(self, "drift_v2", None)
        if dr2 is not None:
            if dr2.objective_drift_count > 0:
                agents = ", ".join(dr2.drift_agents[:3])
                out.append(f"OBJECTIVE_DRIFT: {dr2.objective_drift_count} event(s) — {agents}")
            if dr2.attention_collapse_count > 0:
                out.append(f"ATTENTION_COLLAPSE: {dr2.attention_collapse_count} critical entity collapse(s)")
            if dr2.topic_evolution_count > 0:
                out.append(f"TOPIC_EVOLUTION: {dr2.topic_evolution_count} healthy specialization event(s) (not penalised)")
        elif self.drift is not None:
            dr = self.drift
            if dr.objective_drift_detected:
                onset = f" (onset turn {dr.drift_onset_turn})" if dr.drift_onset_turn else ""
                out.append(f"OBJECTIVE_DRIFT: agent deviated from core task objective{onset}")
            if dr.attention_collapse_turns:
                out.append(f"ATTENTION_COLLAPSE: key entities absent for {len(dr.attention_collapse_turns)} window(s)")

        ri2 = getattr(self, "reflection_impact_v2", None)
        if ri2 is not None and ri2.impacts:
            if ri2.ineffective_count > 0:
                out.append(
                    f"REFLECTION_INEFFECTIVE: {ri2.ineffective_count} reflection(s) had no downstream effect"
                )
            if ri2.stylistic_count > ri2.impactful_count:
                out.append(
                    f"REFLECTION_STYLISTIC: {ri2.stylistic_count} reflection(s) changed wording "
                    f"without fixing underlying problems"
                )
        elif self.reflection_impact is not None and self.reflection_impact.impacts:
            ri = self.reflection_impact
            null_frac = ri.null_count / len(ri.impacts)
            if null_frac >= 0.5:
                out.append(
                    f"REFLECTION_NULL: {ri.null_count}/{len(ri.impacts)} reflections "
                    f"produced no structural improvement"
                )

        mc = self.memory_causality
        if mc is not None and hasattr(mc, "net_causality"):
            if mc.net_causality == "harmful":
                out.append(
                    f"MEMORY_HARMFUL: {mc.reinforcement_count} reinforcement + "
                    f"{mc.stale_reuse_count} stale-reuse event(s)"
                )

        co = self.collaboration
        if co is not None and hasattr(co, "most_siloed_agent") and co.most_siloed_agent:
            out.append(f"AGENT_SILOED: {co.most_siloed_agent} contributed in isolation")

        return out

    # ── Causal narrative ───────────────────────────────────────────────────────

    def causal_narrative(self) -> str:
        """
        One-paragraph narrative answering WHY, WHICH, HOW, WHETHER.
        Uses v2 modules when available, falls back to v1.
        """
        lines = []

        # Assumption causality — prefer clean chain summary
        css = getattr(self, "causal_chain_summary", None)
        if css is not None and css.summaries:
            lines.append(f"ASSUMPTION CAUSALITY: {css.executive_summary}")
            for s in css.summaries[:3]:
                lines.append(f"  • {s.one_line()}")
        elif self.causal_propagation is not None and self.causal_propagation.total_assumptions > 0:
            cp = self.causal_propagation
            lines.append(f"ASSUMPTION CAUSALITY: {cp.summary_narrative}")

        # Drift — prefer v2 (calibrated)
        dr2 = getattr(self, "drift_v2", None)
        if dr2 is not None:
            lines.append(f"DRIFT (calibrated): {dr2.narrative()}")
            for ev in dr2.harmful_events()[:3]:
                lines.append(f"  • {ev.narrative()}")
            if dr2.false_positive_filter:
                lines.append(
                    f"  [{dr2.false_positive_filter} benign topic-evolution events "
                    f"filtered from v1 report]"
                )

        # Memory causality
        mc = self.memory_causality
        if mc is not None and hasattr(mc, "narrative"):
            lines.append(f"MEMORY: {mc.narrative()}")
            harmful = [e for e in mc.events if e.is_harmful()][:3]
            for e in harmful:
                lines.append(f"  ⚠ Turn {e.turn_number} [{e.operation}]: {e.impact_description}")
            beneficial = [e for e in mc.events if e.is_beneficial()][:2]
            for e in beneficial:
                lines.append(f"  ✓ Turn {e.turn_number} [{e.operation}]: {e.impact_description}")

        # Reflection impact — prefer v2
        ri2 = getattr(self, "reflection_impact_v2", None)
        if ri2 is not None and ri2.impacts:
            lines.append(f"REFLECTION IMPACT: {ri2.narrative()}")
            for imp in ri2.impacts[:4]:
                lines.append(f"  • {imp.narrative()}")
        elif self.reflection_impact is not None and self.reflection_impact.impacts:
            ri = self.reflection_impact
            lines.append(f"REFLECTION IMPACT: {ri.narrative()}")

        # Collaboration
        co = self.collaboration
        if co is not None and hasattr(co, "narrative"):
            lines.append(f"COLLABORATION: {co.narrative()}")

        if not lines:
            return "No causal signals detected — trajectory appears stable."
        return "\n".join(lines)


class SemanticTrajectoryAnalyzer:
    """
    Orchestrates all semantic analyzers.

    Usage:
        analyzer = SemanticTrajectoryAnalyzer()
        result   = analyzer.analyze(trajectory)
        print(result.causal_narrative())
        print(result.flags())

    To integrate with HookRegistry:
        hooks.register_post_trajectory(
            lambda ctx: setattr(ctx.trajectory, '_semantic', analyzer.analyze(ctx.trajectory))
        )
    """

    def __init__(self, run_causal: bool = True) -> None:
        """
        run_causal: if True (default), runs all causal intelligence analyzers
                    in addition to the core four.  Set to False for fast
                    benchmarks where causal depth is not needed.
        """
        self.run_causal = run_causal

    def analyze(self, traj: "AgentTrajectory") -> "SemanticAnalysis":
        # Core four — always run
        contradictions = detect_contradictions(traj)
        assumptions    = analyze_assumption_propagation(traj)
        reflections    = analyze_reflection_effectiveness(traj)
        coherence      = score_semantic_coherence(traj)

        if not self.run_causal:
            return SemanticAnalysis(
                contradictions = contradictions,
                assumptions    = assumptions,
                reflections    = reflections,
                coherence      = coherence,
            )

        # Causal intelligence layer (P1-P5)
        from .causal_propagation         import analyze_causal_propagation
        from .drift_analysis             import analyze_drift
        from .objective_drift_v2         import analyze_drift_v2
        from .reflection_impact          import analyze_reflection_impact
        from .reflection_impact_v2       import analyze_reflection_impact_v2
        from .collaboration_intelligence import analyze_collaboration
        from .causal_chain_summarizer    import summarize_causal_chains

        causal_prop  = _safe(analyze_causal_propagation, traj)
        drift        = _safe(analyze_drift, traj)
        drift_v2     = _safe(analyze_drift_v2, traj)
        refl_impact  = _safe(analyze_reflection_impact, traj)
        refl_v2      = _safe(analyze_reflection_impact_v2, traj)
        collab       = _safe(analyze_collaboration, traj)

        # Causal chain summary (clean text, executive-readable)
        causal_chain_summary = None
        if causal_prop is not None:
            causal_chain_summary = _safe(summarize_causal_chains, causal_prop)

        # Memory causality (with instrumentation fallback)
        mem_causality: Optional[object] = None
        try:
            from harpo.memory.causal_memory import analyze_memory_causality
            mem_causality = analyze_memory_causality(traj)
        except Exception:
            pass

        result = SemanticAnalysis(
            contradictions       = contradictions,
            assumptions          = assumptions,
            reflections          = reflections,
            coherence            = coherence,
            causal_propagation   = causal_prop,
            drift                = drift,
            memory_causality     = mem_causality,
            reflection_impact    = refl_impact,
            collaboration        = collab,
        )
        # Attach v2 fields directly (avoids breaking existing consumers)
        result.drift_v2              = drift_v2              # type: ignore[attr-defined]
        result.reflection_impact_v2  = refl_v2               # type: ignore[attr-defined]
        result.causal_chain_summary  = causal_chain_summary  # type: ignore[attr-defined]
        return result
