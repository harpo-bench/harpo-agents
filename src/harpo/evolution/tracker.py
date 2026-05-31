"""
HARPO Evolution Tracker

Compares agent trajectories across self-evolution cycles (v1 → v2 → v3 → ...).
Framework-agnostic: accepts trajectories from any adapter or log reader.

Usage
-----
from harpo.evolution.tracker import EvolutionTracker

tracker = EvolutionTracker()
tracker.add_cycle("v1", plugin_v1)   # HarpoPlugin or AgentTrajectory
tracker.add_cycle("v2", plugin_v2)
tracker.add_cycle("v3", plugin_v3)

print(tracker.improvement_summary())
regressions = tracker.detect_regressions(threshold=0.05)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from harpo.failures.schema import TrajectoryRegression
from harpo.trajectory.pipeline import TrajectoryEvaluator
from harpo.core.schema import AgentTrajectory, TrajectoryScores
from harpo.evolution.comparator import TrajectoryComparator, TrajectoryDiff

if TYPE_CHECKING:
    pass


@dataclass
class CycleComparison:
    """Score comparison between two consecutive evolution cycles."""
    from_label:       str
    to_label:         str
    dimension_deltas: Dict[str, float]   # positive = improved
    overall_delta:    float
    regressions:      List[str]          # dims that dropped > threshold
    improvements:     List[str]          # dims that rose > threshold
    diff:             Optional[TrajectoryDiff] = None


@dataclass
class RegressionAlert:
    """Raised when a dimension regresses beyond the threshold."""
    agent_id:   str
    from_label: str
    to_label:   str
    dimension:  str
    delta:      float
    severity:   str   # "minor" | "major" | "critical"


def _severity(delta: float) -> str:
    if delta < -0.15:
        return "critical"
    if delta < -0.08:
        return "major"
    return "minor"


class EvolutionTracker:
    """
    Track agent improvement across self-evolution cycles.

    Each cycle is one HarpoPlugin (live) or AgentTrajectory (post-hoc).
    Comparisons are always consecutive: v1→v2, v2→v3, etc.
    """

    def __init__(
        self,
        evaluator:            Optional[TrajectoryEvaluator] = None,
        regression_threshold: float = 0.05,
    ) -> None:
        self._evaluator   = evaluator or TrajectoryEvaluator()
        self._threshold   = regression_threshold
        self._comparator  = TrajectoryComparator(self._evaluator, regression_threshold)

        # Ordered list of (label, trajectory) pairs
        self._cycles: List[tuple[str, AgentTrajectory]] = []
        self._scores:  Dict[str, TrajectoryScores] = {}

    # ── Data ingestion ───────────────────────────────────────────

    def add_cycle(
        self,
        label:      str,
        source:     Union[AgentTrajectory, Any],  # AgentTrajectory | HarpoPlugin
    ) -> None:
        """Add an evolution cycle by label. Accepts AgentTrajectory or HarpoPlugin."""
        if hasattr(source, "trajectory"):
            traj = source.trajectory()       # HarpoPlugin
        else:
            traj = source                    # AgentTrajectory

        scores = self._evaluator.evaluate(traj)
        self._cycles.append((label, traj))
        self._scores[label] = scores

    # ── Analysis ─────────────────────────────────────────────────

    def compare_all(self) -> List[CycleComparison]:
        """Compare consecutive cycles: v1→v2, v2→v3, ..."""
        results: List[CycleComparison] = []
        for i in range(1, len(self._cycles)):
            from_label, from_traj = self._cycles[i - 1]
            to_label,   to_traj   = self._cycles[i]
            diff = self._comparator.compare(from_traj, to_traj)
            results.append(CycleComparison(
                from_label       = from_label,
                to_label         = to_label,
                dimension_deltas = diff.summary,
                overall_delta    = diff.overall_delta,
                regressions      = diff.regressions,
                improvements     = diff.improvements,
                diff             = diff,
            ))
        return results

    def improvement_summary(self) -> Dict[str, float]:
        """
        Return per-dimension average delta across all consecutive comparisons.
        Positive = net improvement, negative = net regression.
        """
        comparisons = self.compare_all()
        if not comparisons:
            return {}

        all_dims: Dict[str, List[float]] = {}
        for comp in comparisons:
            for dim, delta in comp.dimension_deltas.items():
                all_dims.setdefault(dim, []).append(delta)

        return {
            dim: round(sum(deltas) / len(deltas), 4)
            for dim, deltas in all_dims.items()
        }

    def detect_regressions(
        self,
        threshold: Optional[float] = None,
    ) -> List[RegressionAlert]:
        """Return RegressionAlerts for all dimension drops exceeding threshold."""
        t = threshold if threshold is not None else self._threshold
        alerts: List[RegressionAlert] = []
        agent_id = self._cycles[0][1].agent_id if self._cycles else "unknown"

        for comp in self.compare_all():
            for dim, delta in comp.dimension_deltas.items():
                if delta < -t:
                    alerts.append(RegressionAlert(
                        agent_id   = agent_id,
                        from_label = comp.from_label,
                        to_label   = comp.to_label,
                        dimension  = dim,
                        delta      = round(delta, 4),
                        severity   = _severity(delta),
                    ))
        return alerts

    def to_regression_schema(self) -> List[TrajectoryRegression]:
        """Convert RegressionAlerts to failures.schema.TrajectoryRegression objects."""
        results: List[TrajectoryRegression] = []
        agent_id = self._cycles[0][1].agent_id if self._cycles else "unknown"
        for comp in self.compare_all():
            if comp.regressions:
                results.append(TrajectoryRegression(
                    agent_id       = agent_id,
                    from_label     = comp.from_label,
                    to_label       = comp.to_label,
                    regressed_dims = comp.regressions,
                    delta_per_dim  = {d: comp.dimension_deltas[d]
                                      for d in comp.regressions},
                    severity       = _severity(
                        min(comp.dimension_deltas[d] for d in comp.regressions)
                    ),
                ))
        return results

    def scores_table(self) -> List[Dict[str, Any]]:
        """Return a list of {cycle, dimension: score, ...} rows for tabular display."""
        rows = []
        for label, _ in self._cycles:
            s = self._scores.get(label)
            if not s:
                continue
            row: Dict[str, Any] = {"cycle": label, "overall": round(s.overall, 4)}
            for dim in [
                "reasoning_stability", "conversational_drift", "memory_utility",
                "assumption_accumulation", "recovery_ability", "collaboration_quality",
                "reflection_usefulness", "long_horizon_reliability",
                "trajectory_coherence", "user_aligned_quality",
            ]:
                ds = getattr(s, dim, None)
                if ds:
                    row[dim] = round(ds.value, 4)
            rows.append(row)
        return rows
