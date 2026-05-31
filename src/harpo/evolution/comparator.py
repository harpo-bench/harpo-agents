"""
HARPO Trajectory Comparator

Side-by-side diff of two trajectories at dimension and step level.
Used by EvolutionTracker and directly by the evaluation API.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from harpo.trajectory.pipeline import TrajectoryEvaluator
from harpo.core.schema import AgentTrajectory, TrajectoryScores


@dataclass
class StepDiff:
    """Dimension-level delta between two trajectory versions at a given turn."""
    turn:      int
    dimension: str    # metric name
    baseline:  float
    candidate: float
    delta:     float  # candidate − baseline (positive = improved)
    signal:    str    # "improved" | "regressed" | "neutral"


@dataclass
class TrajectoryDiff:
    """Full comparison result between a baseline and candidate trajectory."""
    baseline_id:    str
    candidate_id:   str
    step_diffs:     list[StepDiff]
    summary:        dict[str, float]  # {dimension: avg_delta}
    overall_delta:  float
    regressions:    list[str]         # dimension names that dropped > threshold
    improvements:   list[str]         # dimension names that rose > threshold
    regression_threshold: float = 0.05


class TrajectoryComparator:
    """
    Side-by-side diff of two trajectories.

    Evaluation is done per-trajectory-pair (not per-step), because the
    10 metrics operate over the full trajectory.  Step-level diffs are
    synthesized by splitting each trajectory into turn-quartiles and
    scoring each quartile independently.
    """

    def __init__(
        self,
        evaluator:            Optional[TrajectoryEvaluator] = None,
        regression_threshold: float = 0.05,
    ) -> None:
        self._evaluator  = evaluator or TrajectoryEvaluator()
        self._threshold  = regression_threshold

    # ── Public API ──────────────────────────────────────────────

    def compare(
        self,
        baseline:  AgentTrajectory,
        candidate: AgentTrajectory,
    ) -> TrajectoryDiff:
        """Full diff between baseline and candidate trajectories."""
        scores_b = self._evaluator.evaluate(baseline)
        scores_c = self._evaluator.evaluate(candidate)

        dims = self._extract_dims(scores_b, scores_c)
        summary = {d: round(dc - db, 4) for d, (db, dc) in dims.items()}
        regressions  = [d for d, delta in summary.items() if delta < -self._threshold]
        improvements = [d for d, delta in summary.items() if delta >  self._threshold]

        # Synthesize per-quartile step diffs
        step_diffs = self._quartile_diffs(baseline, candidate)

        return TrajectoryDiff(
            baseline_id   = baseline.trajectory_id,
            candidate_id  = candidate.trajectory_id,
            step_diffs    = step_diffs,
            summary       = summary,
            overall_delta = round(scores_c.overall - scores_b.overall, 4),
            regressions   = regressions,
            improvements  = improvements,
        )

    def to_html(self, diff: TrajectoryDiff) -> str:
        """Minimal HTML table for the diff — suitable for dashboard rendering."""
        rows = "".join(
            f"<tr><td>{d}</td>"
            f"<td style='color:{'green' if v>0 else 'red' if v<0 else 'gray'}'>"
            f"{'+' if v>0 else ''}{v:.4f}</td></tr>"
            for d, v in diff.summary.items()
        )
        return (
            f"<table border='1'>"
            f"<tr><th>Dimension</th><th>Delta (candidate − baseline)</th></tr>"
            f"{rows}"
            f"<tr><th>OVERALL</th>"
            f"<th style='color:{'green' if diff.overall_delta>0 else 'red'}'>"
            f"{'+' if diff.overall_delta>0 else ''}{diff.overall_delta:.4f}</th></tr>"
            f"</table>"
        )

    # ── Internal ────────────────────────────────────────────────

    @staticmethod
    def _extract_dims(
        scores_b: TrajectoryScores,
        scores_c: TrajectoryScores,
    ) -> dict[str, tuple[float, float]]:
        """Return {dimension: (baseline_score, candidate_score)}."""
        fields = [
            "reasoning_stability", "conversational_drift", "memory_utility",
            "assumption_accumulation", "recovery_ability", "collaboration_quality",
            "reflection_usefulness", "long_horizon_reliability",
            "trajectory_coherence", "user_aligned_quality",
        ]
        result = {}
        for f in fields:
            ds_b = getattr(scores_b, f, None)
            ds_c = getattr(scores_c, f, None)
            if ds_b and ds_c:
                result[f] = (ds_b.value, ds_c.value)
        return result

    def _quartile_diffs(
        self,
        baseline:  AgentTrajectory,
        candidate: AgentTrajectory,
    ) -> list[StepDiff]:
        """Score each trajectory's turn-quartile independently and diff."""
        diffs: list[StepDiff] = []
        b_turns = baseline.turns()
        c_turns = candidate.turns()
        n = min(len(b_turns), len(c_turns), 4)
        if n < 2:
            return diffs

        chunk = max(1, n // 4) if n >= 4 else 1
        quartiles = [(i * chunk, min((i + 1) * chunk, n)) for i in range(4)]

        for q_idx, (start, end) in enumerate(quartiles):
            b_slice = self._slice_trajectory(baseline, b_turns[start:end])
            c_slice = self._slice_trajectory(candidate, c_turns[start:end])
            if not b_slice.steps or not c_slice.steps:
                continue
            sb = self._evaluator.evaluate(b_slice)
            sc = self._evaluator.evaluate(c_slice)
            for dim, (db, dc) in self._extract_dims(sb, sc).items():
                delta = round(dc - db, 4)
                signal = ("improved" if delta > 0.02 else
                          "regressed" if delta < -0.02 else "neutral")
                diffs.append(StepDiff(
                    turn      = start,
                    dimension = dim,
                    baseline  = round(db, 4),
                    candidate = round(dc, 4),
                    delta     = delta,
                    signal    = signal,
                ))
        return diffs

    @staticmethod
    def _slice_trajectory(
        source: AgentTrajectory,
        turns:  list[list[Any]],
    ) -> AgentTrajectory:
        """Build a lightweight sub-trajectory from a turn slice."""
        from harpo.core.schema import AgentTrajectory as AT
        sliced = AT(
            trajectory_id = source.trajectory_id + "_slice",
            agent_id      = source.agent_id,
            user_intent   = source.user_intent,
        )
        for turn_steps in turns:
            for step in turn_steps:
                sliced.add_step(step)
        return sliced
