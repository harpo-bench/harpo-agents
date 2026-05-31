"""
HARPO-Open Evaluation Pipeline

End-to-end trajectory evaluation.  Feed in an AgentTrajectory,
get back a fully populated TrajectoryScores + FailureReport.

Analogous to HARPOMTv2Evaluator but for behavioral trajectories
rather than recommendation outputs.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from .schema import (
    AgentTrajectory, DimensionScore, TrajectoryScores,
    FailureReport, TrajectoryComparison,
)
from .metrics import (
    score_reasoning_stability,
    score_conversational_drift,
    score_memory_utility,
    score_assumption_accumulation,
    score_recovery_ability,
    score_collaboration_quality,
    score_reflection_usefulness,
    score_long_horizon_reliability,
    score_trajectory_coherence,
    score_user_aligned_quality,
    detect_failure_modes,
)


# ────────────────────────────────────────────────────────────────
# Weights (sum = 1.0) — tunable per deployment
# ────────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS: Dict[str, float] = {
    "reasoning_stability":      0.12,
    "conversational_drift":     0.10,
    "memory_utility":           0.10,
    "assumption_accumulation":  0.08,
    "recovery_ability":         0.12,
    "collaboration_quality":    0.08,
    "reflection_usefulness":    0.08,
    "long_horizon_reliability": 0.12,
    "trajectory_coherence":     0.10,
    "user_aligned_quality":     0.10,
}


class TrajectoryEvaluator:
    """
    Primary evaluation entry point for HARPO-Open.

    Usage
    -----
    evaluator = TrajectoryEvaluator()
    scores = evaluator.evaluate(trajectory)
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        cache_dir: str = "./eval_cache/trajectory",
    ):
        self.weights = weights or DEFAULT_WEIGHTS
        self.cache_dir = cache_dir

    def evaluate(
        self,
        traj: AgentTrajectory,
        task_success: Optional[float] = None,
        use_cache: bool = True,
    ) -> TrajectoryScores:
        """
        Run all 10 behavioral metrics and produce a TrajectoryScores.

        Parameters
        ----------
        traj : AgentTrajectory
        task_success : optional override for task-level binary success
        use_cache : whether to cache results by trajectory_id
        """
        cache_path = os.path.join(self.cache_dir, f"{traj.trajectory_id}.json")
        if use_cache and os.path.exists(cache_path):
            with open(cache_path) as f:
                raw = json.load(f)
            return _dict_to_scores(raw)

        t0 = time.perf_counter()

        scores = TrajectoryScores(
            reasoning_stability=       score_reasoning_stability(traj),
            conversational_drift=      score_conversational_drift(traj),
            memory_utility=            score_memory_utility(traj),
            assumption_accumulation=   score_assumption_accumulation(traj),
            recovery_ability=          score_recovery_ability(traj),
            collaboration_quality=     score_collaboration_quality(traj),
            reflection_usefulness=     score_reflection_usefulness(traj),
            long_horizon_reliability=  score_long_horizon_reliability(traj),
            trajectory_coherence=      score_trajectory_coherence(traj),
            user_aligned_quality=      score_user_aligned_quality(traj),
        )
        scores.task_success = task_success
        scores.overall = self._compute_overall(scores)

        traj.scores = scores
        traj.failure_report = detect_failure_modes(traj)

        elapsed = time.perf_counter() - t0

        if use_cache:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(_scores_to_dict(scores), f, indent=2)

        return scores

    def evaluate_batch(
        self,
        trajectories: List[AgentTrajectory],
        task_successes: Optional[List[float]] = None,
    ) -> List[TrajectoryScores]:
        """Evaluate a list of trajectories."""
        if task_successes is None:
            task_successes = [None] * len(trajectories)
        return [
            self.evaluate(t, ts)
            for t, ts in zip(trajectories, task_successes)
        ]

    def compare(
        self,
        traj_a: AgentTrajectory,
        traj_b: AgentTrajectory,
    ) -> TrajectoryComparison:
        """
        Head-to-head comparison of two trajectories for the same task.
        Produces a structured diff of every behavioral dimension.
        """
        scores_a = traj_a.scores or self.evaluate(traj_a)
        scores_b = traj_b.scores or self.evaluate(traj_b)

        dict_a = scores_a.as_dict()
        dict_b = scores_b.as_dict()

        delta: Dict[str, float] = {
            dim: round(dict_a[dim] - dict_b[dim], 4)
            for dim in dict_a
            if dim != "overall"
        }
        per_dim_winner = {
            dim: ("a" if v > 0 else ("b" if v < 0 else "tie"))
            for dim, v in delta.items()
        }
        a_wins = sum(1 for v in per_dim_winner.values() if v == "a")
        b_wins = sum(1 for v in per_dim_winner.values() if v == "b")
        winner = "a" if a_wins > b_wins else ("b" if b_wins > a_wins else "tie")

        narrative = (
            f"Trajectory A overall={dict_a['overall']:.3f}, "
            f"B overall={dict_b['overall']:.3f}. "
            f"Winner: {winner.upper()}. "
            f"A leads in: {[d for d,w in per_dim_winner.items() if w=='a']}. "
            f"B leads in: {[d for d,w in per_dim_winner.items() if w=='b']}."
        )

        return TrajectoryComparison(
            trajectory_a_id=traj_a.trajectory_id,
            trajectory_b_id=traj_b.trajectory_id,
            task_id=traj_a.task_id,
            delta_scores=delta,
            winner=winner,
            narrative=narrative,
            per_dimension_winner=per_dim_winner,
        )

    def aggregate_report(
        self,
        trajectories: List[AgentTrajectory],
    ) -> Dict[str, float]:
        """
        Population-level statistics across a set of trajectories.

        Returns mean / std per dimension plus aggregate quality profile.
        Useful for benchmarking an agent version against previous runs.
        """
        all_scores = [
            t.scores or self.evaluate(t) for t in trajectories
        ]
        dims = list(DEFAULT_WEIGHTS.keys())
        report: Dict[str, float] = {}

        for dim in dims:
            vals = [getattr(s, dim).value for s in all_scores]
            mean = sum(vals) / len(vals)
            variance = sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)
            report[f"{dim}_mean"] = round(mean, 4)
            report[f"{dim}_std"]  = round(variance ** 0.5, 4)

        overall_vals = [s.overall for s in all_scores]
        report["overall_mean"] = round(sum(overall_vals) / len(overall_vals), 4)

        # Worst-dimension identification
        dim_means = {dim: report[f"{dim}_mean"] for dim in dims}
        report["weakest_dimension"] = min(dim_means, key=dim_means.get)
        report["strongest_dimension"] = max(dim_means, key=dim_means.get)

        return report

    # ── private ─────────────────────────────────────────────────

    def _compute_overall(self, scores: TrajectoryScores) -> float:
        dims = {
            "reasoning_stability":      scores.reasoning_stability.value,
            "conversational_drift":     scores.conversational_drift.value,
            "memory_utility":           scores.memory_utility.value,
            "assumption_accumulation":  scores.assumption_accumulation.value,
            "recovery_ability":         scores.recovery_ability.value,
            "collaboration_quality":    scores.collaboration_quality.value,
            "reflection_usefulness":    scores.reflection_usefulness.value,
            "long_horizon_reliability": scores.long_horizon_reliability.value,
            "trajectory_coherence":     scores.trajectory_coherence.value,
            "user_aligned_quality":     scores.user_aligned_quality.value,
        }
        total = sum(self.weights.get(k, 0.1) * v for k, v in dims.items())
        return round(total, 4)


# ────────────────────────────────────────────────────────────────
# Serialisation helpers
# ────────────────────────────────────────────────────────────────

def _scores_to_dict(scores: TrajectoryScores) -> dict:
    out = {}
    dims = [
        "reasoning_stability", "conversational_drift", "memory_utility",
        "assumption_accumulation", "recovery_ability", "collaboration_quality",
        "reflection_usefulness", "long_horizon_reliability", "trajectory_coherence",
        "user_aligned_quality",
    ]
    for dim in dims:
        ds: DimensionScore = getattr(scores, dim)
        out[dim] = {
            "value": ds.value,
            "explanation": ds.explanation,
            "evidence_steps": ds.evidence_steps,
            "confidence": ds.confidence,
        }
    out["overall"] = scores.overall
    out["task_success"] = scores.task_success
    return out


def _dict_to_scores(raw: dict) -> TrajectoryScores:
    dims = [
        "reasoning_stability", "conversational_drift", "memory_utility",
        "assumption_accumulation", "recovery_ability", "collaboration_quality",
        "reflection_usefulness", "long_horizon_reliability", "trajectory_coherence",
        "user_aligned_quality",
    ]
    kwargs = {}
    for dim in dims:
        d = raw.get(dim, {})
        kwargs[dim] = DimensionScore(
            value=d.get("value", 0.0),
            explanation=d.get("explanation", ""),
            evidence_steps=d.get("evidence_steps", []),
            confidence=d.get("confidence", 1.0),
        )
    kwargs["overall"] = raw.get("overall", 0.0)
    kwargs["task_success"] = raw.get("task_success")
    return TrajectoryScores(**kwargs)
