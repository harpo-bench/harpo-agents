"""HARPO-Open: Trajectory-level behavioral evaluation for AI agents."""

from .schema import (
    AgentTrajectory, TrajectoryStep, TrajectoryScores, DimensionScore,
    FailureReport, TrajectoryComparison, StepType, StepOutcome,
    TrajectoryStatus, FailureMode, ToolCall, MemoryAccess, AssumptionRecord,
)
from .logger import TrajectoryLogger, log_trajectory
from .pipeline import TrajectoryEvaluator, DEFAULT_WEIGHTS
from .metrics import (
    score_reasoning_stability, score_conversational_drift,
    score_memory_utility, score_assumption_accumulation,
    score_recovery_ability, score_collaboration_quality,
    score_reflection_usefulness, score_long_horizon_reliability,
    score_trajectory_coherence, score_user_aligned_quality,
    detect_failure_modes,
)
from .multi_agent import MultiAgentEvaluator, MultiAgentReport

__all__ = [
    "AgentTrajectory", "TrajectoryStep", "TrajectoryScores", "DimensionScore",
    "FailureReport", "TrajectoryComparison", "StepType", "StepOutcome",
    "TrajectoryStatus", "FailureMode", "ToolCall", "MemoryAccess", "AssumptionRecord",
    "TrajectoryLogger", "log_trajectory",
    "TrajectoryEvaluator", "DEFAULT_WEIGHTS",
    "score_reasoning_stability", "score_conversational_drift",
    "score_memory_utility", "score_assumption_accumulation",
    "score_recovery_ability", "score_collaboration_quality",
    "score_reflection_usefulness", "score_long_horizon_reliability",
    "score_trajectory_coherence", "score_user_aligned_quality",
    "detect_failure_modes",
    "MultiAgentEvaluator", "MultiAgentReport",
]
