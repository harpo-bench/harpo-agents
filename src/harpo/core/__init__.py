"""
src/core — Universal trajectory/evaluation engine.

Re-exports the canonical HARPO data models from src/trajectory/ so that
existing code importing from harpo.trajectory.* continues to work unchanged,
while new code can import from harpo.core.* directly.
"""

from harpo.trajectory.schema import (
    AgentTrajectory,
    AssumptionRecord,
    DimensionScore,
    FailureMode,
    MemoryAccess,
    StepOutcome,
    StepType,
    ToolCall,
    TrajectoryScores,
    TrajectoryStatus,
    TrajectoryStep,
)
from harpo.trajectory.pipeline import TrajectoryEvaluator
from harpo.trajectory.logger import TrajectoryLogger, log_trajectory
from harpo.trajectory.multi_agent import MultiAgentEvaluator

from .events import GenericAgentEvent, GenericEventType, GenericMemoryAccess, GenericToolCall
from .hooks import HookRegistry, HookContext, default_hooks

# Expose schema alias for new code
schema = None  # importable as: from core import AgentTrajectory

__all__ = [
    # Schema
    "AgentTrajectory", "TrajectoryStep", "TrajectoryScores", "DimensionScore",
    "TrajectoryStatus", "StepType", "StepOutcome", "FailureMode",
    "ToolCall", "MemoryAccess", "AssumptionRecord",
    # Pipeline
    "TrajectoryEvaluator",
    # Logger
    "TrajectoryLogger", "log_trajectory",
    # Multi-agent
    "MultiAgentEvaluator",
    # Universal events
    "GenericAgentEvent", "GenericEventType", "GenericToolCall", "GenericMemoryAccess",
    # Hooks
    "HookRegistry", "HookContext", "default_hooks",
]
