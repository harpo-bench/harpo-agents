"""
HARPO Trajectory Schema
Core data models for trajectory-level agent evaluation.

Every unit of agent behavior is captured here before evaluation.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


# ============================================================
# Enums
# ============================================================

class StepType(str, Enum):
    THINK       = "think"        # internal chain-of-thought
    TOOL_CALL   = "tool_call"    # external tool/API invocation
    OBSERVATION = "observation"  # result of a tool call
    RESPONSE    = "response"     # final response to user/caller
    REFLECTION  = "reflection"   # self-critique / self-edit
    MEMORY_READ = "memory_read"  # memory retrieval
    MEMORY_WRITE= "memory_write" # memory storage
    HANDOFF     = "handoff"      # sub-agent delegation
    RECOVERY    = "recovery"     # error correction step


class StepOutcome(str, Enum):
    SUCCESS  = "success"
    FAILURE  = "failure"
    RETRY    = "retry"
    SKIPPED  = "skipped"
    PARTIAL  = "partial"


class TrajectoryStatus(str, Enum):
    IN_PROGRESS = "in_progress"
    COMPLETED   = "completed"
    FAILED      = "failed"
    ABANDONED   = "abandoned"


class FailureMode(str, Enum):
    HALLUCINATION       = "hallucination"
    TOOL_MISUSE         = "tool_misuse"
    CONTEXT_LOSS        = "context_loss"
    ASSUMPTION_ERROR    = "assumption_error"
    LOOP_DETECTED       = "loop_detected"
    CONTRADICTION       = "contradiction"
    PREMATURE_STOP      = "premature_stop"
    OVER_REASONING      = "over_reasoning"
    MEMORY_MISS         = "memory_miss"
    COLLABORATION_FAIL  = "collaboration_fail"


# ============================================================
# Core step-level data
# ============================================================

@dataclass
class ToolCall:
    name: str
    arguments: Dict[str, Any]
    result: Optional[Any] = None
    error: Optional[str] = None
    latency_ms: float = 0.0


@dataclass
class MemoryAccess:
    operation: str          # "read" | "write" | "update" | "invalidate"
    key: str
    value: Optional[Any]    # written value or retrieved value
    hit: bool = True        # False = cache miss
    relevance_score: float = 0.0  # 0-1: how relevant was what was retrieved
    version: int = 1        # version of the memory object read/written
    is_stale: bool = False  # True = a newer version existed when this was read
    current_version: int = 1  # latest version at time of read (for stale detection)


@dataclass
class AssumptionRecord:
    text: str
    turn_introduced: int
    verified: bool = False
    contradicted: bool = False
    impact_score: float = 0.0  # downstream decision impact


@dataclass
class TrajectoryStep:
    """A single atomic step in an agent's trajectory."""

    # Identity
    step_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory_id: str = ""
    turn_number: int = 0
    step_index: int = 0  # step within a turn
    step_type: StepType = StepType.THINK
    outcome: StepOutcome = StepOutcome.SUCCESS

    # Content
    input_text: str = ""
    output_text: str = ""
    raw_tokens: int = 0

    # Timing
    timestamp: float = field(default_factory=time.time)
    latency_ms: float = 0.0

    # Rich payload
    tool_call: Optional[ToolCall] = None
    memory_access: Optional[MemoryAccess] = None
    assumptions: List[AssumptionRecord] = field(default_factory=list)

    # Multi-agent tagging (populated by adapter or builder)
    agent_id:    str = ""          # which agent produced this step
    agent_roles: List[str] = field(default_factory=list)

    # Embeddings (optional, populated by analyzer)
    hidden_vector: Optional[List[float]] = None
    semantic_hash: Optional[str] = None


# ============================================================
# Trajectory-level container
# ============================================================

@dataclass
class AgentTrajectory:
    """
    Full trajectory for one agent session.

    Analogous to a conversation thread but tracks every internal step,
    not just the visible turns.
    """

    trajectory_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str = ""
    agent_id: str = ""
    agent_version: str = ""
    task_id: str = ""
    task_description: str = ""

    # Participants (for multi-agent runs)
    agent_roles: List[str] = field(default_factory=list)  # e.g. ["orchestrator", "tool_agent"]

    # Timeline
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    status: TrajectoryStatus = TrajectoryStatus.IN_PROGRESS

    # Ordered steps
    steps: List[TrajectoryStep] = field(default_factory=list)

    # High-level intent / expected outcome (used for alignment scoring)
    user_intent: str = ""
    expected_outcome: Optional[str] = None
    final_output: str = ""

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Evaluation results (populated post-run)
    scores: Optional["TrajectoryScores"] = None
    failure_report: Optional["FailureReport"] = None

    # ---- convenience helpers ----

    def add_step(self, step: TrajectoryStep) -> None:
        step.trajectory_id = self.trajectory_id
        self.steps.append(step)

    def turns(self) -> List[List[TrajectoryStep]]:
        """Group steps by turn_number."""
        if not self.steps:
            return []
        max_turn = max(s.turn_number for s in self.steps)
        return [
            [s for s in self.steps if s.turn_number == t]
            for t in range(max_turn + 1)
        ]

    def duration_ms(self) -> float:
        if self.ended_at:
            return (self.ended_at - self.started_at) * 1000
        return (time.time() - self.started_at) * 1000

    def step_count(self, step_type: Optional[StepType] = None) -> int:
        if step_type is None:
            return len(self.steps)
        return sum(1 for s in self.steps if s.step_type == step_type)


# ============================================================
# Evaluation output models
# ============================================================

@dataclass
class DimensionScore:
    """Score for one evaluation dimension with supporting evidence."""
    value: float              # 0.0 – 1.0
    explanation: str = ""
    evidence_steps: List[str] = field(default_factory=list)  # step_ids
    confidence: float = 1.0


@dataclass
class TrajectoryScores:
    """
    Full HARPO-Open behavioral scoring output for one trajectory.

    10 primary dimensions mapped to the HARPO evaluation philosophy:
    trajectory quality > final correctness.
    """

    # Core HARPO behavioral dimensions
    reasoning_stability:        DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    conversational_drift:       DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    memory_utility:             DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    assumption_accumulation:    DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    recovery_ability:           DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    collaboration_quality:      DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    reflection_usefulness:      DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    long_horizon_reliability:   DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    trajectory_coherence:       DimensionScore = field(default_factory=lambda: DimensionScore(0.0))
    user_aligned_quality:       DimensionScore = field(default_factory=lambda: DimensionScore(0.0))

    # Aggregate
    overall: float = 0.0

    # Optional task-success overlay (orthogonal to trajectory quality)
    task_success: Optional[float] = None

    def as_dict(self) -> Dict[str, float]:
        return {
            "reasoning_stability":      self.reasoning_stability.value,
            "conversational_drift":     self.conversational_drift.value,
            "memory_utility":           self.memory_utility.value,
            "assumption_accumulation":  self.assumption_accumulation.value,
            "recovery_ability":         self.recovery_ability.value,
            "collaboration_quality":    self.collaboration_quality.value,
            "reflection_usefulness":    self.reflection_usefulness.value,
            "long_horizon_reliability": self.long_horizon_reliability.value,
            "trajectory_coherence":     self.trajectory_coherence.value,
            "user_aligned_quality":     self.user_aligned_quality.value,
            "overall":                  self.overall,
        }


@dataclass
class FailureReport:
    """Structured failure analysis for one trajectory."""
    failure_modes: List[FailureMode] = field(default_factory=list)
    first_failure_turn: Optional[int] = None
    cascade_detected: bool = False
    recovery_attempted: bool = False
    recovery_succeeded: bool = False
    root_cause: str = ""
    contributing_steps: List[str] = field(default_factory=list)  # step_ids
    severity: float = 0.0   # 0-1

    def is_clean(self) -> bool:
        return len(self.failure_modes) == 0


# ============================================================
# Multi-trajectory comparison
# ============================================================

@dataclass
class TrajectoryComparison:
    """Side-by-side comparison of two trajectories for the same task."""
    trajectory_a_id: str
    trajectory_b_id: str
    task_id: str

    delta_scores: Dict[str, float] = field(default_factory=dict)  # dim → a - b
    winner: Optional[str] = None   # "a" | "b" | "tie"
    narrative: str = ""
    per_dimension_winner: Dict[str, str] = field(default_factory=dict)
