"""
HARPO Failure Intelligence — Data Schemas

Lightweight dataclasses only. No computation here.
These containers are populated by FailureDetectors and FailureAnalyzers
(Phase 2 implementations) and stored on AgentTrajectory or returned
from HookRegistry post_trajectory callbacks.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class FailureEvent:
    """One observable failure signal within a trajectory step."""
    failure_id:    str   = field(default_factory=lambda: str(uuid.uuid4()))
    trajectory_id: str   = ""
    step_id:       str   = ""
    turn_number:   int   = 0
    failure_type:  str   = ""    # "tool_error"|"loop"|"context_loss"|"assumption"|"contradiction"
    severity:      str   = "medium"  # "low"|"medium"|"high"|"critical"
    description:   str   = ""
    timestamp:     float = field(default_factory=time.time)
    related_steps: List[str] = field(default_factory=list)


@dataclass
class FailureReport:
    """Aggregated failure summary for one trajectory."""
    trajectory_id:    str             = ""
    failure_events:   List[FailureEvent] = field(default_factory=list)
    dominant_failure: Optional[str]   = None   # most frequent failure_type
    failure_density:  float           = 0.0    # failures per turn
    recovery_rate:    float           = 0.0    # recovered failures / total failures
    unrecovered_count: int            = 0
    cascade_detected: bool            = False


@dataclass
class AssumptionPropagation:
    """Tracks how an unverified assumption propagates across turns."""
    assumption_text:     str       = ""
    introduced_at_turn:  int       = 0
    propagated_to_turns: List[int] = field(default_factory=list)
    contradicted:        bool      = False
    contradiction_turn:  Optional[int] = None
    propagation_radius:  int       = 0   # len(propagated_to_turns)


@dataclass
class MemoryCollapse:
    """Signals that an agent lost important context mid-trajectory."""
    trajectory_id:    str   = ""
    detected_at_turn: int   = 0
    lost_context_hint: str  = ""   # snippet of the dropped context
    prior_hit_rate:   float = 0.0  # avg memory hit rate in first half
    post_hit_rate:    float = 0.0  # avg memory hit rate in second half
    delta:            float = 0.0  # post − prior (negative = degradation)


@dataclass
class ReflectionFailure:
    """A reflection step that fired but produced no behavioral change."""
    step_id:         str   = ""
    turn_number:     int   = 0
    reflection_text: str   = ""
    behavior_changed: bool = False   # True only if next step was meaningfully different
    reason:          str   = ""      # why it's classified as a failure


@dataclass
class RecoveryFailure:
    """A recovery attempt that did not resolve the original failure."""
    step_id:          str  = ""
    original_failure: str  = ""
    recovery_attempt: str  = ""
    resolved:         bool = False
    retry_count:      int  = 0


@dataclass
class TrajectoryRegression:
    """Detected regression between two trajectory versions (evolution cycles)."""
    agent_id:        str              = ""
    from_label:      str              = ""
    to_label:        str              = ""
    regressed_dims:  List[str]        = field(default_factory=list)
    delta_per_dim:   Dict[str, float] = field(default_factory=dict)
    severity:        str              = "minor"   # "minor"|"major"|"critical"
    suggested_cause: str              = ""        # Phase 2: LLM-assisted attribution
