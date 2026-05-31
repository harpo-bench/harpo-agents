"""Unit tests for trajectory schema."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from harpo.trajectory.schema import (
    AgentTrajectory, TrajectoryStep, StepType, StepOutcome, TrajectoryStatus
)
import time, uuid


def _step(step_type=StepType.THINK, text="test reasoning output"):
    return TrajectoryStep(
        trajectory_id="test-traj",
        turn_number=1,
        step_index=0,
        step_type=step_type,
        outcome=StepOutcome.SUCCESS,
        input_text="",
        output_text=text,
        timestamp=time.time(),
    )


def test_trajectory_creation():
    traj = AgentTrajectory(trajectory_id="t1", agent_id="agent-a", user_intent="test")
    assert traj.trajectory_id == "t1"
    assert traj.status == TrajectoryStatus.IN_PROGRESS


def test_add_step():
    traj = AgentTrajectory(trajectory_id="t1", agent_id="agent-a", user_intent="test")
    step = _step()
    traj.add_step(step)
    assert len(traj.steps) == 1
    assert traj.steps[0].step_type == StepType.THINK


def test_step_types_exist():
    for t in [StepType.THINK, StepType.TOOL_CALL, StepType.OBSERVATION,
              StepType.RESPONSE, StepType.REFLECTION, StepType.RECOVERY,
              StepType.MEMORY_READ, StepType.MEMORY_WRITE]:
        assert t is not None
