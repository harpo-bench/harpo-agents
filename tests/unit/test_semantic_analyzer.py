"""Unit tests for SemanticTrajectoryAnalyzer."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep, StepType, StepOutcome
from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
import time


def _make_traj(steps_text):
    traj = AgentTrajectory(trajectory_id="test", agent_id="agent-a", user_intent="test task")
    for i, text in enumerate(steps_text):
        traj.add_step(TrajectoryStep(
            trajectory_id="test", turn_number=i+1, step_index=i,
            step_type=StepType.THINK, outcome=StepOutcome.SUCCESS,
            input_text="", output_text=text, timestamp=time.time(),
        ))
    return traj


def test_analyzer_returns_analysis():
    traj = _make_traj(["I think the budget is $5M. Assuming that is correct, we proceed."])
    analysis = SemanticTrajectoryAnalyzer(run_causal=False).analyze(traj)
    assert analysis is not None
    assert analysis.contradictions is not None
    assert analysis.assumptions is not None


def test_no_crash_on_empty_trajectory():
    traj = AgentTrajectory(trajectory_id="empty", agent_id="a", user_intent="test")
    analysis = SemanticTrajectoryAnalyzer(run_causal=False).analyze(traj)
    assert analysis is not None


def test_assumption_detected():
    traj = _make_traj([
        "I assume the budget is $5 million for this project.",
        "Based on the $5M assumption, we will hire 20 engineers and spend $1M on cloud.",
    ])
    analysis = SemanticTrajectoryAnalyzer(run_causal=False).analyze(traj)
    assert analysis.assumptions.total_assumptions > 0
