"""
HARPO-Open Trajectory Logger

Captures every agent step in real time and assembles AgentTrajectory objects.
Designed as a lightweight context-manager / decorator that wraps agent execution.

Example
-------
with TrajectoryLogger(agent_id="my_agent", task_id="task_001") as traj:
    # agent code here; call traj.log_think(), traj.log_tool(), etc.
    pass

# traj.trajectory is now a complete AgentTrajectory ready for evaluation
"""

from __future__ import annotations

import hashlib
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict
from typing import Any, Callable, Dict, Generator, List, Optional

from .schema import (
    AgentTrajectory, AssumptionRecord, MemoryAccess, StepOutcome,
    StepType, ToolCall, TrajectoryStatus, TrajectoryStep,
)


class TrajectoryLogger:
    """
    Thin logging harness that wraps agent execution.

    The logger is intentionally side-effect-free with respect to the agent:
    it only observes and records, never modifies agent behaviour.
    """

    def __init__(
        self,
        agent_id: str,
        agent_version: str = "unknown",
        task_id: str = "",
        task_description: str = "",
        user_intent: str = "",
        expected_outcome: Optional[str] = None,
        agent_roles: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.trajectory = AgentTrajectory(
            agent_id=agent_id,
            agent_version=agent_version,
            task_id=task_id,
            task_description=task_description,
            user_intent=user_intent,
            expected_outcome=expected_outcome,
            agent_roles=agent_roles or [],
            metadata=metadata or {},
        )
        self._current_turn: int = 0
        self._step_index: int = 0

    # ─── context manager ────────────────────────────────────────

    def __enter__(self) -> TrajectoryLogger:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        status = TrajectoryStatus.FAILED if exc_type else TrajectoryStatus.COMPLETED
        self.trajectory.status = status
        self.trajectory.ended_at = time.time()
        return False  # never suppress exceptions

    # ─── turn management ────────────────────────────────────────

    def next_turn(self) -> None:
        """Call at the start of each new user/agent turn."""
        self._current_turn += 1
        self._step_index = 0

    # ─── step logging API ───────────────────────────────────────

    def log_think(
        self,
        text: str,
        latency_ms: float = 0.0,
        outcome: StepOutcome = StepOutcome.SUCCESS,
        assumptions: Optional[List[str]] = None,
    ) -> TrajectoryStep:
        """Log an internal reasoning / chain-of-thought step."""
        return self._add(
            step_type=StepType.THINK,
            output_text=text,
            latency_ms=latency_ms,
            outcome=outcome,
            assumptions=[
                AssumptionRecord(text=a, turn_introduced=self._current_turn)
                for a in (assumptions or [])
            ],
        )

    def log_tool_call(
        self,
        name: str,
        arguments: Dict[str, Any],
        result: Any = None,
        error: Optional[str] = None,
        latency_ms: float = 0.0,
    ) -> TrajectoryStep:
        """Log an external tool / API call."""
        outcome = StepOutcome.FAILURE if error else StepOutcome.SUCCESS
        tool = ToolCall(name=name, arguments=arguments, result=result, error=error, latency_ms=latency_ms)
        return self._add(
            step_type=StepType.TOOL_CALL,
            input_text=f"{name}({arguments})",
            output_text=str(result) if result is not None else (error or ""),
            latency_ms=latency_ms,
            outcome=outcome,
            tool_call=tool,
        )

    def log_observation(self, text: str, latency_ms: float = 0.0) -> TrajectoryStep:
        """Log the result of an observation (e.g. tool output)."""
        return self._add(
            step_type=StepType.OBSERVATION,
            output_text=text,
            latency_ms=latency_ms,
        )

    def log_response(
        self,
        text: str,
        outcome: StepOutcome = StepOutcome.SUCCESS,
        latency_ms: float = 0.0,
    ) -> TrajectoryStep:
        """Log the agent's final response for this turn."""
        if outcome == StepOutcome.SUCCESS:
            self.trajectory.final_output = text
        return self._add(
            step_type=StepType.RESPONSE,
            output_text=text,
            latency_ms=latency_ms,
            outcome=outcome,
        )

    def log_reflection(self, text: str, latency_ms: float = 0.0) -> TrajectoryStep:
        """Log a self-critique or self-correction step."""
        return self._add(
            step_type=StepType.REFLECTION,
            output_text=text,
            latency_ms=latency_ms,
        )

    def log_memory_read(
        self,
        key: str,
        value: Any,
        hit: bool = True,
        relevance_score: float = 0.0,
    ) -> TrajectoryStep:
        """Log a memory retrieval operation."""
        access = MemoryAccess(
            operation="read",
            key=key,
            value=value,
            hit=hit,
            relevance_score=relevance_score,
        )
        return self._add(
            step_type=StepType.MEMORY_READ,
            input_text=key,
            output_text=str(value)[:200] if value is not None else "",
            memory_access=access,
        )

    def log_memory_write(self, key: str, value: Any) -> TrajectoryStep:
        """Log a memory storage operation."""
        access = MemoryAccess(operation="write", key=key, value=value)
        return self._add(
            step_type=StepType.MEMORY_WRITE,
            input_text=key,
            output_text=str(value)[:200] if value is not None else "",
            memory_access=access,
        )

    def log_handoff(
        self,
        target_agent: str,
        task_spec: str,
        latency_ms: float = 0.0,
    ) -> TrajectoryStep:
        """Log delegation to a sub-agent."""
        return self._add(
            step_type=StepType.HANDOFF,
            input_text=target_agent,
            output_text=task_spec,
            latency_ms=latency_ms,
        )

    def log_recovery(
        self,
        context: str,
        corrective_action: str,
        outcome: StepOutcome = StepOutcome.SUCCESS,
    ) -> TrajectoryStep:
        """Log an explicit error-recovery attempt."""
        return self._add(
            step_type=StepType.RECOVERY,
            input_text=context,
            output_text=corrective_action,
            outcome=outcome,
        )

    # ─── decorator ──────────────────────────────────────────────

    def instrument(self, func: Callable) -> Callable:
        """
        Decorator that auto-logs the function call as a TOOL_CALL step.

        Usage::

            @traj_logger.instrument
            def search_web(query: str) -> str: ...
        """
        import functools

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                result = func(*args, **kwargs)
                latency = (time.perf_counter() - t0) * 1000
                self.log_tool_call(
                    name=func.__name__,
                    arguments={"args": args, "kwargs": kwargs},
                    result=result,
                    latency_ms=latency,
                )
                return result
            except Exception as e:
                latency = (time.perf_counter() - t0) * 1000
                self.log_tool_call(
                    name=func.__name__,
                    arguments={"args": args, "kwargs": kwargs},
                    error=str(e),
                    latency_ms=latency,
                )
                raise

        return wrapper

    # ─── internal ───────────────────────────────────────────────

    def _add(self, **kwargs) -> TrajectoryStep:
        step = TrajectoryStep(
            trajectory_id=self.trajectory.trajectory_id,
            turn_number=self._current_turn,
            step_index=self._step_index,
            raw_tokens=len(kwargs.get("output_text", "").split()),
            semantic_hash=_short_hash(kwargs.get("output_text", "")),
            **kwargs,
        )
        self._step_index += 1
        self.trajectory.add_step(step)
        return step


def _short_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()[:8]


# ─── convenience: async-compatible context manager ──────────────

@contextmanager
def log_trajectory(
    agent_id: str,
    task_id: str = "",
    user_intent: str = "",
    **kwargs,
) -> Generator[TrajectoryLogger, None, None]:
    """
    Convenience context manager.

        with log_trajectory("my_agent", task_id="t1", user_intent="find bugs") as L:
            L.log_think("Analysing code...")
            L.log_tool_call("grep", {"pattern": "TODO"}, result=["line 42"])
            L.log_response("Found 1 TODO on line 42.")
    """
    logger = TrajectoryLogger(agent_id=agent_id, task_id=task_id, user_intent=user_intent, **kwargs)
    try:
        yield logger
        logger.trajectory.status = TrajectoryStatus.COMPLETED
    except Exception:
        logger.trajectory.status = TrajectoryStatus.FAILED
        raise
    finally:
        logger.trajectory.ended_at = time.time()
