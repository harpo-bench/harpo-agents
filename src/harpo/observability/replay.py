"""
HARPO Trajectory Replay

Re-emit a stored trajectory's steps through the monitor and diagnostic hooks,
at configurable speed. Useful for:
  - Post-hoc debugging: replay a failed run through live alerts
  - Testing: verify monitor + detector behaviour against scripted trajectories
  - Dashboard: replay historical runs in the UI

Usage
-----
from harpo.observability.replay import TrajectoryReplayer
from harpo.observability.realtime import TrajectoryMonitor
from harpo.core.hooks import default_hooks

replayer = TrajectoryReplayer(
    monitor=TrajectoryMonitor("replay-001"),
    hooks=default_hooks,
    speed=0,   # 0 = instant, 1.0 = real-time, 2.0 = 2× real-time
)
replayer.replay(trajectory)
print(replayer.monitor.snapshot())
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Any

from harpo.observability.realtime import TrajectoryMonitor
from harpo.core.hooks import HookContext, HookRegistry
from harpo.core.schema import AgentTrajectory, TrajectoryStep


@dataclass
class ReplayEvent:
    """Emitted after each step during replay — for testing / assertions."""
    step_index: int
    step:       TrajectoryStep
    monitor_snapshot: Dict[str, Any]


class TrajectoryReplayer:
    """
    Re-plays a stored AgentTrajectory through the monitor + hooks.

    Parameters
    ----------
    monitor : TrajectoryMonitor
        Receives ingested steps (same as live execution).
    hooks   : HookRegistry
        post_step_hooks are called after each step.
    speed   : float
        0   = instant (no sleep)
        1.0 = real-time (sleep proportional to step.latency_ms)
        N   = N× real-time speed
    on_step : optional callback
        Called with a ReplayEvent after each step — useful in tests.
    """

    def __init__(
        self,
        monitor:  Optional[TrajectoryMonitor] = None,
        hooks:    Optional[HookRegistry]     = None,
        speed:    float = 0,
        on_step:  Optional[Callable[[ReplayEvent], None]] = None,
    ) -> None:
        self._monitor  = monitor
        self._hooks    = hooks
        self._speed    = speed
        self._on_step  = on_step
        self._replayed: List[ReplayEvent] = []

    # ── Public ──────────────────────────────────────────────────

    def replay(self, trajectory: AgentTrajectory) -> List[ReplayEvent]:
        """
        Replay all steps in trajectory order.
        Returns the list of ReplayEvents emitted.
        """
        self._replayed = []

        monitor = self._monitor or TrajectoryMonitor(trajectory.trajectory_id)
        hooks   = self._hooks

        for idx, step in enumerate(trajectory.steps):
            monitor.ingest(step)

            if hooks:
                ctx = HookContext(trajectory=trajectory, step=step)
                hooks.run_post_step(ctx)

            snap = monitor.snapshot()
            evt  = ReplayEvent(step_index=idx, step=step, monitor_snapshot=snap)
            self._replayed.append(evt)

            if self._on_step:
                self._on_step(evt)

            if self._speed > 0 and step.latency_ms > 0:
                time.sleep(step.latency_ms / 1000.0 / self._speed)

        # Run post-trajectory hooks after all steps
        if hooks:
            ctx = HookContext(trajectory=trajectory)
            hooks.run_post_trajectory(ctx)

        return self._replayed

    def events(self) -> List[ReplayEvent]:
        """Return replay events from the last replay() call."""
        return self._replayed

    def monitor_snapshot(self) -> Dict[str, Any]:
        """Return the monitor snapshot after the last replayed step."""
        if not self._replayed:
            return {}
        return self._replayed[-1].monitor_snapshot
