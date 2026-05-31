"""
HARPO Diagnostic Hooks System

Pluggable extension points injected into the evaluation pipeline and
trajectory monitor. Phase 1: registry + context only.
Phase 2: register real FailureDetectors and TrajectoryDiagnostics here.

Usage
-----
# from harpo.core.hooks import default_hooks, HookContext  (self-referential — shown for documentation only)

# Register a custom post-step handler
default_hooks.register_post_step(lambda ctx: print(ctx.step.step_type))

# Register a failure detector (Phase 2)
from harpo.failures.detectors import LoopDetector
default_hooks.register_failure_detector(LoopDetector())
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional

if TYPE_CHECKING:
    from harpo.core.schema import AgentTrajectory, TrajectoryScores, TrajectoryStep
    from harpo.failures.interfaces import FailureDetector, TrajectoryDiagnostic


@dataclass
class HookContext:
    """Passed to every hook. Contains whatever is available at the call site."""
    trajectory: Any                  # AgentTrajectory (Any to avoid circular import)
    scores:     Optional[Any] = None # TrajectoryScores, present only in post_trajectory hooks
    step:       Optional[Any] = None # TrajectoryStep, present only in post_step hooks


class HookRegistry:
    """
    Central registry for pipeline extension points.

    Evaluation pipeline calls:
        hooks.run_post_step(ctx)        — after every step is ingested by Monitor
        hooks.run_post_trajectory(ctx)  — after TrajectoryEvaluator.evaluate() completes

    Both calls are no-ops until hooks/detectors/diagnostics are registered.
    """

    def __init__(self) -> None:
        self.post_step_hooks:       List[Callable[[HookContext], None]] = []
        self.post_trajectory_hooks: List[Callable[[HookContext], None]] = []
        self.failure_detectors:     List[Any] = []  # List[FailureDetector]
        self.diagnostic_processors: List[Any] = []  # List[TrajectoryDiagnostic]

    # ── Registration ────────────────────────────────────────────

    def register_post_step(self, fn: Callable[[HookContext], None]) -> None:
        self.post_step_hooks.append(fn)

    def register_post_trajectory(self, fn: Callable[[HookContext], None]) -> None:
        self.post_trajectory_hooks.append(fn)

    def register_failure_detector(self, detector: Any) -> None:
        self.failure_detectors.append(detector)

    def register_diagnostic(self, diagnostic: Any) -> None:
        self.diagnostic_processors.append(diagnostic)

    # ── Execution ───────────────────────────────────────────────

    def run_post_step(self, ctx: HookContext) -> None:
        for fn in self.post_step_hooks:
            try:
                fn(ctx)
            except Exception:
                pass  # hooks must never crash the evaluation pipeline

    def run_post_trajectory(self, ctx: HookContext) -> None:
        for fn in self.post_trajectory_hooks:
            try:
                fn(ctx)
            except Exception:
                pass

        for detector in self.failure_detectors:
            try:
                detector.detect(ctx.trajectory)
            except Exception:
                pass

        for diag in self.diagnostic_processors:
            try:
                diag.diagnose(ctx.trajectory)
            except Exception:
                pass

    def clear(self) -> None:
        self.post_step_hooks.clear()
        self.post_trajectory_hooks.clear()
        self.failure_detectors.clear()
        self.diagnostic_processors.clear()


# Module-level default registry — used by HarpoPlugin unless overridden
default_hooks = HookRegistry()
