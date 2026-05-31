"""
HARPO Failure Intelligence — Abstract Base Classes

These interfaces define the extension protocol for Phase 2 diagnostics.
All concrete implementations live in detectors.py or separate modules.

Extension protocol
------------------
1. Subclass one of the ABCs below.
2. Implement the required method(s).
3. Register with HarpoPlugin:

    from harpo.failures.detectors import LoopDetector
    plugin.hooks.register_failure_detector(LoopDetector())

4. The detector runs automatically at the end of each trajectory evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List, Optional

from harpo.failures.schema import (
    FailureEvent,
    FailureReport,
    MemoryCollapse,
    TrajectoryRegression,
)

if TYPE_CHECKING:
    from harpo.core.schema import AgentTrajectory


class FailureDetector(ABC):
    """
    Scans a completed trajectory for one class of failure.

    One detector per failure type (tool errors, loops, context loss, etc.).
    Returns a list of FailureEvents; empty list means no failures detected.
    """

    @abstractmethod
    def detect(self, trajectory: "AgentTrajectory") -> List[FailureEvent]:
        """Return FailureEvents found in this trajectory."""


class FailureAnalyzer(ABC):
    """
    Aggregates FailureEvents into a FailureReport.

    Typically composed with one or more FailureDetectors:
        events = detector_1.detect(t) + detector_2.detect(t)
        report = analyzer.analyze(events)
    """

    @abstractmethod
    def analyze(self, events: List[FailureEvent]) -> FailureReport:
        """Aggregate raw events into a structured FailureReport."""


class TrajectoryDiagnostic(ABC):
    """
    Higher-level diagnosis that may span multiple trajectories or
    require external context (e.g. comparison with baseline).
    """

    @abstractmethod
    def diagnose(self, trajectory: "AgentTrajectory") -> dict:
        """
        Return a diagnostic report as a plain dict.
        Schema is diagnostic-specific; consumers should treat it as opaque
        unless they know the concrete type.
        """


class RegressionAnalyzer(ABC):
    """Compare two trajectory versions and detect regressions per dimension."""

    @abstractmethod
    def compare(
        self,
        baseline:  "AgentTrajectory",
        candidate: "AgentTrajectory",
    ) -> TrajectoryRegression:
        """
        Return a TrajectoryRegression describing which dimensions degraded
        between baseline and candidate.
        """


class MemoryFailureAnalyzer(ABC):
    """Detect memory degradation patterns within a trajectory."""

    @abstractmethod
    def analyze(self, trajectory: "AgentTrajectory") -> Optional[MemoryCollapse]:
        """
        Return a MemoryCollapse if significant context degradation is detected,
        otherwise None.
        """
