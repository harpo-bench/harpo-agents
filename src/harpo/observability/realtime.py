"""
HARPO-Open Real-Time Observability

Streams live trajectory metrics as a trajectory executes.
Designed to integrate with dashboards, alerting, and live debugging.

Architecture:
  TrajectoryMonitor          — subscribes to a TrajectoryLogger and emits events
  MetricStreamBuffer         — rolling window of recent metric samples
  AnomalyDetector            — signals when a metric crosses a threshold
  ObservabilityBridge        — connects to external sinks (Prometheus, OTEL, SSE)
"""

from __future__ import annotations

import math
import queue
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional

try:
    from ..trajectory.schema import (
        AgentTrajectory, StepOutcome, StepType, TrajectoryStep,
    )
except ImportError:
    from harpo.trajectory.schema import (  # type: ignore
        AgentTrajectory, StepOutcome, StepType, TrajectoryStep,
    )


# ───────────────────────────────────────────────────────────────
# Event types
# ───────────────────────────────────────────────────────────────

@dataclass
class LiveMetricEvent:
    """Emitted after each step is logged."""
    trajectory_id: str
    step_id: str
    timestamp: float
    metric_name: str
    value: float
    turn: int


@dataclass
class AlertEvent:
    """Emitted when a metric crosses a threshold."""
    trajectory_id: str
    alert_type: str
    metric_name: str
    current_value: float
    threshold: float
    severity: str          # "warn" | "critical"
    message: str
    timestamp: float = field(default_factory=time.time)


# ───────────────────────────────────────────────────────────────
# Rolling metric buffer
# ───────────────────────────────────────────────────────────────

class MetricStreamBuffer:
    """
    Rolling window of per-step metric samples.
    Supports: current value, moving average, trend direction, spike detection.
    """

    def __init__(self, window: int = 50):
        self._window = window
        self._series: Dict[str, Deque[float]] = {}

    def push(self, name: str, value: float) -> None:
        if name not in self._series:
            self._series[name] = deque(maxlen=self._window)
        self._series[name].append(value)

    def moving_average(self, name: str, n: int = 10) -> Optional[float]:
        s = self._series.get(name)
        if not s:
            return None
        window = list(s)[-n:]
        return sum(window) / len(window)

    def trend(self, name: str) -> float:
        """Positive = rising, negative = falling, 0 = flat."""
        s = list(self._series.get(name, []))
        if len(s) < 4:
            return 0.0
        half = len(s) // 2
        return (sum(s[half:]) / (len(s) - half)) - (sum(s[:half]) / half)

    def is_spike(self, name: str, z_threshold: float = 2.5) -> bool:
        s = list(self._series.get(name, []))
        if len(s) < 5:
            return False
        mean = sum(s) / len(s)
        std  = math.sqrt(sum((v - mean) ** 2 for v in s) / len(s))
        if std == 0:
            return False
        return abs(s[-1] - mean) > z_threshold * std

    def all_current(self) -> Dict[str, float]:
        return {name: list(buf)[-1] for name, buf in self._series.items() if buf}


# ───────────────────────────────────────────────────────────────
# Alert rules
# ───────────────────────────────────────────────────────────────

DEFAULT_ALERT_RULES: List[Dict[str, Any]] = [
    {"metric": "failure_rate",         "threshold": 0.3,  "severity": "critical"},
    {"metric": "consecutive_failures", "threshold": 3.0,  "severity": "critical"},
    {"metric": "avg_latency_ms",       "threshold": 8000, "severity": "warn"},
    {"metric": "assumption_density",   "threshold": 0.5,  "severity": "warn"},
    {"metric": "tool_error_rate",      "threshold": 0.4,  "severity": "critical"},
    {"metric": "context_overlap_drop", "threshold": 0.2,  "severity": "warn"},
]


class AnomalyDetector:
    """Rule-based + statistical anomaly detection over live metrics."""

    def __init__(self, rules: Optional[List[Dict]] = None):
        self._rules = rules or DEFAULT_ALERT_RULES

    def check(
        self,
        trajectory_id: str,
        buffer: MetricStreamBuffer,
    ) -> List[AlertEvent]:
        alerts = []
        current = buffer.all_current()

        for rule in self._rules:
            metric = rule["metric"]
            val = current.get(metric)
            if val is None:
                continue
            if val >= rule["threshold"]:
                alerts.append(AlertEvent(
                    trajectory_id=trajectory_id,
                    alert_type="threshold_breach",
                    metric_name=metric,
                    current_value=val,
                    threshold=rule["threshold"],
                    severity=rule["severity"],
                    message=(
                        f"{metric} = {val:.3f} exceeds threshold {rule['threshold']} "
                        f"[{rule['severity']}]"
                    ),
                ))

        # Statistical: spike detection on latency
        if buffer.is_spike("avg_latency_ms", z_threshold=2.5):
            alerts.append(AlertEvent(
                trajectory_id=trajectory_id,
                alert_type="statistical_spike",
                metric_name="avg_latency_ms",
                current_value=current.get("avg_latency_ms", 0.0),
                threshold=0.0,
                severity="warn",
                message="Latency spike detected (>2.5σ from rolling mean).",
            ))

        return alerts


# ───────────────────────────────────────────────────────────────
# Main monitor
# ───────────────────────────────────────────────────────────────

class TrajectoryMonitor:
    """
    Subscribes to a live trajectory and emits real-time metrics.

    Usage
    -----
    monitor = TrajectoryMonitor(trajectory_id="t-001")
    monitor.on_metric(lambda e: print(e))
    monitor.on_alert(lambda a: send_to_slack(a))

    # Then call monitor.ingest(step) from your logger callback
    """

    def __init__(
        self,
        trajectory_id: str,
        alert_rules: Optional[List[Dict]] = None,
        window: int = 50,
    ):
        self.trajectory_id = trajectory_id
        self._buffer = MetricStreamBuffer(window=window)
        self._detector = AnomalyDetector(alert_rules)
        self._metric_callbacks: List[Callable[[LiveMetricEvent], None]] = []
        self._alert_callbacks:  List[Callable[[AlertEvent], None]] = []
        self._consecutive_failures = 0

    def on_metric(self, callback: Callable[[LiveMetricEvent], None]) -> None:
        self._metric_callbacks.append(callback)

    def on_alert(self, callback: Callable[[AlertEvent], None]) -> None:
        self._alert_callbacks.append(callback)

    def ingest(self, step: TrajectoryStep) -> None:
        """Process a new step and emit metrics + alerts."""
        metrics = self._extract_step_metrics(step)
        for name, val in metrics.items():
            self._buffer.push(name, val)
            evt = LiveMetricEvent(
                trajectory_id=self.trajectory_id,
                step_id=step.step_id,
                timestamp=step.timestamp,
                metric_name=name,
                value=val,
                turn=step.turn_number,
            )
            for cb in self._metric_callbacks:
                cb(evt)

        alerts = self._detector.check(self.trajectory_id, self._buffer)
        for alert in alerts:
            for cb in self._alert_callbacks:
                cb(alert)

    def snapshot(self) -> Dict[str, Any]:
        """Return current live metric snapshot."""
        current = self._buffer.all_current()
        return {
            "trajectory_id": self.trajectory_id,
            "metrics": current,
            "trends": {
                name: self._buffer.trend(name)
                for name in current
            },
            "moving_averages": {
                name: self._buffer.moving_average(name, n=10)
                for name in current
            },
        }

    # ─── private ────────────────────────────────────────────────

    def _extract_step_metrics(self, step: TrajectoryStep) -> Dict[str, float]:
        metrics: Dict[str, float] = {}

        # Binary: failure this step
        is_failure = float(step.outcome == StepOutcome.FAILURE)
        metrics["step_failure"] = is_failure

        if step.outcome == StepOutcome.FAILURE:
            self._consecutive_failures += 1
        else:
            self._consecutive_failures = 0
        metrics["consecutive_failures"] = float(self._consecutive_failures)

        # Latency
        if step.latency_ms > 0:
            metrics["avg_latency_ms"] = step.latency_ms

        # Tool error rate
        if step.step_type == StepType.TOOL_CALL:
            metrics["tool_error_rate"] = float(bool(step.tool_call and step.tool_call.error))

        # Memory hit rate
        if step.step_type == StepType.MEMORY_READ and step.memory_access:
            metrics["memory_hit"] = float(step.memory_access.hit)
            metrics["memory_relevance"] = step.memory_access.relevance_score

        # Token output length: prefer raw_tokens from adapter; fall back to
        # whitespace-word count × 1.3 (rough chars-per-token proxy) so the
        # metric is never silently 0 when the runtime doesn't expose counts.
        if step.raw_tokens > 0:
            metrics["output_tokens"] = float(step.raw_tokens)
        elif step.output_text:
            metrics["output_tokens"] = float(len(step.output_text.split()) * 1.3)

        # Assumption density
        assumption_phrases = ["I assume", "probably", "likely", "I think", "perhaps"]
        density = sum(
            1 for p in assumption_phrases if p.lower() in step.output_text.lower()
        )
        metrics["assumption_density"] = float(density)

        return metrics


# ───────────────────────────────────────────────────────────────
# ObservabilityBridge: export to external sinks
# ───────────────────────────────────────────────────────────────

class ObservabilityBridge:
    """
    Routes live events to pluggable sinks.

    Built-in sinks:
    - Prometheus push gateway (requires `prometheus_client`)
    - OpenTelemetry (requires `opentelemetry-sdk`)
    - Server-Sent Events queue (for dashboard)
    - Simple JSON file sink
    """

    def __init__(self):
        self._sse_queue: queue.Queue = queue.Queue(maxsize=1000)
        self._json_sink: Optional[str] = None  # file path

    def enable_json_sink(self, path: str) -> None:
        self._json_sink = path

    def consume_sse(self) -> Optional[dict]:
        """Non-blocking read for SSE endpoint."""
        try:
            return self._sse_queue.get_nowait()
        except queue.Empty:
            return None

    def handle_metric(self, event: LiveMetricEvent) -> None:
        payload = {
            "type": "metric",
            "trajectory_id": event.trajectory_id,
            "metric": event.metric_name,
            "value": event.value,
            "turn": event.turn,
            "ts": event.timestamp,
        }
        self._push(payload)

    def handle_alert(self, alert: AlertEvent) -> None:
        payload = {
            "type": "alert",
            "trajectory_id": alert.trajectory_id,
            "alert_type": alert.alert_type,
            "metric": alert.metric_name,
            "value": alert.current_value,
            "severity": alert.severity,
            "message": alert.message,
            "ts": alert.timestamp,
        }
        self._push(payload)

    def _push(self, payload: dict) -> None:
        try:
            self._sse_queue.put_nowait(payload)
        except queue.Full:
            pass  # drop if buffer full

        if self._json_sink:
            import json
            try:
                with open(self._json_sink, "a") as f:
                    f.write(json.dumps(payload) + "\n")
            except OSError:
                pass
