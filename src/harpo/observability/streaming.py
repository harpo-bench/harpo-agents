"""
HARPO Real-Time Streaming Server

WebSocket + SSE endpoints that stream live trajectory metrics and alerts
as they are emitted by TrajectoryMonitor / ObservabilityBridge.

Architecture
------------
HarpoStreamServer wraps ObservabilityBridge and exposes:
  WS  /v1/ws/trajectories/{id}   — push GenericAgentEvent (JSON), receive metrics + alerts
  SSE /v1/sse/trajectories/{id}  — same stream as SSE for browser dashboards
  POST /v1/ingest/event           — REST ingestion (non-streaming adapters)

Usage (FastAPI app)
-------------------
from harpo.observability.streaming import HarpoStreamServer
from harpo.observability.realtime import ObservabilityBridge

bridge = ObservabilityBridge()
server = HarpoStreamServer(bridge)
app.include_router(server.get_router(), prefix="/v1")
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Callable, Dict, Optional

try:
    from fastapi import APIRouter, WebSocket, WebSocketDisconnect
    from fastapi.responses import StreamingResponse
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from harpo.observability.realtime import (
    AlertEvent,
    LiveMetricEvent,
    ObservabilityBridge,
    TrajectoryMonitor,
)
from harpo.core.schema import TrajectoryStep


class HarpoStreamServer:
    """
    Bridges live trajectory events to WebSocket and SSE clients.

    One ObservabilityBridge instance is shared across all monitors.
    Each trajectory gets its own TrajectoryMonitor that feeds the bridge.
    """

    def __init__(self, bridge: Optional[ObservabilityBridge] = None) -> None:
        self._bridge = bridge or ObservabilityBridge()
        self._monitors: Dict[str, TrajectoryMonitor] = {}

    # ── Monitor management ───────────────────────────────────────

    def get_or_create_monitor(self, trajectory_id: str) -> TrajectoryMonitor:
        if trajectory_id not in self._monitors:
            monitor = TrajectoryMonitor(trajectory_id)
            monitor.on_metric(self._bridge.handle_metric)
            monitor.on_alert(self._bridge.handle_alert)
            self._monitors[trajectory_id] = monitor
        return self._monitors[trajectory_id]

    def ingest_step(self, trajectory_id: str, step: TrajectoryStep) -> None:
        """Feed one step into the live monitor for a trajectory."""
        monitor = self.get_or_create_monitor(trajectory_id)
        monitor.ingest(step)

    # ── FastAPI router ───────────────────────────────────────────

    def get_router(self) -> "APIRouter":
        if not FASTAPI_AVAILABLE:
            raise ImportError("fastapi is required for HarpoStreamServer. pip install fastapi")

        router = APIRouter()

        @router.websocket("/ws/trajectories/{trajectory_id}")
        async def ws_trajectory(websocket: "WebSocket", trajectory_id: str):
            """
            WebSocket endpoint for live trajectory monitoring.

            Client sends: JSON-encoded step events (from adapter or test harness)
            Server sends: LiveMetricEvent + AlertEvent as JSON
            """
            await websocket.accept()
            monitor = self.get_or_create_monitor(trajectory_id)

            # Subscribe websocket to events from this trajectory's monitor
            async def send_metric(evt: LiveMetricEvent):
                try:
                    await websocket.send_json({
                        "type":    "metric",
                        "metric":  evt.metric_name,
                        "value":   evt.value,
                        "turn":    evt.turn,
                        "ts":      evt.timestamp,
                    })
                except Exception:
                    pass

            async def send_alert(alert: AlertEvent):
                try:
                    await websocket.send_json({
                        "type":     "alert",
                        "metric":   alert.metric_name,
                        "severity": alert.severity,
                        "message":  alert.message,
                        "ts":       alert.timestamp,
                    })
                except Exception:
                    pass

            # Wrap async senders into sync callbacks (monitor uses sync)
            loop = asyncio.get_event_loop()
            monitor.on_metric(lambda e: asyncio.run_coroutine_threadsafe(send_metric(e), loop))
            monitor.on_alert(lambda a: asyncio.run_coroutine_threadsafe(send_alert(a), loop))

            try:
                while True:
                    await websocket.receive_text()   # keep connection alive
            except WebSocketDisconnect:
                pass

        @router.get("/sse/trajectories/{trajectory_id}")
        async def sse_trajectory(trajectory_id: str):
            """
            SSE endpoint — for browser dashboards using EventSource.
            Polls ObservabilityBridge queue and streams events as text/event-stream.
            """
            bridge = self._bridge

            async def event_generator():
                while True:
                    payload = bridge.consume_sse()
                    if payload:
                        yield f"data: {json.dumps(payload)}\n\n"
                    else:
                        yield ": heartbeat\n\n"
                        await asyncio.sleep(0.5)

            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
            )

        return router

    # ── Snapshot ────────────────────────────────────────────────

    def snapshot(self, trajectory_id: str) -> Dict[str, Any]:
        """Return current metric snapshot for a trajectory."""
        monitor = self._monitors.get(trajectory_id)
        if monitor is None:
            return {"error": f"No monitor for trajectory {trajectory_id}"}
        return monitor.snapshot()
