"""
HARPO-Open Trajectory Evaluation API

FastAPI routes exposing the evaluation pipeline over HTTP.
Designed to integrate with Open-Hive agent execution infrastructure.

Endpoints:
  POST   /trajectories/                 submit a completed trajectory for evaluation
  GET    /trajectories/{id}/scores      retrieve evaluation scores
  GET    /trajectories/{id}/failure     retrieve failure report
  POST   /trajectories/compare          compare two trajectories head-to-head
  POST   /trajectories/batch            evaluate a list of trajectories
  GET    /trajectories/{id}/stream      SSE stream of live metrics
  POST   /ingest/step                   ingest a single live step
  GET    /agents/{id}/report            aggregate quality report for an agent
  GET    /benchmarks/{suite}/leaderboard  benchmark leaderboard
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

from ..trajectory.schema import (
    AgentTrajectory, StepType, StepOutcome,
    TrajectoryStep, ToolCall, MemoryAccess,
    AssumptionRecord, TrajectoryStatus,
)
from ..trajectory.pipeline import TrajectoryEvaluator
from ..trajectory.multi_agent import MultiAgentEvaluator
from ..observability.realtime import TrajectoryMonitor, ObservabilityBridge

import time
import asyncio


# ─── in-memory stores (replace with DB adapter for production) ────
_trajectories: Dict[str, AgentTrajectory] = {}
_monitors: Dict[str, TrajectoryMonitor]  = {}
_bridge = ObservabilityBridge()
_bridge.enable_json_sink("/tmp/harpo_events.jsonl")

_evaluator      = TrajectoryEvaluator()
_multi_evaluator = MultiAgentEvaluator()


# ─── Pydantic request/response models ───────────────────────────

if HAS_FASTAPI:

    class StepPayload(BaseModel):
        trajectory_id: str
        turn_number: int = 0
        step_index: int = 0
        step_type: str
        outcome: str = "success"
        input_text: str = ""
        output_text: str = ""
        raw_tokens: int = 0
        latency_ms: float = 0.0
        agent_id_local: Optional[str] = None
        tool_name: Optional[str] = None
        tool_args: Optional[Dict] = None
        tool_result: Optional[Any] = None
        tool_error: Optional[str] = None
        memory_key: Optional[str] = None
        memory_value: Optional[Any] = None
        memory_hit: bool = True
        memory_relevance: float = 0.0

    class TrajectoryPayload(BaseModel):
        agent_id: str
        agent_version: str = "unknown"
        task_id: str = ""
        task_description: str = ""
        user_intent: str = ""
        expected_outcome: Optional[str] = None
        agent_roles: List[str] = []
        steps: List[StepPayload] = []
        final_output: str = ""
        metadata: Dict = {}

    class CompareRequest(BaseModel):
        trajectory_a_id: str
        trajectory_b_id: str

    class BatchRequest(BaseModel):
        trajectory_ids: List[str]

    # ─── Router ─────────────────────────────────────────────────

    router = APIRouter(prefix="/trajectories", tags=["trajectories"])
    agents_router = APIRouter(prefix="/agents", tags=["agents"])
    ingest_router = APIRouter(prefix="/ingest", tags=["ingest"])

    # ── Submit trajectory ──────────────────────────────────────

    @router.post("/")
    async def submit_trajectory(payload: TrajectoryPayload) -> Dict:
        """
        Submit a completed trajectory for evaluation.
        Returns trajectory_id + full scores immediately.
        """
        traj = _build_trajectory(payload)
        _trajectories[traj.trajectory_id] = traj

        scores = _evaluator.evaluate(traj)
        failure = traj.failure_report
        multi = _multi_evaluator.evaluate(traj)

        return {
            "trajectory_id": traj.trajectory_id,
            "scores": scores.as_dict(),
            "failure_modes": [fm.value for fm in (failure.failure_modes if failure else [])],
            "failure_severity": failure.severity if failure else 0.0,
            "collaboration": {
                "adds_value": multi.collaboration_adds_value,
                "gain": multi.collaboration_value_gain,
                "orchestration_efficiency": multi.orchestration_efficiency,
            },
        }

    # ── Scores ────────────────────────────────────────────────

    @router.get("/{trajectory_id}/scores")
    async def get_scores(trajectory_id: str) -> Dict:
        traj = _get_or_404(trajectory_id)
        if not traj.scores:
            _evaluator.evaluate(traj)
        return traj.scores.as_dict()

    # ── Failure report ────────────────────────────────────────

    @router.get("/{trajectory_id}/failure")
    async def get_failure(trajectory_id: str) -> Dict:
        traj = _get_or_404(trajectory_id)
        if not traj.failure_report:
            from ..trajectory.metrics import detect_failure_modes
            traj.failure_report = detect_failure_modes(traj)
        fr = traj.failure_report
        return {
            "failure_modes": [fm.value for fm in fr.failure_modes],
            "first_failure_turn": fr.first_failure_turn,
            "cascade_detected": fr.cascade_detected,
            "recovery_attempted": fr.recovery_attempted,
            "recovery_succeeded": fr.recovery_succeeded,
            "root_cause": fr.root_cause,
            "severity": fr.severity,
        }

    # ── Compare ───────────────────────────────────────────────

    @router.post("/compare")
    async def compare_trajectories(req: CompareRequest) -> Dict:
        a = _get_or_404(req.trajectory_a_id)
        b = _get_or_404(req.trajectory_b_id)
        comparison = _evaluator.compare(a, b)
        return {
            "winner": comparison.winner,
            "delta_scores": comparison.delta_scores,
            "per_dimension_winner": comparison.per_dimension_winner,
            "narrative": comparison.narrative,
        }

    # ── Batch eval ────────────────────────────────────────────

    @router.post("/batch")
    async def batch_evaluate(req: BatchRequest) -> Dict:
        trajs = [_get_or_404(tid) for tid in req.trajectory_ids]
        all_scores = _evaluator.evaluate_batch(trajs)
        report = _evaluator.aggregate_report(trajs)
        return {
            "per_trajectory": {
                t.trajectory_id: s.as_dict()
                for t, s in zip(trajs, all_scores)
            },
            "aggregate": report,
        }

    # ── SSE live stream ───────────────────────────────────────

    @router.get("/{trajectory_id}/stream")
    async def stream_metrics(trajectory_id: str) -> StreamingResponse:
        """Server-Sent Events stream of live metrics for a running trajectory."""

        async def event_generator():
            while True:
                payload = _bridge.consume_sse()
                if payload and payload.get("trajectory_id") == trajectory_id:
                    yield f"data: {json.dumps(payload)}\n\n"
                else:
                    await asyncio.sleep(0.1)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # ── Ingest live step ──────────────────────────────────────

    @ingest_router.post("/step")
    async def ingest_step(payload: StepPayload) -> Dict:
        """
        Ingest a single step from a running trajectory.
        Creates the trajectory if it doesn't exist yet.
        Emits live metrics via SSE bridge.
        """
        tid = payload.trajectory_id
        if tid not in _trajectories:
            _trajectories[tid] = AgentTrajectory(trajectory_id=tid)
            _monitors[tid] = TrajectoryMonitor(trajectory_id=tid)
            _monitors[tid].on_metric(_bridge.handle_metric)
            _monitors[tid].on_alert(_bridge.handle_alert)

        step = _step_from_payload(payload)
        _trajectories[tid].add_step(step)
        _monitors[tid].ingest(step)

        snapshot = _monitors[tid].snapshot()
        return {"status": "ok", "snapshot": snapshot["metrics"]}

    # ── Agent aggregate report ────────────────────────────────

    @agents_router.get("/{agent_id}/report")
    async def agent_report(agent_id: str) -> Dict:
        agent_trajs = [
            t for t in _trajectories.values()
            if t.agent_id == agent_id
        ]
        if not agent_trajs:
            raise HTTPException(status_code=404, detail=f"No trajectories for agent '{agent_id}'")
        return _evaluator.aggregate_report(agent_trajs)

    # ─── helpers ────────────────────────────────────────────────

    def _get_or_404(tid: str) -> AgentTrajectory:
        if tid not in _trajectories:
            raise HTTPException(status_code=404, detail=f"Trajectory '{tid}' not found")
        return _trajectories[tid]

    def _build_trajectory(payload: TrajectoryPayload) -> AgentTrajectory:
        traj = AgentTrajectory(
            agent_id=payload.agent_id,
            agent_version=payload.agent_version,
            task_id=payload.task_id,
            task_description=payload.task_description,
            user_intent=payload.user_intent,
            expected_outcome=payload.expected_outcome,
            agent_roles=payload.agent_roles,
            final_output=payload.final_output,
            metadata=payload.metadata,
            status=TrajectoryStatus.COMPLETED,
            ended_at=time.time(),
        )
        for sp in payload.steps:
            traj.add_step(_step_from_payload(sp))
        return traj

    def _step_from_payload(sp: StepPayload) -> TrajectoryStep:
        tool_call = None
        if sp.tool_name:
            tool_call = ToolCall(
                name=sp.tool_name,
                arguments=sp.tool_args or {},
                result=sp.tool_result,
                error=sp.tool_error,
                latency_ms=sp.latency_ms,
            )
        mem = None
        if sp.memory_key:
            op = "write" if sp.outcome == "success" and sp.memory_value is not None else "read"
            mem = MemoryAccess(
                operation=op,
                key=sp.memory_key,
                value=sp.memory_value,
                hit=sp.memory_hit,
                relevance_score=sp.memory_relevance,
            )
        return TrajectoryStep(
            trajectory_id=sp.trajectory_id,
            turn_number=sp.turn_number,
            step_index=sp.step_index,
            step_type=StepType(sp.step_type),
            outcome=StepOutcome(sp.outcome),
            input_text=sp.input_text,
            output_text=sp.output_text,
            raw_tokens=sp.raw_tokens,
            latency_ms=sp.latency_ms,
            tool_call=tool_call,
            memory_access=mem,
        )


def create_app():
    """Factory function — call this in your main.py."""
    if not HAS_FASTAPI:
        raise ImportError("fastapi is required: pip install fastapi uvicorn")
    from fastapi import FastAPI
    app = FastAPI(
        title="HARPO-Open Trajectory Evaluation API",
        description="Behavioral trajectory evaluation for long-horizon AI agents",
        version="1.0.0",
    )
    app.include_router(router)
    app.include_router(agents_router)
    app.include_router(ingest_router)
    return app
