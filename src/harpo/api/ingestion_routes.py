"""
HARPO Ingestion API Routes

Handles live event ingestion (REST + batch JSONL upload) and post-hoc
log ingestion from Hive JSONL files.

Routes
------
POST /v1/ingest/event           — push one GenericAgentEvent (live, non-streaming)
POST /v1/ingest/events          — push a batch of GenericAgentEvents
POST /v1/ingest/logs            — upload a JSONL log file for post-hoc ingestion
GET  /v1/evolution/{agent_id}   — evolution history for an agent
GET  /v1/leaderboard            — agent leaderboard by overall score
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

from harpo.core.events import GenericAgentEvent, GenericEventType
from harpo.core.schema import (
    AgentTrajectory, TrajectoryStatus,
)
from harpo.trajectory.pipeline import TrajectoryEvaluator


# ── Request / response models ────────────────────────────────────────────────

if FASTAPI_AVAILABLE:
    class EventPayload(BaseModel):
        """Single GenericAgentEvent submitted via REST."""
        trajectory_id: str
        agent_id:      str = "agent"
        event_type:    str = "think"
        text_output:   str = ""
        tokens:        int = 0
        latency_ms:    float = 0.0
        turn_number:   int = 0
        success:       bool = True
        tool_name:     Optional[str] = None
        tool_result:   Optional[str] = None
        tool_error:    Optional[str] = None

    class BatchEventPayload(BaseModel):
        events: List[EventPayload]


# ── In-memory trajectory store (replace with DB in production) ───────────────

_TRAJECTORIES: Dict[str, AgentTrajectory] = {}
_EVALUATOR = TrajectoryEvaluator()


def _get_or_create(trajectory_id: str, agent_id: str = "agent") -> AgentTrajectory:
    if trajectory_id not in _TRAJECTORIES:
        _TRAJECTORIES[trajectory_id] = AgentTrajectory(
            trajectory_id = trajectory_id,
            agent_id      = agent_id,
        )
    return _TRAJECTORIES[trajectory_id]


def _payload_to_event(p: "EventPayload") -> GenericAgentEvent:
    from harpo.core.events import GenericToolCall
    tc = None
    if p.tool_name:
        tc = GenericToolCall(
            tool_name = p.tool_name,
            arguments = {},
            result    = p.tool_result or "",
            error     = p.tool_error,
        )
    return GenericAgentEvent(
        event_id    = str(uuid.uuid4()),
        agent_id    = p.agent_id,
        run_id      = p.trajectory_id,
        event_type  = GenericEventType(p.event_type),
        timestamp   = time.time(),
        turn_number = p.turn_number,
        text_output = p.text_output,
        tokens      = p.tokens,
        latency_ms  = p.latency_ms,
        success     = p.success,
        tool_call   = tc,
    )


# ── Router factory ───────────────────────────────────────────────────────────

def create_ingestion_router(
    stream_server: Optional[Any] = None,   # HarpoStreamServer if wired
) -> "APIRouter":
    if not FASTAPI_AVAILABLE:
        raise ImportError("fastapi is required for ingestion routes.")

    router = APIRouter()

    @router.post("/ingest/event")
    async def ingest_event(payload: "EventPayload"):
        """Push one GenericAgentEvent into a live or buffered trajectory."""
        try:
            traj = _get_or_create(payload.trajectory_id, payload.agent_id)
            evt  = _payload_to_event(payload)

            # Convert to TrajectoryStep via HarpoPlugin-like mapping
            from harpo.sdk.plugin import HarpoPlugin, _EVENT_TO_STEP
            from harpo.core.schema import TrajectoryStep, StepOutcome
            step_type = _EVENT_TO_STEP.get(evt.event_type)
            if step_type:
                step = TrajectoryStep(
                    trajectory_id = payload.trajectory_id,
                    turn_number   = payload.turn_number,
                    step_index    = len(traj.steps),
                    step_type     = step_type,
                    outcome       = StepOutcome.SUCCESS if payload.success else StepOutcome.FAILURE,
                    output_text   = payload.text_output,
                    raw_tokens    = payload.tokens,
                    latency_ms    = payload.latency_ms,
                )
                traj.add_step(step)
                if stream_server:
                    stream_server.ingest_step(payload.trajectory_id, step)

            return {"status": "ok", "steps": len(traj.steps)}
        except Exception as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @router.post("/ingest/events")
    async def ingest_batch(payload: "BatchEventPayload"):
        """Push a batch of events in one request."""
        results = []
        for ep in payload.events:
            await ingest_event(ep)
            results.append(ep.trajectory_id)
        return {"status": "ok", "ingested": len(results)}

    @router.post("/ingest/logs")
    async def ingest_logs(file: "UploadFile", user_intent: str = ""):
        """
        Upload a Hive JSONL log file for post-hoc trajectory ingestion.
        Expects the steps.jsonl format from Hive's RuntimeLogger L3.
        """
        try:
            from harpo.adapters.open_hive.log_reader import HiveLogReader
            import tempfile, os
            content = await file.read()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
                tmp.write(content)
                tmp_path = tmp.name

            # HiveLogReader expects a directory — write to temp dir
            import os, pathlib
            run_dir = pathlib.Path(tmp_path).parent / f"run_{uuid.uuid4().hex[:8]}"
            run_dir.mkdir(exist_ok=True)
            (run_dir / "steps.jsonl").write_bytes(content)

            reader = HiveLogReader()
            traj = reader.read_run(str(run_dir), user_intent=user_intent)
            _TRAJECTORIES[traj.trajectory_id] = traj
            scores = _EVALUATOR.evaluate(traj)

            os.unlink(tmp_path)
            return {
                "trajectory_id": traj.trajectory_id,
                "steps":         len(traj.steps),
                "overall_score": round(scores.overall, 4),
            }
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @router.get("/evolution/{agent_id}")
    async def evolution_history(agent_id: str):
        """List all trajectories for an agent, ordered by start time."""
        agent_trajs = [
            {
                "trajectory_id": t.trajectory_id,
                "started_at":    t.started_at,
                "status":        str(t.status),
                "steps":         len(t.steps),
            }
            for t in _TRAJECTORIES.values()
            if t.agent_id == agent_id
        ]
        agent_trajs.sort(key=lambda x: x["started_at"])
        return {"agent_id": agent_id, "cycles": agent_trajs}

    @router.get("/leaderboard")
    async def leaderboard():
        """Return agents ranked by overall trajectory score."""
        rows = []
        for traj in _TRAJECTORIES.values():
            try:
                scores = _EVALUATOR.evaluate(traj)
                rows.append({
                    "agent_id":   traj.agent_id,
                    "traj_id":    traj.trajectory_id,
                    "overall":    round(scores.overall, 4),
                    "steps":      len(traj.steps),
                })
            except Exception:
                pass
        rows.sort(key=lambda r: r["overall"], reverse=True)
        return {"leaderboard": rows}

    return router
