"""
HARPO Open-Hive Log Reader

Converts Hive's three-level JSONL runtime logs into an AgentTrajectory
for post-hoc evaluation (no live EventBus required).

Log file layout (written by Hive's RuntimeLogger):
    {run_dir}/summary.jsonl     — L1 RunSummaryLog  (one record per run)
    {run_dir}/details.jsonl     — L2 NodeDetail     (one record per node)
    {run_dir}/steps.jsonl       — L3 NodeStepLog    (one record per step/iteration)

Usage
-----
from harpo.adapters.open_hive.log_reader import HiveLogReader

traj = HiveLogReader().read_run("/path/to/run/dir", user_intent="...")
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from harpo.core.schema import (
    AgentTrajectory,
    MemoryAccess,
    StepOutcome,
    StepType,
    ToolCall,
    TrajectoryStatus,
    TrajectoryStep,
)


class HiveLogReader:
    """
    Parse Hive L1/L2/L3 JSONL files into an AgentTrajectory.

    All three files are optional: the reader builds what it can from
    whatever is present, so partial logs (e.g. from a crashed run) still
    produce a valid trajectory.
    """

    def read_run(
        self,
        run_dir: str,
        user_intent: str = "",
        agent_id: str = "",
    ) -> AgentTrajectory:
        path = Path(run_dir)
        summary = self._load_jsonl(path / "summary.jsonl")
        steps   = self._load_jsonl(path / "steps.jsonl")

        l1 = summary[0] if summary else {}

        traj = AgentTrajectory(
            trajectory_id  = l1.get("run_id", str(uuid.uuid4())),
            agent_id       = agent_id or l1.get("agent_id", "hive-agent"),
            task_description = user_intent,
            user_intent    = user_intent,
            status         = self._parse_status(l1.get("status", "")),
            metadata       = {k: v for k, v in l1.items()
                              if k not in ("run_id", "agent_id", "status")},
        )

        turn_idx   = 0
        step_index = 0

        for raw in steps:
            parsed = self._parse_node_step(raw, traj.trajectory_id, step_index, turn_idx)
            for step in parsed:
                traj.add_step(step)
                step_index += 1
            # Each NodeStepLog with an LLM response is one turn
            if raw.get("llm_text"):
                turn_idx += 1

        traj.ended_at = time.time()
        return traj

    # ── Helpers ──────────────────────────────────────────────────

    def _parse_node_step(
        self,
        raw: Dict[str, Any],
        traj_id: str,
        base_idx: int,
        turn: int,
    ) -> list[TrajectoryStep]:
        steps: list[TrajectoryStep] = []
        node_id    = raw.get("node_id", "")
        latency_ms = float(raw.get("latency_ms", 0))
        verdict    = raw.get("verdict", "")
        llm_text   = raw.get("llm_text", "")
        error      = raw.get("error", "")
        is_partial = raw.get("is_partial", False)

        # THINK step — LLM text output
        if llm_text:
            steps.append(TrajectoryStep(
                trajectory_id = traj_id,
                turn_number   = turn,
                step_index    = base_idx + len(steps),
                step_type     = StepType.THINK,
                outcome       = StepOutcome.FAILURE if error else StepOutcome.SUCCESS,
                output_text   = llm_text,
                raw_tokens    = raw.get("output_tokens", 0),
                latency_ms    = latency_ms,
            ))

        # TOOL_CALL + OBSERVATION steps
        for tc_raw in raw.get("tool_calls", []):
            has_error = tc_raw.get("is_error", False)
            tc = ToolCall(
                name       = tc_raw.get("tool_name", ""),
                arguments  = tc_raw.get("tool_input", {}),
                result     = tc_raw.get("result", ""),
                error      = tc_raw.get("result", "") if has_error else None,
                latency_ms = tc_raw.get("duration_s", 0) * 1000,
            )
            steps.append(TrajectoryStep(
                trajectory_id = traj_id,
                turn_number   = turn,
                step_index    = base_idx + len(steps),
                step_type     = StepType.TOOL_CALL,
                outcome       = StepOutcome.FAILURE if has_error else StepOutcome.SUCCESS,
                output_text   = tc.result or "",
                tool_call     = tc,
                latency_ms    = tc.latency_ms,
            ))

        # REFLECTION / RESPONSE from verdict
        if verdict in ("RETRY", "ESCALATE"):
            steps.append(TrajectoryStep(
                trajectory_id = traj_id,
                turn_number   = turn,
                step_index    = base_idx + len(steps),
                step_type     = StepType.REFLECTION,
                outcome       = StepOutcome.RETRY if verdict == "RETRY" else StepOutcome.FAILURE,
                output_text   = raw.get("verdict_feedback", ""),
            ))
        elif verdict == "ACCEPT":
            steps.append(TrajectoryStep(
                trajectory_id = traj_id,
                turn_number   = turn,
                step_index    = base_idx + len(steps),
                step_type     = StepType.RESPONSE,
                outcome       = StepOutcome.SUCCESS,
                output_text   = llm_text,
            ))

        # Error step
        if error and not steps:
            steps.append(TrajectoryStep(
                trajectory_id = traj_id,
                turn_number   = turn,
                step_index    = base_idx + len(steps),
                step_type     = StepType.RECOVERY,
                outcome       = StepOutcome.FAILURE,
                output_text   = error,
            ))

        return steps

    @staticmethod
    def _parse_status(status: str) -> TrajectoryStatus:
        mapping = {
            "success": TrajectoryStatus.COMPLETED,
            "failure": TrajectoryStatus.FAILED,
            "degraded": TrajectoryStatus.FAILED,
        }
        return mapping.get(status.lower(), TrajectoryStatus.COMPLETED)

    @staticmethod
    def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
        if not path.exists():
            return []
        records = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return records
