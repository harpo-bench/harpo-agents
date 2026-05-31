#!/usr/bin/env python3
"""
HARPO Self-Evolution Cycle Demo

Demonstrates EvolutionTracker comparing three trajectory versions of the same
agent task — simulating a Hive self-evolution cycle (v1 → v2 → v3).

v1: degraded run — high assumption density, tool errors, no reflections
v2: partially improved — errors fixed, some reflections
v3: optimised — clean trajectory, low assumptions, high recovery rate

Usage
-----
cd /home/anand/HARPO-D881
python scripts/demo_evolution.py
"""

from __future__ import annotations

import sys
import os
import time
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from harpo.core.schema import (
    AgentTrajectory, TrajectoryStep, StepType, StepOutcome,
    ToolCall, MemoryAccess, TrajectoryStatus,
)
from evolution.tracker import EvolutionTracker
from evolution.comparator import TrajectoryComparator
from harpo.observability.replay import TrajectoryReplayer
from harpo.observability.realtime import TrajectoryMonitor


# ── Trajectory builders ───────────────────────────────────────────────────────

def _step(traj_id: str, turn: int, idx: int,
          stype: StepType, outcome: StepOutcome,
          text: str = "", tokens: int = 30, latency: float = 500,
          tool: ToolCall = None, mem: MemoryAccess = None) -> TrajectoryStep:
    return TrajectoryStep(
        step_id       = str(uuid.uuid4()),
        trajectory_id = traj_id,
        turn_number   = turn,
        step_index    = idx,
        step_type     = stype,
        outcome       = outcome,
        output_text   = text,
        raw_tokens    = tokens,
        timestamp     = time.time(),
        latency_ms    = latency,
        tool_call     = tool,
        memory_access = mem,
    )


def build_v1_degraded(user_intent: str) -> AgentTrajectory:
    """v1: high assumption density, tool error without recovery, no reflection."""
    t = AgentTrajectory(agent_id="hive-researcher", user_intent=user_intent,
                        status=TrajectoryStatus.FAILED)
    tid = t.trajectory_id

    t.add_step(_step(tid, 0, 0, StepType.THINK, StepOutcome.SUCCESS,
        "I probably need to search for this. I assume the database is up."))
    t.add_step(_step(tid, 0, 1, StepType.THINK, StepOutcome.SUCCESS,
        "I think the API likely returns JSON. I'll probably just parse it."))
    t.add_step(_step(tid, 1, 2, StepType.TOOL_CALL, StepOutcome.FAILURE,
        text="search(query='climate')",
        tool=ToolCall(name="search", arguments={"query": "climate"},
                      error="ConnectionTimeout")))
    # No recovery step — just moves on
    t.add_step(_step(tid, 1, 3, StepType.THINK, StepOutcome.SUCCESS,
        "I assume the search worked. Let me continue as if it did."))
    t.add_step(_step(tid, 2, 4, StepType.TOOL_CALL, StepOutcome.FAILURE,
        text="parse_json(data=None)",
        tool=ToolCall(name="parse_json", arguments={"data": None},
                      error="NullPointerError")))
    t.add_step(_step(tid, 2, 5, StepType.RESPONSE, StepOutcome.FAILURE,
        "I believe the answer is: mitigation is probably possible."))
    return t


def build_v2_partial(user_intent: str) -> AgentTrajectory:
    """v2: tool errors fixed, basic recovery, some reflection but not behaviour-changing."""
    t = AgentTrajectory(agent_id="hive-researcher", user_intent=user_intent,
                        status=TrajectoryStatus.COMPLETED)
    tid = t.trajectory_id

    t.add_step(_step(tid, 0, 0, StepType.THINK, StepOutcome.SUCCESS,
        "I need to research climate mitigation. Starting with a web search."))
    t.add_step(_step(tid, 1, 1, StepType.TOOL_CALL, StepOutcome.FAILURE,
        text="search(query='climate mitigation 2024')",
        tool=ToolCall(name="search", arguments={"query": "climate mitigation 2024"},
                      error="RateLimited")))
    t.add_step(_step(tid, 1, 2, StepType.RECOVERY, StepOutcome.RETRY,
        "Tool failed — retrying with backoff"))
    t.add_step(_step(tid, 1, 3, StepType.TOOL_CALL, StepOutcome.SUCCESS,
        text="search succeeded",
        tool=ToolCall(name="search", arguments={"query": "climate mitigation 2024"},
                      result="Carbon capture, renewables, reforestation")))
    t.add_step(_step(tid, 2, 4, StepType.REFLECTION, StepOutcome.RETRY,
        "Judge asked for more detail — expanding answer"))
    # Reflection fires but next THINK is nearly identical (not effective)
    t.add_step(_step(tid, 2, 5, StepType.THINK, StepOutcome.SUCCESS,
        "More detail: carbon capture removes CO2"))
    t.add_step(_step(tid, 3, 6, StepType.MEMORY_READ, StepOutcome.SUCCESS,
        mem=MemoryAccess(operation="read", key="climate_cache", value="cached",
                         hit=True, relevance_score=0.72)))
    t.add_step(_step(tid, 3, 7, StepType.RESPONSE, StepOutcome.SUCCESS,
        "Top strategies: 1) Renewables 2) Carbon capture 3) Reforestation"))
    return t


def build_v3_optimised(user_intent: str) -> AgentTrajectory:
    """v3: clean trajectory — verified assumptions, effective reflections, high memory use."""
    t = AgentTrajectory(agent_id="hive-researcher", user_intent=user_intent,
                        status=TrajectoryStatus.COMPLETED)
    tid = t.trajectory_id

    t.add_step(_step(tid, 0, 0, StepType.THINK, StepOutcome.SUCCESS,
        "I will search for climate mitigation strategies and verify each claim before reporting."))
    t.add_step(_step(tid, 1, 1, StepType.TOOL_CALL, StepOutcome.SUCCESS,
        tool=ToolCall(name="web_search",
                      arguments={"query": "climate mitigation strategies 2024 quantitative"},
                      result="IEA: renewables 4.5 GtCO2/yr, CCS 1.5 GtCO2/yr, reforestation 0.8 GtCO2/yr")))
    t.add_step(_step(tid, 1, 2, StepType.MEMORY_WRITE, StepOutcome.SUCCESS,
        mem=MemoryAccess(operation="write", key="sts_2024", value="IEA data cached",
                         hit=True, relevance_score=0.95)))
    t.add_step(_step(tid, 2, 3, StepType.TOOL_CALL, StepOutcome.SUCCESS,
        tool=ToolCall(name="verify_source",
                      arguments={"url": "iea.org/2024"},
                      result="Source verified — peer reviewed")))
    t.add_step(_step(tid, 2, 4, StepType.REFLECTION, StepOutcome.SUCCESS,
        "I should present strategies ranked by CO2 impact for clarity"))
    t.add_step(_step(tid, 2, 5, StepType.THINK, StepOutcome.SUCCESS,
        "Revised: rank by impact descending — renewables first"))
    t.add_step(_step(tid, 3, 6, StepType.MEMORY_READ, StepOutcome.SUCCESS,
        mem=MemoryAccess(operation="read", key="sts_2024", value="IEA data",
                         hit=True, relevance_score=0.95)))
    t.add_step(_step(tid, 3, 7, StepType.TOOL_CALL, StepOutcome.SUCCESS,
        tool=ToolCall(name="summarise",
                      arguments={"mode": "executive"},
                      result="Ranked summary ready")))
    t.add_step(_step(tid, 4, 8, StepType.RESPONSE, StepOutcome.SUCCESS,
        text=(
            "Top 3 Climate Mitigation Strategies (IEA 2024):\n"
            "1. Renewable Energy — 4.5 GtCO2/yr by 2035\n"
            "2. Carbon Capture — 1.5 GtCO2/yr by 2050\n"
            "3. Reforestation — 0.8 GtCO2/yr\n"
            "Source: IEA World Energy Outlook 2024 (verified)."
        ), tokens=95))
    return t


# ── Display helpers ───────────────────────────────────────────────────────────

def _bar(v: float, w: int = 25) -> str:
    n = int(max(0, min(1, v)) * w)
    return "█" * n + "░" * (w - n)


def _arrow(delta: float) -> str:
    if delta > 0.02:
        return f"\033[32m▲ +{delta:.4f}\033[0m"
    if delta < -0.02:
        return f"\033[31m▼  {delta:.4f}\033[0m"
    return f"\033[33m─  {delta:+.4f}\033[0m"


def main() -> None:
    task = "Research and summarise top 3 climate change mitigation strategies"
    print(f"\n{'='*65}")
    print(f"  HARPO Self-Evolution Cycle Demo")
    print(f"  Task: {task}")
    print(f"{'='*65}\n")

    v1 = build_v1_degraded(task)
    v2 = build_v2_partial(task)
    v3 = build_v3_optimised(task)

    tracker = EvolutionTracker()
    tracker.add_cycle("v1-degraded",  v1)
    tracker.add_cycle("v2-partial",   v2)
    tracker.add_cycle("v3-optimised", v3)

    # ── Scores table ──────────────────────────────────────────
    print("Cycle scores:")
    rows = tracker.scores_table()
    header = f"  {'Cycle':<15} {'Overall':>8} "
    print(header)
    print("  " + "─" * 60)
    for row in rows:
        print(f"  {row['cycle']:<15} {row['overall']:>8.4f}  {_bar(row['overall'])}")

    # ── Comparison ────────────────────────────────────────────
    print(f"\nCycle-by-cycle comparison:")
    comparisons = tracker.compare_all()
    for comp in comparisons:
        print(f"\n  {comp.from_label}  →  {comp.to_label}   "
              f"overall: {_arrow(comp.overall_delta)}")
        for dim, delta in sorted(comp.dimension_deltas.items(), key=lambda x: x[1]):
            print(f"    {dim:<35} {_arrow(delta)}")
        if comp.regressions:
            print(f"  ⚠ Regressions: {comp.regressions}")
        if comp.improvements:
            print(f"  ✓ Improvements: {comp.improvements}")

    # ── Improvement summary ───────────────────────────────────
    print(f"\nNet improvement (v1 → v3), avg delta per dimension:")
    summary = tracker.improvement_summary()
    for dim, avg in sorted(summary.items(), key=lambda x: -x[1]):
        print(f"  {dim:<35} {_arrow(avg)}")

    # ── Regression alerts ─────────────────────────────────────
    alerts = tracker.detect_regressions(threshold=0.05)
    if alerts:
        print(f"\nRegression alerts ({len(alerts)}):")
        for a in alerts:
            print(f"  [{a.severity.upper()}] {a.from_label}→{a.to_label}: "
                  f"{a.dimension} dropped {a.delta:.4f}")
    else:
        print("\nNo regression alerts — all dimensions neutral or improved.")

    # ── Replay v3 through monitor ─────────────────────────────
    print(f"\nReplaying v3 trajectory through TrajectoryMonitor...")
    monitor  = TrajectoryMonitor(v3.trajectory_id)
    replayer = TrajectoryReplayer(monitor=monitor, speed=0)
    events   = replayer.replay(v3)
    snap     = replayer.monitor_snapshot()
    print(f"  Replayed {len(events)} steps. Live metrics:")
    for k, v in snap.get("metrics", {}).items():
        val = f"{v:.4f}" if isinstance(v, float) else str(v)
        print(f"    {k}: {val}")

    # ── HTML diff ─────────────────────────────────────────────
    print(f"\nGenerating HTML diff (v1 → v3)...")
    comparator = TrajectoryComparator()
    diff = comparator.compare(v1, v3)
    html = comparator.to_html(diff)
    out_path = os.path.join(os.path.dirname(__file__), "..", "evolution_diff_v1_v3.html")
    with open(out_path, "w") as f:
        f.write(f"<html><body><h2>v1 → v3 Trajectory Diff</h2>{html}</body></html>")
    print(f"  Saved to: {os.path.abspath(out_path)}")

    print(f"\n{'='*65}")
    print("  Demo complete.")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
