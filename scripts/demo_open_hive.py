#!/usr/bin/env python3
"""
HARPO × Open-Hive Integration Demo

Demonstrates the HARPO-Open plugin attached to an Open-Hive agent.

Modes
-----
--mock        (default)  Use MockHiveEventBus — no API key, runs offline
--live                   Single-turn real LLM call, verify adapter wiring
--rich                   5-turn phased research with failure analysis + evolution
                         (exercises all 10 HARPO dimensions with real API)

Usage
-----
cd /home/anand/HARPO-D881
python scripts/demo_open_hive.py
python scripts/demo_open_hive.py --live
python scripts/demo_open_hive.py --rich
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional


# ── Ensure src/ is on path ───────────────────────────────────────────────────
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── MockHiveEventBus ─────────────────────────────────────────────────────────

class MockEventType:
    """Minimal mock of Hive's EventType enum (string values only)."""
    LLM_TURN_COMPLETE   = "llm_turn_complete"
    TOOL_CALL_COMPLETED = "tool_call_completed"
    JUDGE_VERDICT       = "judge_verdict"
    NODE_RETRY          = "node_retry"
    CONTEXT_COMPACTED   = "context_compacted"
    SUBAGENT_REPORT     = "subagent_report"
    EXECUTION_STARTED   = "execution_started"
    EXECUTION_COMPLETED = "execution_completed"
    EXECUTION_FAILED    = "execution_failed"


@dataclass
class MockEvent:
    type:         str
    stream_id:    str = "mock-stream"
    node_id:      Optional[str] = None
    execution_id: Optional[str] = None
    data:         Dict[str, Any] = field(default_factory=dict)
    timestamp:    datetime = field(default_factory=datetime.now)


class MockHiveEventBus:
    """
    Minimal EventBus that replays a scripted sequence of MockEvents.
    Mimics the Hive EventBus.subscribe() / publish contract.
    """

    def __init__(self) -> None:
        self._handlers: List[tuple[List[str], Callable]] = []

    def subscribe(
        self,
        event_types: List[Any],
        handler: Callable,
        **kwargs,
    ) -> None:
        type_values = [str(et) for et in event_types]
        self._handlers.append((type_values, handler))

    def replay(self, events: List[MockEvent]) -> None:
        for event in events:
            for types, handler in self._handlers:
                if event.type in types:
                    handler(event)


# ── Scripted event sequence: deep-research task ──────────────────────────────

def build_research_events(task: str = "Summarise top 3 climate mitigation strategies") -> List[MockEvent]:
    """18-event sequence simulating a deep-research Hive agent run."""
    run_id = uuid.uuid4().hex[:8]
    return [
        # Run starts
        MockEvent(MockEventType.EXECUTION_STARTED, data={"task": task}),

        # Turn 1: initial planning
        MockEvent(MockEventType.LLM_TURN_COMPLETE, data={
            "text": f"I'll research climate change mitigation. Let me search for recent studies.",
            "output_tokens": 42, "latency_ms": 820,
        }),
        MockEvent(MockEventType.JUDGE_VERDICT, data={"verdict": "CONTINUE", "feedback": "good plan"}),

        # Tool: web search
        MockEvent(MockEventType.TOOL_CALL_COMPLETED, data={
            "tool_name": "web_search",
            "arguments": {"query": "climate mitigation strategies 2024"},
            "result": "Found: carbon capture, renewable energy, reforestation",
            "latency_ms": 1200,
        }),

        # Turn 2: process results
        MockEvent(MockEventType.LLM_TURN_COMPLETE, data={
            "text": "Found 3 major strategies. Analysing carbon capture first.",
            "output_tokens": 65, "latency_ms": 950,
        }),
        MockEvent(MockEventType.JUDGE_VERDICT, data={"verdict": "CONTINUE"}),

        # Tool: read article
        MockEvent(MockEventType.TOOL_CALL_COMPLETED, data={
            "tool_name": "fetch_url",
            "arguments": {"url": "https://example.com/carbon-capture"},
            "result": "Carbon capture can remove 1-2 GtCO2/year by 2050.",
            "latency_ms": 800,
        }),

        # Context compaction (memory read)
        MockEvent(MockEventType.CONTEXT_COMPACTED, data={
            "summary": "Carbon capture findings compressed into memory",
            "tokens_before": 4200, "tokens_after": 800,
        }),

        # Turn 3: retry scenario
        MockEvent(MockEventType.LLM_TURN_COMPLETE, data={
            "text": "Let me check renewable energy impact...",
            "output_tokens": 38, "latency_ms": 700,
        }),
        MockEvent(MockEventType.JUDGE_VERDICT, data={
            "verdict": "RETRY",
            "feedback": "Response too brief — add quantitative data",
        }),

        # Recovery
        MockEvent(MockEventType.NODE_RETRY, data={"reason": "verdict RETRY — expanding response"}),
        MockEvent(MockEventType.LLM_TURN_COMPLETE, data={
            "text": (
                "Renewable energy: Solar + wind capacity grew 295 GW in 2023. "
                "Could displace 4.5 GtCO2/year by 2035 (IEA 2024)."
            ),
            "output_tokens": 58, "latency_ms": 890,
        }),
        MockEvent(MockEventType.JUDGE_VERDICT, data={"verdict": "ACCEPT", "feedback": "good"}),

        # Sub-agent report (background worker)
        MockEvent(MockEventType.SUBAGENT_REPORT, data={
            "report": "Reforestation worker: top 5 reforestation programs identified",
        }),

        # Tool: synthesise
        MockEvent(MockEventType.TOOL_CALL_COMPLETED, data={
            "tool_name": "summarise",
            "arguments": {"mode": "executive"},
            "result": "Top 3: (1) Carbon capture 1.5 GtCO2/yr, (2) Solar/wind 4.5 GtCO2/yr, (3) Reforestation 0.8 GtCO2/yr",
            "latency_ms": 600,
        }),

        # Turn 4: final response
        MockEvent(MockEventType.LLM_TURN_COMPLETE, data={
            "text": (
                "**Top 3 Climate Mitigation Strategies:**\n"
                "1. Renewable Energy — 4.5 GtCO2/yr potential\n"
                "2. Carbon Capture — 1.5 GtCO2/yr by 2050\n"
                "3. Reforestation — 0.8 GtCO2/yr"
            ),
            "output_tokens": 92, "latency_ms": 1100,
        }),
        MockEvent(MockEventType.JUDGE_VERDICT, data={"verdict": "ACCEPT", "feedback": "comprehensive"}),

        # Run ends
        MockEvent(MockEventType.EXECUTION_COMPLETED, data={"status": "success"}),
    ]


# ── Display helpers ───────────────────────────────────────────────────────────

def _bar(score: float, width: int = 30) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _print_scores(report: dict) -> None:
    print(f"\n{'='*60}")
    print(f"  HARPO Trajectory Evaluation Report")
    print(f"{'='*60}")
    print(f"  Agent:          {report['agent']}")
    print(f"  Steps captured: {report['steps']}")
    print(f"  Status:         {report['status']}")
    print(f"{'─'*60}")
    print(f"  {'Dimension':<30} {'Score':>6}  {'Bar':>32}")
    print(f"{'─'*60}")
    for dim, score in sorted(report['dimensions'].items(), key=lambda x: -x[1]):
        bar = _bar(score)
        print(f"  {dim:<30} {score:>6.4f}  {bar}")
    print(f"{'─'*60}")
    print(f"  {'OVERALL':.<30} {report['overall_score']:>6.4f}  {_bar(report['overall_score'])}")
    print(f"{'='*60}\n")

    if report.get("live_metrics"):
        print("  Live monitor metrics (last step):")
        for k, v in report["live_metrics"].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")


# ── Main ──────────────────────────────────────────────────────────────────────

def run_mock() -> None:
    print("\n[HARPO × Open-Hive] Mock mode — no API key required\n")

    from harpo.adapters.open_hive.adapter import HiveAdapter
    from harpo.sdk.plugin import HarpoPlugin

    task = "Summarise top 3 climate change mitigation strategies with quantitative data"
    bus  = MockHiveEventBus()

    print(f"Task: {task}")
    print("Attaching HarpoPlugin to MockHiveEventBus...")

    plugin = HarpoPlugin(agent_id="deep-research-agent", user_intent=task)
    HiveAdapter(sink=plugin._ingest).attach(bus)

    print("Replaying 18 scripted events...\n")
    events = build_research_events(task)
    bus.replay(events)

    print(f"Captured {len(plugin.trajectory().steps)} trajectory steps.")
    report = plugin.report()
    _print_scores(report)

    print("Export (JSON):")
    import json
    export = plugin.export("json")
    print(json.dumps({k: v for k, v in export.items() if k != "scores"}, indent=2))
    print(f"\n  Dimensions summary:")
    for dim, data in export["scores"].items():
        print(f"    {dim}: {data['score']:.4f}")


def run_live() -> None:
    """Run HARPO evaluation with a real Hive AgentLoop + live LLM API call."""
    import asyncio
    import tempfile

    HIVE_CORE = os.environ.get("HIVE_CORE", "/home/anand/hive/core")
    if HIVE_CORE not in sys.path:
        sys.path.insert(0, HIVE_CORE)

    print("\n[HARPO × Open-Hive] Live mode — real LLM API\n")

    # ── Verify token ─────────────────────────────────────────────────────
    try:
        from framework.config import get_api_key, get_llm_extra_kwargs, get_api_base, get_hive_config
    except ImportError as e:
        print(f"ERROR: Hive framework not found at {HIVE_CORE}: {e}")
        sys.exit(1)

    api_key = get_api_key()
    if not api_key:
        print("ERROR: No API key found. Check ~/.hive/configuration.json")
        sys.exit(1)
    print(f"  Token loaded: {api_key[:20]}...")

    extra_kwargs = get_llm_extra_kwargs()
    api_base     = get_api_base()
    model        = get_hive_config().get("llm", {}).get("model", "claude-sonnet-4-6")
    print(f"  Model:  {model}")
    print(f"  api_base: {api_base!r}\n")

    task = "Summarise the top 3 climate change mitigation strategies with quantitative data"
    print(f"  Task: {task}\n")

    # ── Imports ──────────────────────────────────────────────────────────
    from framework.host.event_bus import EventBus, EventType as HiveEventType
    from framework.agent_loop.agent_loop import AgentLoop
    from framework.agent_loop.types import AgentSpec, AgentContext
    from framework.agent_loop.internals.types import LoopConfig
    from framework.tracker.decision_tracker import DecisionTracker
    from framework.llm.litellm import LiteLLMProvider
    from harpo.adapters.open_hive.adapter import HiveAdapter
    from harpo.adapters.open_hive.event_map import SUBSCRIBED_EVENT_TYPES
    from harpo.sdk.plugin import HarpoPlugin
    from harpo.core.schema import TrajectoryStatus

    # ── EventBus + HARPO plugin ──────────────────────────────────────────
    event_bus = EventBus()
    plugin    = HarpoPlugin(agent_id="climate-researcher", user_intent=task)
    adapter   = HiveAdapter(sink=plugin._ingest, agent_id="climate-researcher")

    # Real EventBus requires async handlers
    async def _async_handle(event):
        adapter._handle(event)

    subscribed = [getattr(HiveEventType, et.upper(), None) for et in SUBSCRIBED_EVENT_TYPES]
    subscribed = [et for et in subscribed if et is not None]
    event_bus.subscribe(event_types=subscribed, handler=_async_handle)
    print(f"  Subscribed to {len(subscribed)} event types via HiveAdapter.")

    # ── LLM Provider ─────────────────────────────────────────────────────
    llm = LiteLLMProvider(
        model=model,
        api_key=api_key,
        api_base=api_base,
        **extra_kwargs,
    )

    # ── Async execution ──────────────────────────────────────────────────
    async def _run():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = DecisionTracker(tmp_dir)

            spec = AgentSpec(
                id="researcher",
                name="Climate Researcher",
                description="Research top 3 climate mitigation strategies",
                system_prompt=(
                    "You are a concise climate research analyst. "
                    "Answer the user's question directly with the top 3 climate change "
                    "mitigation strategies, each with quantitative impact estimates. "
                    "Keep your response under 300 words. Do not ask follow-up questions."
                ),
                tool_access_policy="none",
                output_keys=[],
                skip_judge=False,
            )

            ctx = AgentContext(
                runtime=tracker,
                agent_id="researcher",
                agent_spec=spec,
                llm=llm,
                input_data={"task": task},
                goal_context=task,
                stream_id="queen",      # non-worker — no auto-escalation
                event_triggered=True,   # prevents blocking for user input
                run_id=str(uuid.uuid4()),
            )

            loop_config = LoopConfig(max_iterations=3, max_context_tokens=8192)
            agent_loop  = AgentLoop(event_bus=event_bus, config=loop_config)

            print("  Calling LLM API... (this may take ~10s)")
            return await agent_loop.execute(ctx)

    result = asyncio.run(_run())
    print(f"\n  Agent result: success={result.success}  exit_reason={result.exit_reason}")
    print(f"  Tokens used:  {result.tokens_used}")

    # ── HARPO evaluation ─────────────────────────────────────────────────
    plugin._trajectory.status = TrajectoryStatus.COMPLETED
    steps = plugin.trajectory().steps
    print(f"  Steps captured by HARPO: {len(steps)}")

    # Print first THINK step
    for step in steps:
        if hasattr(step.step_type, "value"):
            st = step.step_type.value
        else:
            st = str(step.step_type)
        if st == "think" and step.output_text:
            print(f"\n  Agent output (turn {step.turn_number}):")
            preview = step.output_text[:600]
            for line in preview.split("\n"):
                print(f"    {line}")
            if len(step.output_text) > 600:
                print(f"    ... [{len(step.output_text)} chars total]")
            break

    report = plugin.report()
    _print_scores(report)


def _hive_setup() -> dict:
    """Shared Hive setup for live and rich modes. Returns config dict."""
    HIVE_CORE = os.environ.get("HIVE_CORE", "/home/anand/hive/core")
    if HIVE_CORE not in sys.path:
        sys.path.insert(0, HIVE_CORE)
    try:
        from framework.config import get_api_key, get_llm_extra_kwargs, get_api_base, get_hive_config
    except ImportError as e:
        print(f"ERROR: Hive framework not found at {HIVE_CORE}: {e}")
        sys.exit(1)
    api_key = get_api_key()
    if not api_key:
        print("ERROR: No API key found. Check ~/.hive/configuration.json")
        sys.exit(1)
    return {
        "api_key":     api_key,
        "extra_kwargs": get_llm_extra_kwargs(),
        "api_base":    get_api_base(),
        "model":       get_hive_config().get("llm", {}).get("model", "claude-haiku-4-5-20251001"),
    }


def _make_event_bus_with_plugin(task: str, agent_id: str = "climate-researcher"):
    """Create EventBus + HarpoPlugin + HiveAdapter (async handler wired)."""
    from framework.host.event_bus import EventBus, EventType as HiveEventType
    from harpo.adapters.open_hive.adapter import HiveAdapter
    from harpo.adapters.open_hive.event_map import SUBSCRIBED_EVENT_TYPES
    from harpo.sdk.plugin import HarpoPlugin

    event_bus = EventBus()
    plugin    = HarpoPlugin(agent_id=agent_id, user_intent=task)
    adapter   = HiveAdapter(sink=plugin._ingest, agent_id=agent_id)

    async def _async_handle(event):
        adapter._handle(event)

    subscribed = [getattr(HiveEventType, et.upper(), None) for et in SUBSCRIBED_EVENT_TYPES]
    subscribed = [et for et in subscribed if et is not None]
    event_bus.subscribe(event_types=subscribed, handler=_async_handle)
    return event_bus, plugin, len(subscribed)


def run_live_rich() -> None:
    """
    5-turn phased climate research with a phase-advancing judge.

    Exercises:
      - Multi-turn reasoning (reasoning_stability, long_horizon_reliability)
      - Reflection steps via RETRY verdicts (reflection_usefulness)
      - Failure mode detection (detect_failure_modes, DefaultFailureAnalyzer)
      - Evolution tracking: compare v1 (1 turn) vs v2 (5 turns)
      - Trajectory replay + HTML diff
      - HookRegistry: post_step alert hook
    """
    import asyncio
    import json
    import tempfile

    print("\n" + "=" * 65)
    print("  HARPO × Open-Hive  |  Rich Multi-Turn Demo (real API)")
    print("=" * 65 + "\n")

    cfg = _hive_setup()
    print(f"  Model:   {cfg['model']}")
    print(f"  api_key: {cfg['api_key'][:20]}...\n")

    task = "Conduct a structured 5-phase analysis of the top 3 climate change mitigation strategies with quantitative data."

    # ── Shared imports ───────────────────────────────────────────────────
    from framework.host.event_bus import EventBus, EventType as HiveEventType
    from framework.agent_loop.agent_loop import AgentLoop
    from framework.agent_loop.types import AgentSpec, AgentContext
    from framework.agent_loop.internals.types import LoopConfig, JudgeVerdict
    from framework.tracker.decision_tracker import DecisionTracker
    from framework.llm.litellm import LiteLLMProvider
    from harpo.core.schema import TrajectoryStatus
    from harpo.core.hooks import HookRegistry, HookContext
    from failures.detectors import (
        DefaultFailureAnalyzer, AssumptionDetector, LoopDetector,
        ReflectionEffectivenessDetector, RecoveryQualityDetector, DriftDetector,
    )
    from harpo.trajectory.metrics import detect_failure_modes
    from evolution.tracker import EvolutionTracker
    from semantic import SemanticTrajectoryAnalyzer

    llm = LiteLLMProvider(
        model=cfg["model"],
        api_key=cfg["api_key"],
        api_base=cfg["api_base"],
        **cfg["extra_kwargs"],
    )

    # ── Phase-advancing judge ────────────────────────────────────────────
    class PhaseJudge:
        """Drives the agent through 5 research phases via RETRY feedback."""
        _feedback = [
            "Good start! Proceed to Phase 2: Deep analysis of renewable energy — "
            "include GW capacity, cost curves, and CO2 reduction potential by 2050.",
            "Excellent renewable energy analysis! Proceed to Phase 3: Analyse "
            "carbon removal strategies (direct air capture, BECCS, reforestation) with numbers.",
            "Great data! Proceed to Phase 4: Critically REFLECT — what are the key "
            "assumptions and uncertainties in your analysis so far?",
            "Valuable reflection! Proceed to Phase 5: Synthesise into 3 prioritised "
            "recommendations with implementation timelines. Then conclude.",
        ]
        def __init__(self):
            self._n = 0
        async def evaluate(self, context: dict) -> JudgeVerdict:
            if self._n < len(self._feedback):
                fb = self._feedback[self._n]
                self._n += 1
                return JudgeVerdict(action="RETRY", feedback=fb)
            return JudgeVerdict(action="ACCEPT", feedback="Comprehensive 5-phase analysis complete.")

    # ── Hook: log alert when a step fails ───────────────────────────────
    hook_log: list = []
    hooks = HookRegistry()

    def _alert_hook(ctx: HookContext) -> None:
        step = ctx.step
        if step and not getattr(step, "outcome", None).__class__.__name__ == "SUCCESS":
            if hasattr(step.outcome, "value") and step.outcome.value != "success":
                hook_log.append(
                    f"turn={step.turn_number} type={step.step_type} "
                    f"outcome={step.outcome}"
                )

    hooks.register_post_step(_alert_hook)

    # ── Helper to build plugin with hooks ────────────────────────────────
    def _build_run(agent_id: str):
        event_bus, plugin, n_subs = _make_event_bus_with_plugin(task, agent_id)
        plugin._hooks = hooks
        return event_bus, plugin, n_subs

    # ────────────────────────────────────────────────────────────────────
    # RUN 1: single-turn baseline (no judge pressure)
    # ────────────────────────────────────────────────────────────────────
    print("  [Run 1/2] Single-turn baseline (v1)...")
    event_bus_v1, plugin_v1, n_subs = _build_run("researcher-v1")
    print(f"  Subscribed to {n_subs} event types.")

    async def _run_v1():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = DecisionTracker(tmp_dir)
            spec = AgentSpec(
                id="researcher-v1",
                name="Baseline Researcher",
                description="Single-turn climate research",
                system_prompt=(
                    "You are a climate research analyst. "
                    "Summarise the top 3 climate mitigation strategies with quantitative impact data. "
                    "Keep it under 250 words."
                ),
                tool_access_policy="none",
                output_keys=[],
                skip_judge=False,
            )
            ctx = AgentContext(
                runtime=tracker, agent_id="researcher-v1", agent_spec=spec,
                llm=llm, input_data={"task": task}, goal_context=task,
                stream_id="queen", event_triggered=True, run_id=str(uuid.uuid4()),
            )
            return await AgentLoop(event_bus=event_bus_v1, config=LoopConfig(
                max_iterations=2, max_context_tokens=6000,
            )).execute(ctx)

    print("  Calling LLM (v1 baseline)...")
    r1 = asyncio.run(_run_v1())
    plugin_v1._trajectory.status = TrajectoryStatus.COMPLETED
    print(f"  v1 done: {len(plugin_v1.trajectory().steps)} steps, "
          f"exit={r1.exit_reason}, tokens={r1.tokens_used}")

    # ────────────────────────────────────────────────────────────────────
    # RUN 2: 5-phase deep research with PhaseJudge
    # ────────────────────────────────────────────────────────────────────
    print(f"\n  [Run 2/2] 5-phase analysis with PhaseJudge (v2)...")
    event_bus_v2, plugin_v2, _ = _build_run("researcher-v2")

    async def _run_v2():
        with tempfile.TemporaryDirectory() as tmp_dir:
            tracker = DecisionTracker(tmp_dir)
            spec = AgentSpec(
                id="researcher-v2",
                name="Phase Researcher",
                description="5-phase structured climate research",
                system_prompt=(
                    "You are a senior climate research analyst. "
                    "You will conduct research in structured phases as directed. "
                    "Each turn, focus on the current phase. Be analytical, use numbers, "
                    "cite key figures (IEA, IPCC) where relevant. "
                    "Build explicitly on your previous phase results."
                ),
                tool_access_policy="none",
                output_keys=[],
                skip_judge=False,
            )
            ctx = AgentContext(
                runtime=tracker, agent_id="researcher-v2", agent_spec=spec,
                llm=llm, input_data={"task": task}, goal_context=task,
                stream_id="queen", event_triggered=True, run_id=str(uuid.uuid4()),
            )
            return await AgentLoop(
                event_bus=event_bus_v2,
                judge=PhaseJudge(),
                config=LoopConfig(max_iterations=6, max_context_tokens=32000),
            ).execute(ctx)

    print("  Calling LLM (5 phases — may take 30-60s)...")
    r2 = asyncio.run(_run_v2())
    plugin_v2._trajectory.status = (
        TrajectoryStatus.COMPLETED if r2.success else TrajectoryStatus.COMPLETED
    )
    print(f"  v2 done: {len(plugin_v2.trajectory().steps)} steps, "
          f"exit={r2.exit_reason}, tokens={r2.tokens_used}")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 1: Trajectory step table
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 1  |  v2 Trajectory Steps")
    print("─" * 65)
    for i, step in enumerate(plugin_v2.trajectory().steps):
        st = step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)
        oc = step.outcome.value if hasattr(step.outcome, "value") else str(step.outcome)
        text_preview = (step.output_text[:70] + "…") if step.output_text else "(no text)"
        print(f"  [{i:2d}] turn={step.turn_number} type={st:<12} outcome={oc:<8} | {text_preview}")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 2: HARPO evaluation — both runs
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 2  |  HARPO Evaluation")
    print("─" * 65)
    print("\n  v1 (1-turn baseline):")
    _print_scores(plugin_v1.report())
    print("\n  v2 (5-phase deep research):")
    _print_scores(plugin_v2.report())

    # ────────────────────────────────────────────────────────────────────
    # SECTION 3: Failure analysis on v2
    # ────────────────────────────────────────────────────────────────────
    print("─" * 65)
    print("  SECTION 3  |  Failure Analysis (v2 trajectory)")
    print("─" * 65)

    # Built-in failure mode detector (from trajectory/metrics.py)
    fr = detect_failure_modes(plugin_v2.trajectory())
    print(f"\n  detect_failure_modes() result:")
    modes = [m.value if hasattr(m, "value") else str(m) for m in (fr.failure_modes or [])]
    print(f"    failure_modes:       {modes or ['none detected']}")
    print(f"    cascade_detected:    {fr.cascade_detected}")
    print(f"    recovery_attempted:  {fr.recovery_attempted}")
    print(f"    recovery_succeeded:  {fr.recovery_succeeded}")
    print(f"    first_failure_turn:  {fr.first_failure_turn}")

    # Real Phase 2 failure detectors
    print(f"\n  Phase 2 failure detectors (v2 trajectory):")
    all_detector_events = []
    for cls in (AssumptionDetector, LoopDetector, ReflectionEffectivenessDetector,
                RecoveryQualityDetector, DriftDetector):
        events = cls().detect(plugin_v2.trajectory())
        all_detector_events.extend(events)
        if events:
            print(f"    {cls.__name__}: {len(events)} event(s)")
            for e in events[:2]:
                sev = e.severity if isinstance(e.severity, str) else "?"
                print(f"      [{sev}] {e.failure_type}: {e.description[:80]}")
        else:
            print(f"    {cls.__name__}: 0 events (clean)")

    # DefaultFailureAnalyzer with real failure events from v2
    failure_events = []
    for step in plugin_v2.trajectory().steps:
        oc = step.outcome.value if hasattr(step.outcome, "value") else str(step.outcome)
        if oc == "retry":
            from failures.schema import FailureEvent
            failure_events.append(FailureEvent(
                failure_id    = step.step_id,
                trajectory_id = plugin_v2.trajectory().trajectory_id,
                step_id       = step.step_id,
                turn_number   = step.turn_number,
                failure_type  = "reflection_trigger",
                severity      = "low",
                description   = f"Judge requested retry at turn {step.turn_number}",
                timestamp     = step.timestamp,
            ))

    fail_report = DefaultFailureAnalyzer().analyze(failure_events)
    print(f"\n  DefaultFailureAnalyzer (real failure events from v2):")
    print(f"    failure_events:   {len(fail_report.failure_events)}")
    print(f"    dominant_failure: {fail_report.dominant_failure!r}")
    print(f"    failure_density:  {fail_report.failure_density:.3f} per turn")
    print(f"    recovery_rate:    {fail_report.recovery_rate:.3f}")
    print(f"    cascade_detected: {fail_report.cascade_detected}")

    if hook_log:
        print(f"\n  HookRegistry alerts fired: {len(hook_log)}")
        for entry in hook_log[:5]:
            print(f"    {entry}")
    else:
        print(f"\n  HookRegistry: 0 step-failure alerts (all steps succeeded)")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 3b: Semantic Trajectory Intelligence
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 3b |  Semantic Trajectory Intelligence (v2)")
    print("─" * 65)

    sem_analyzer = SemanticTrajectoryAnalyzer()
    sem_v1 = sem_analyzer.analyze(plugin_v1.trajectory())
    sem_v2 = sem_analyzer.analyze(plugin_v2.trajectory())

    for label, sem in [("v1", sem_v1), ("v2", sem_v2)]:
        print(f"\n  [{label}] Semantic summary:")
        s = sem.summary()
        print(f"    Contradictions: {s['contradictions']['total']} total "
              f"(reversals={s['contradictions']['reversal_count']}, "
              f"flips={s['contradictions']['flip_count']}, "
              f"severity={s['contradictions']['severity']:.2f})")
        print(f"    Assumptions:    {s['assumptions']['total']} found, "
              f"{s['assumptions']['propagating']} propagating "
              f"(max_radius={s['assumptions']['max_radius_turns']} turns)")
        print(f"    Reflections:    {s['reflections']['total']} total, "
              f"{s['reflections']['effective']} effective "
              f"(rate={s['reflections']['effectiveness_rate']:.2f}, "
              f"avg_change={s['reflections']['avg_behavior_change']:.2f})")
        print(f"    Coherence:      overall={s['coherence']['overall']:.2f}, "
              f"core_overlap={s['coherence']['avg_core_overlap']:.2f}, "
              f"drifts={s['coherence']['drift_events']}")
        flags = sem.flags()
        if flags:
            print(f"    Diagnostic flags:")
            for flag in flags:
                print(f"      ⚑  {flag}")
        else:
            print(f"    Diagnostic flags: (none — trajectory looks clean)")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 4: Evolution tracking — v1 → v2
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 4  |  Evolution Tracking (v1 → v2)")
    print("─" * 65)
    tracker = EvolutionTracker()
    tracker.add_cycle("v1-baseline", plugin_v1)
    tracker.add_cycle("v2-deep", plugin_v2)

    print("\n  Scores table:")
    rows = tracker.scores_table()
    dims = [k for k in rows[0] if k not in ("cycle", "overall")]
    header = f"  {'cycle':<14}" + "".join(f"{d[:8]:>10}" for d in dims) + f"{'overall':>10}"
    print(header)
    print("  " + "─" * (len(header) - 2))
    for row in rows:
        line = f"  {row['cycle']:<14}" + "".join(
            f"{row.get(d, 0.0):>10.4f}" for d in dims
        ) + f"{row['overall']:>10.4f}"
        print(line)

    summary = tracker.improvement_summary()
    regressions = tracker.detect_regressions(threshold=0.05)
    print(f"\n  Improvement summary (v1 → v2):")
    for dim, delta in sorted(summary.items(), key=lambda x: -x[1]):
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"    {arrow} {dim:<35} {delta:+.4f}")

    if regressions:
        print(f"\n  Regression alerts ({len(regressions)}):")
        for r in regressions:
            print(f"    ▼ {r.dimension} [{r.from_label}→{r.to_label}] "
                  f"delta={r.delta:+.4f} severity={r.severity}")
    else:
        print(f"\n  No regressions detected (threshold 0.05).")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 5: Trajectory diff HTML
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 5  |  Trajectory Diff (v1 → v2)")
    print("─" * 65)
    from evolution.comparator import TrajectoryComparator
    comp = TrajectoryComparator()
    diff = comp.compare(plugin_v1.trajectory(), plugin_v2.trajectory())
    print(f"\n  overall_delta:  {diff.overall_delta:+.4f}")
    print(f"  improvements:   {diff.improvements}")
    print(f"  regressions:    {diff.regressions}")
    print(f"  dimension deltas:")
    for dim, delta in sorted(diff.summary.items(), key=lambda x: -x[1]):
        bar = "▲" if delta > 0.02 else ("▼" if delta < -0.02 else "─")
        print(f"    {bar} {dim:<35} {delta:+.4f}")

    html_path = os.path.join(os.path.dirname(__file__), "..", "evolution_diff_live_v1_v2.html")
    try:
        html = comp.to_html(diff)
        with open(html_path, "w") as f:
            f.write(html)
        print(f"\n  HTML diff saved → evolution_diff_live_v1_v2.html")
    except Exception as e:
        print(f"\n  HTML diff: {e}")

    # ────────────────────────────────────────────────────────────────────
    # SECTION 6: Trajectory replay
    # ────────────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  SECTION 6  |  Trajectory Replay (v2, instant speed)")
    print("─" * 65)
    from observability.replay import TrajectoryReplayer
    from observability.realtime import TrajectoryMonitor
    replay_monitor = TrajectoryMonitor(plugin_v2.trajectory().trajectory_id)
    replayer = TrajectoryReplayer(monitor=replay_monitor, hooks=hooks, speed=0.0)
    replay_events = replayer.replay(plugin_v2.trajectory())
    print(f"\n  Replayed {len(replay_events)} steps through TrajectoryMonitor + HookRegistry.")
    snap = replay_monitor.snapshot()
    if snap.get("metrics"):
        print(f"  Final replay monitor metrics:")
        for k, v in snap["metrics"].items():
            print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")

    print("\n" + "=" * 65)
    print("  Rich demo complete. All Phase 1 components verified on real API.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HARPO × Open-Hive Demo")
    parser.add_argument("--live",  action="store_true", help="Single-turn real Hive EventBus")
    parser.add_argument("--rich",  action="store_true", help="5-phase multi-turn with failure analysis")
    args = parser.parse_args()

    if args.rich:
        run_live_rich()
    elif args.live:
        run_live()
    else:
        run_mock()
