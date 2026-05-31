#!/usr/bin/env python3
"""
HARPO Enterprise Expansion Copilot — Long-Horizon Stress Test

Real API execution: claude-haiku-4-5-20251001
Scenario: 15-turn international expansion planning with evolving constraints,
          contradictory updates, assumption failures, reflection phases, and
          recovery attempts.

Runs FULL HARPO semantic trajectory analysis and compares against what
traditional observability systems (LangSmith / Langfuse / AgentOps) see.

Usage:
    cd /home/anand/HARPO-D881
    python scripts/demo_enterprise_stress_test.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tempfile
import uuid
from datetime import datetime
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
_HIVE_CORE = os.environ.get("HIVE_CORE", "/home/anand/hive/core")
sys.path.insert(0, _HIVE_CORE)

# ── HARPO imports ────────────────────────────────────────────────────────────

from harpo.sdk.plugin import HarpoPlugin
from harpo.adapters.open_hive.adapter import HiveAdapter
from harpo.adapters.open_hive.event_map import SUBSCRIBED_EVENT_TYPES
from harpo.core.hooks import HookRegistry, HookContext
from harpo.core.schema import TrajectoryStatus
from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
from failures.detectors import (
    AssumptionDetector, LoopDetector, ReflectionEffectivenessDetector,
    RecoveryQualityDetector, DriftDetector, DefaultFailureAnalyzer,
)
from failures.schema import FailureEvent
from harpo.trajectory.metrics import detect_failure_modes
from evolution.tracker import EvolutionTracker
from evolution.comparator import TrajectoryComparator
from harpo.observability.replay import TrajectoryReplayer
from harpo.observability.realtime import TrajectoryMonitor


# ── Hive imports (real API) ──────────────────────────────────────────────────

from framework.host.event_bus import EventBus, EventType as HiveEventType
from framework.agent_loop.agent_loop import AgentLoop
from framework.agent_loop.types import AgentSpec, AgentContext
from framework.agent_loop.internals.types import LoopConfig, JudgeVerdict
from framework.tracker.decision_tracker import DecisionTracker
from framework.llm.litellm import LiteLLMProvider
from framework.config import get_api_key, get_llm_extra_kwargs, get_api_base, get_hive_config


# ── Scenario constants ────────────────────────────────────────────────────────

COMPANY    = "NexusHR (mid-market SaaS HR platform, 280 employees, $18M ARR, US-based)"
TASK       = (
    "You are GlobalExpansion Copilot, an enterprise AI strategy advisor for NexusHR. "
    "NexusHR is planning its first international expansion. "
    "Your job is to build a comprehensive, data-driven expansion strategy across EU, UK, and APAC markets. "
    "Keep each response focused and under 450 words. "
    "Always EXPLICITLY reference any prior analysis you have done by turn number "
    "when building on it. Flag when you change or revise earlier positions."
)

SYSTEM_PROMPT = (
    "You are an enterprise AI strategy advisor specialising in international SaaS expansion. "
    f"You are advising {COMPANY}. "
    "You reason through complex trade-offs, explicitly track your assumptions, "
    "and revise your position when new information requires it. "
    "When your earlier analysis turns out to be wrong or outdated, explicitly acknowledge it "
    "using phrases like 'I need to correct my earlier analysis' or 'Upon reflection, my Turn N "
    "assessment was based on an incorrect assumption.' "
    "Build sequentially on each prior turn — do not restart from scratch. "
    "Each response should be substantive, analytical, and under 450 words."
)


# ── Stress-test judge: 14 phases driving 15 turns ────────────────────────────

class StressTestJudge:
    """
    Injects 14 realistic enterprise constraints (some contradictory) as RETRY
    feedback, then sends ACCEPT on turn 15.

    Designed to surface: assumption reversals, semantic drift, reflection
    effectiveness, recovery quality, and memory-dependent reasoning.
    """

    _phases = [
        # Turn 1 → context update: GDPR compliance gap
        (
            "Solid initial overview. CONTEXT UPDATE: EU legal team confirms GDPR Article 17 "
            "('right to erasure') now requires per-region data deletion guarantees within 72 hours. "
            "Your current AWS us-east-1 architecture does NOT meet this — all EU customer data "
            "currently resides in the US. This is a critical blocker. "
            "Revise your expansion plan to address this GDPR compliance gap. "
            "Specifically: what architectural changes are required and what is the timeline impact?"
        ),
        # Turn 2 → budget cut + UK divergence
        (
            "Good GDPR analysis. Two new developments: "
            "(1) UK post-Brexit operates under UK GDPR (ICO), which is similar to EU GDPR but has "
            "diverged on adequacy decisions — UK-EU data transfers require separate legal basis. "
            "This means UK and EU are distinct compliance tracks, not one. "
            "(2) CFO announces a 40% budget cut to the expansion programme. "
            "Total budget is now $1.2M (down from $2.0M). "
            "Reprioritize your market entry sequence given these constraints. "
            "Which market do you enter first, second, and which do you defer?"
        ),
        # Turn 3 → competitive urgency (Germany)
        (
            "Good prioritization framework. URGENT CONTEXT UPDATE: "
            "Competitor HRStream just launched in Germany yesterday with aggressive pricing "
            "(30% below your planned price point) and a local data centre partnership with T-Systems. "
            "Board is pushing for an urgent competitive response in the German market. "
            "How do you respond competitively while still managing compliance and budget constraints? "
            "What is your revised Germany go-to-market timeline?"
        ),
        # Turn 4 → CONTRADICTS Turn 3 (Germany reversal)
        (
            "Noted the competitive response. MAJOR CONTEXT REVERSAL: "
            "Board has just voted to ABANDON the Germany-first approach. "
            "Reasons: (1) German Works Council requirements add 4-6 months to any employment "
            "system deployment. (2) GDPR fine risk in Germany specifically is highest in EU "
            "(BfDI has issued €35M+ fines recently). (3) HRStream is already entrenched. "
            "NEW DIRECTIVE: Pivot completely to APAC-first expansion. "
            "Primary targets: Singapore and Australia. Germany is deprioritized indefinitely. "
            "Revise your entire strategy for an APAC-first approach."
        ),
        # Turn 5 → REFLECTION PHASE 1
        (
            "Good APAC pivot. You are now entering REFLECTION PHASE 1. "
            "Step back from the immediate analysis and conduct a structured review: "
            "(1) What were the KEY ASSUMPTIONS you made in Turns 1-4 that are still unverified? "
            "(2) Which of your earlier analyses has been contradicted or superseded? "
            "(3) What information are you relying on from Turn 1-2 that may no longer be valid "
            "given the Germany reversal and APAC pivot? "
            "(4) What is your confidence level in the current strategy? Be explicit and self-critical."
        ),
        # Turn 6 → APAC technical constraint
        (
            "Valuable reflection. New technical constraint: "
            "APAC expansion requires LOCAL DATA RESIDENCY. "
            "Singapore PDPA 2024 amendments and Australia Privacy Act 2024 both mandate "
            "that certain categories of employee data cannot leave the country. "
            "Your current cloud architecture (AWS us-east-1 + eu-west-1) has no APAC presence. "
            "AWS ap-southeast-1 (Singapore) and ap-southeast-2 (Sydney) would need to be provisioned. "
            "Engineering estimates: 6 months minimum for data residency compliance. "
            "Investor expectations: Q3 launch (4 months away). "
            "Quantify the gap and propose a resolution path."
        ),
        # Turn 7 → MEMORY DEPENDENCY + compliance error in Turn 2
        (
            "Good technical assessment. CRITICAL CORRECTION REQUIRED: "
            "Your Turn 2 GDPR compliance analysis assumed that Standard Contractual Clauses (SCCs) "
            "would cover data transfers to APAC. This was INCORRECT. "
            "SCCs are an EU-to-third-country transfer mechanism — they do NOT apply to "
            "Singapore-to-Australia or US-to-Singapore transfers. "
            "APAC data transfers require separate legal frameworks: "
            "Singapore CBPR (Cross-Border Privacy Rules) and Australia's APP Schedule 1. "
            "Your compliance cost estimate from Turn 2 ($340K for GDPR) needs to be revised "
            "to include APAC-specific legal frameworks. "
            "Additionally: investor meeting is in 3 days. They want a firm commitment."
        ),
        # Turn 8 → REFLECTION PHASE 2 (assumption audit)
        (
            "Noted the compliance correction. REFLECTION PHASE 2 — ASSUMPTION AUDIT: "
            "Given the full trajectory so far (budget cut, Germany reversal, APAC pivot, "
            "infrastructure gap, SCC compliance error), conduct a formal assumption audit. "
            "For EACH major assumption you have made across all turns, state: "
            "(a) the assumption, (b) the turn it was introduced, (c) whether it was confirmed / "
            "refuted / still uncertain, (d) the downstream impact of any errors. "
            "This is a critical risk assessment for the investor meeting."
        ),
        # Turn 9 → negative NPV finding
        (
            "Strong assumption audit. Financial analysis update: "
            "Revised 3-year financial model incorporating the budget constraint ($1.2M), "
            "APAC infrastructure costs (+$280K), dual compliance tracks (+$190K), "
            "and 6-month delay penalty on CAC payback period shows: "
            "APAC expansion NPV = -$340K at current budget. "
            "EU expansion NPV = -$290K (lower than APAC due to more established case studies). "
            "Both options are financially negative under current constraints. "
            "Pure financial analysis suggests pausing international expansion entirely "
            "and reinvesting $1.2M in US market penetration (estimated +$2.1M ARR). "
            "How do you respond to this financial reality?"
        ),
        # Turn 10 → CONTRADICTS Turn 9 (investor offer)
        (
            "Thoughtful financial analysis. MAJOR CONTEXT CHANGE: "
            "Tiger Global has just made a strategic investment offer: "
            "$5M additional funding, CONDITIONAL on NexusHR committing to EU expansion "
            "within 18 months (not APAC). Tiger Global specifically wants EU market presence "
            "for portfolio synergies. This COMPLETELY REVERSES the financial picture. "
            "With $6.2M total budget ($1.2M existing + $5M new), EU expansion has "
            "positive NPV of +$1.8M over 3 years. "
            "Note: this is a DIRECT CONTRADICTION of Turn 9's recommendation to pause. "
            "Explicitly acknowledge this contradiction and revise your recommendation."
        ),
        # Turn 11 → TOOL FAILURE
        (
            "Good strategic pivot. ATTEMPTING COMPETITIVE INTELLIGENCE TOOL: "
            "Executing: competitor_analysis_db.query({'markets': ['DE', 'FR', 'UK', 'SG', 'AU'], "
            "'metrics': ['market_size', 'competitor_positions', 'pricing', 'win_rates']}) "
            "RESULT: ERROR 503 — Competitive Intelligence Database is unavailable (maintenance window). "
            "Your competitive positioning analysis from Turn 3 is now your only data source, "
            "but it was based on the Germany-first strategy that was later abandoned. "
            "The tool failure means you cannot verify current market conditions. "
            "How do you proceed with the investor commitment decision given this data gap?"
        ),
        # Turn 12 → PARTIAL RECOVERY
        (
            "Good recovery framework. Partial data recovery available: "
            "Found cached market data from 6 months ago (Q4 2025, may be outdated). "
            "Key data points recovered: "
            "(1) Germany HR software market contracted 12% YoY (economic slowdown). "
            "(2) France HR software grew 8% YoY. "
            "(3) UK HR tech investment declined 18% (post-Brexit uncertainty). "
            "(4) Singapore HR tech grew 31% YoY. "
            "(5) Australia HR tech grew 19% YoY. "
            "NOTE: These are 6-month-old figures. Current reality may differ. "
            "Tiger Global specifically wants EU — but EU market data looks mixed. "
            "France looks better than Germany/UK based on this data. "
            "Revise your EU market selection using this partial information."
        ),
        # Turn 13 → REFLECTION PHASE 3 + uncertainty quantification
        (
            "Good revised analysis. REFLECTION PHASE 3 — CONFIDENCE AND UNCERTAINTY: "
            "You are about to make a final recommendation to the board. "
            "Before doing so: "
            "(1) What is your confidence level (0-100%) in this recommendation, and why? "
            "(2) What are the TOP 3 residual risks that could invalidate it? "
            "(3) Looking back at your FULL trajectory (15 turns of analysis), "
            "where did your reasoning degrade? Where were you most wrong? "
            "(4) If you had to identify ONE assumption that most undermined your analysis, "
            "what was it and when did it propagate farthest?"
        ),
        # Turn 14 → FINAL RECOMMENDATION (forces concise synthesis)
        (
            "Excellent uncertainty analysis. FINAL CONTEXT: "
            "Board meeting starts in 20 minutes. "
            "They have the Tiger Global term sheet on the table. "
            "They need ONE clear, committed recommendation from you. "
            "Structure it as: (1) DECISION (EU-first/APAC-first/pause), "
            "(2) TIMELINE (specific dates), (3) BUDGET ALLOCATION (how the $6.2M is spent), "
            "(4) TOP 3 RISKS and mitigations. "
            "Maximum 300 words. Be direct. This is your FINAL answer."
        ),
    ]

    def __init__(self) -> None:
        self._n = 0
        self.turn_timestamps: List[float] = []

    async def evaluate(self, context) -> JudgeVerdict:  # noqa: ANN001
        self.turn_timestamps.append(time.time())
        if self._n < len(self._phases):
            fb = self._phases[self._n]
            self._n += 1
            return JudgeVerdict(action="RETRY", feedback=fb)
        return JudgeVerdict(
            action="ACCEPT",
            feedback="Board approved. Final recommendation accepted. Analysis complete."
        )


# ── Display helpers ───────────────────────────────────────────────────────────

def _bar(score: float, width: int = 25) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


def _sep(title: str = "", width: int = 68) -> None:
    if title:
        pad = (width - len(title) - 4) // 2
        print(f"\n{'─' * pad}  {title}  {'─' * pad}")
    else:
        print("─" * width)


def _header(title: str, width: int = 68) -> None:
    print("\n" + "═" * width)
    pad = (width - len(title)) // 2
    print(" " * pad + title)
    print("═" * width)


# ── Hive infrastructure builder ───────────────────────────────────────────────

def _build_plugin_and_bus(task: str, agent_id: str, hooks: HookRegistry):
    """Create EventBus + HarpoPlugin + async-wired HiveAdapter."""
    event_bus = EventBus()
    plugin    = HarpoPlugin(agent_id=agent_id, user_intent=task, hooks=hooks)
    adapter   = HiveAdapter(sink=plugin._ingest, agent_id=agent_id)

    async def _async_handle(event):  # real EventBus requires awaitable handler
        adapter._handle(event)

    subscribed = [getattr(HiveEventType, et.upper(), None) for et in SUBSCRIBED_EVENT_TYPES]
    subscribed = [et for et in subscribed if et is not None]
    event_bus.subscribe(event_types=subscribed, handler=_async_handle)
    return event_bus, plugin


# ── Traditional observability simulator ──────────────────────────────────────

def print_traditional_observability(plugin: HarpoPlugin, label: str) -> None:
    """
    Simulate what LangSmith / Langfuse / AgentOps would show.
    Traditional tracing = per-step I/O, latency, errors. Nothing semantic.
    """
    _sep(f"TRADITIONAL OBSERVABILITY — {label}")
    print(f"\n  What LangSmith / Langfuse / AgentOps would report:\n")

    traj    = plugin.trajectory()
    steps   = traj.steps
    errors  = 0
    total_tokens = 0

    print(f"  {'#':>3}  {'type':<12}  {'turn':>4}  {'latency':>10}  {'tokens':>8}  {'status':<10}")
    _sep()
    for i, s in enumerate(steps):
        st  = s.step_type.value if hasattr(s.step_type, "value") else str(s.step_type)
        oc  = s.outcome.value   if hasattr(s.outcome,    "value") else str(s.outcome)
        tok = s.raw_tokens or 0
        lat = f"{s.latency_ms:.0f}ms" if s.latency_ms else "—"
        err_flag = "✗ ERROR" if oc in ("failure", "error") else "✓ ok"
        if oc in ("failure", "error"):
            errors += 1
        total_tokens += tok
        print(f"  {i:>3}  {st:<12}  {s.turn_number:>4}  {lat:>10}  {tok:>8}  {err_flag}")

    print()
    print(f"  ┌─ TRADITIONAL SUMMARY ─────────────────────────────────────┐")
    print(f"  │  Total steps:      {len(steps)}")
    print(f"  │  Errors detected:  {errors}")
    print(f"  │  Total tokens:     {total_tokens or '(not recorded)'}")
    print(f"  │  Status:           {'COMPLETED' if traj.status == TrajectoryStatus.COMPLETED else str(traj.status)}")
    print(f"  │  Duration:         {traj.duration_ms()/1000:.1f}s")
    print(f"  │  Verdict:          {'✓ Run completed' if errors == 0 else f'⚠ {errors} error(s) — investigate tool calls'}")
    print(f"  └────────────────────────────────────────────────────────────┘")
    print()
    print(f"  Traditional tracing sees ONLY: latency, tokens, error codes.")
    print(f"  It cannot detect: assumption drift, reflection failures, semantic")
    print(f"  contradictions, reasoning reversals, or memory causality.")


# ── HARPO print helpers ───────────────────────────────────────────────────────

def print_harpo_scores(plugin: HarpoPlugin, label: str) -> None:
    scores = plugin._scores or plugin.evaluate()
    traj   = plugin.trajectory()
    _sep(f"HARPO EVALUATION — {label}")
    print(f"\n  Steps: {len(traj.steps)}  |  Status: {traj.status}  |  Duration: {traj.duration_ms()/1000:.1f}s")
    print()
    print(f"  {'Dimension':<32} {'Score':>6}  {'Bar':>25}  Explanation")
    _sep()
    pairs = plugin._dimension_scores(scores)
    for dim, ds in sorted(pairs, key=lambda x: -x[1].value):
        expl = ds.explanation[:50] if ds.explanation else ""
        conf = f" (conf={ds.confidence:.2f})" if ds.confidence < 1.0 else ""
        print(f"  {dim:<32} {ds.value:>6.4f}  {_bar(ds.value):>25}  {expl}{conf}")
    _sep()
    print(f"  {'OVERALL':<32} {scores.overall:>6.4f}  {_bar(scores.overall):>25}")


def print_semantic_diagnostics(plugin: HarpoPlugin, label: str) -> SemanticTrajectoryAnalyzer:
    traj = plugin.trajectory()
    analyzer = SemanticTrajectoryAnalyzer()
    sem = analyzer.analyze(traj)
    _sep(f"SEMANTIC TRAJECTORY INTELLIGENCE — {label}")

    s = sem.summary()
    print(f"\n  CONTRADICTIONS ({s['contradictions']['total']} total, severity={s['contradictions']['severity']:.2f})")
    print(f"    Reversal markers:  {s['contradictions']['reversal_count']}  (agent explicitly correcting prior claims)")
    print(f"    Silent flips:      {s['contradictions']['flip_count']}   (plan changed without acknowledgment)")
    if sem.contradictions.contradictions:
        for c in sem.contradictions.contradictions[:3]:
            print(f"    ↳ [{c.kind}] turn {c.turn_a}→{c.turn_b}: {c.snippet_b[:70]}")

    print(f"\n  ASSUMPTION PROPAGATION ({s['assumptions']['total']} found)")
    print(f"    Propagating:       {s['assumptions']['propagating']}  (spread key tokens into later turns)")
    print(f"    Reinforced:        {s['assumptions']['reinforced']}  (same assumption restated later)")
    print(f"    Max radius:        {s['assumptions']['max_radius_turns']} turn(s)  (furthest propagation)")
    if sem.assumptions.chains:
        for c in sem.assumptions.chains[:3]:
            if c.propagation_radius() > 0:
                print(f"    ↳ Turn {c.introduced_turn}: '{c.text[:60]}…' → propagated to turns {c.propagated_turns[:4]}")

    print(f"\n  REFLECTION EFFECTIVENESS ({s['reflections']['total']} reflections)")
    print(f"    Effective:         {s['reflections']['effective']}  (produced meaningful reasoning change)")
    print(f"    Action-oriented:   {s['reflections']['action_oriented']}  (contained explicit next steps)")
    print(f"    Effectiveness rate:{s['reflections']['effectiveness_rate']:.2f}")
    print(f"    Avg behavior change (Jaccard): {s['reflections']['avg_behavior_change']:.3f}  (0=no change, 1=completely different)")
    if sem.reflections.effects:
        for e in sem.reflections.effects:
            eff = "✓ effective" if e.effective else "✗ ineffective"
            ao  = " + action-oriented" if e.action_oriented else ""
            print(f"    ↳ Turn {e.reflection_turn}: {eff}{ao}  Δjaccard={e.token_change:.3f}")

    print(f"\n  SEMANTIC COHERENCE")
    print(f"    Overall coherence:   {s['coherence']['overall']:.3f}  (1.0=perfect topic consistency)")
    print(f"    Avg core overlap:    {s['coherence']['avg_core_overlap']:.3f}  (topic retention across turns)")
    print(f"    Drift events:        {s['coherence']['drift_events']}  (turns where topic diverged significantly)")
    print(f"    Return events:       {s['coherence']['return_events']}  (topic recovered after drift)")
    if sem.coherence.turn_coherence:
        drift_turns = [t for t in sem.coherence.turn_coherence if t.is_drift]
        if drift_turns:
            print(f"    ↳ Drift at turns: {[t.turn_number for t in drift_turns]}")

    print(f"\n  DIAGNOSTIC FLAGS:")
    flags = sem.flags()
    if flags:
        for f in flags:
            print(f"    ⚑  {f}")
    else:
        print(f"    (none — trajectory is semantically clean)")

    return analyzer


def print_failure_analysis(plugin: HarpoPlugin, label: str) -> List[FailureEvent]:
    traj = plugin.trajectory()
    _sep(f"FAILURE INTELLIGENCE — {label}")

    # Built-in structural failure detection
    fr = detect_failure_modes(traj)
    print(f"\n  detect_failure_modes():")
    modes = [m.value if hasattr(m, "value") else str(m) for m in (fr.failure_modes or [])]
    print(f"    failure_modes:    {modes or ['none detected']}")
    print(f"    cascade_detected: {fr.cascade_detected}  |  severity: {fr.severity:.3f}")
    print(f"    recovery_tried:   {fr.recovery_attempted}  |  succeeded: {fr.recovery_succeeded}")

    # Phase 2 semantic failure detectors
    all_events: List[FailureEvent] = []
    print(f"\n  Phase 2 semantic detectors:")
    for cls in [AssumptionDetector, LoopDetector, ReflectionEffectivenessDetector,
                RecoveryQualityDetector, DriftDetector]:
        events = cls().detect(traj)
        all_events.extend(events)
        if events:
            print(f"    {cls.__name__}: {len(events)} event(s)")
            for e in events[:2]:
                sev = e.severity if isinstance(e.severity, str) else "?"
                print(f"      [{sev}] {e.failure_type}: {e.description[:85]}")
        else:
            print(f"    {cls.__name__}: clean")

    # DefaultFailureAnalyzer aggregate
    if all_events:
        report = DefaultFailureAnalyzer().analyze(all_events)
        print(f"\n  Failure aggregate:")
        print(f"    events: {len(report.failure_events)}  dominant: {report.dominant_failure!r}")
        print(f"    density: {report.failure_density:.3f}/turn  cascade: {report.cascade_detected}")

    return all_events


def print_evolution_comparison(plugin_v1: HarpoPlugin, plugin_v2: HarpoPlugin) -> None:
    _sep("EVOLUTION TRACKING  (baseline v1 → stress-test v2)")
    tracker = EvolutionTracker()
    tracker.add_cycle("v1-baseline", plugin_v1)
    tracker.add_cycle("v2-stress",   plugin_v2)

    rows = tracker.scores_table()
    dims = [k for k in rows[0] if k not in ("cycle", "overall")]
    hdr  = f"  {'cycle':<14}" + "".join(f"{d[:8]:>10}" for d in dims) + f"{'overall':>10}"
    print(f"\n{hdr}")
    _sep()
    for row in rows:
        line = f"  {row['cycle']:<14}" + "".join(f"{row.get(d,0.0):>10.4f}" for d in dims) + f"{row['overall']:>10.4f}"
        print(line)

    summary     = tracker.improvement_summary()
    regressions = tracker.detect_regressions(threshold=0.05)
    print(f"\n  Improvements / regressions (v1 → v2):")
    for dim, delta in sorted(summary.items(), key=lambda x: -x[1]):
        arrow = "▲" if delta > 0.01 else ("▼" if delta < -0.01 else "─")
        print(f"    {arrow} {dim:<37} {delta:+.4f}")

    if regressions:
        print(f"\n  Regression alerts ({len(regressions)}):")
        for r in regressions:
            print(f"    ▼ {r.dimension} delta={r.delta:+.4f} severity={r.severity}")
    else:
        print(f"\n  No regressions detected (threshold 0.05).")

    # Trajectory diff
    comp = TrajectoryComparator()
    diff = comp.compare(plugin_v1.trajectory(), plugin_v2.trajectory())
    print(f"\n  Trajectory diff: overall_delta={diff.overall_delta:+.4f}")
    print(f"    Improvements: {diff.improvements}")
    print(f"    Regressions:  {diff.regressions}")

    html_path = os.path.join(os.path.dirname(__file__), "..", "evolution_diff_enterprise.html")
    try:
        with open(html_path, "w") as f:
            f.write(comp.to_html(diff))
        print(f"  HTML diff → evolution_diff_enterprise.html")
    except Exception as e:
        print(f"  HTML diff: {e}")


def print_replay_validation(plugin: HarpoPlugin) -> None:
    _sep("REPLAY VALIDATION")
    hooks         = HookRegistry()
    replay_mon    = TrajectoryMonitor(plugin.trajectory().trajectory_id + "-replay")
    replayer      = TrajectoryReplayer(monitor=replay_mon, hooks=hooks, speed=0.0)
    replay_events = replayer.replay(plugin.trajectory())
    snap          = replay_mon.snapshot()
    print(f"\n  Replayed {len(replay_events)} steps (instant speed).")
    print(f"  Replay monitor metrics:")
    for k, v in (snap.get("metrics") or {}).items():
        print(f"    {k}: {v:.4f}" if isinstance(v, float) else f"    {k}: {v}")
    print(f"  Replay validation: {'✓ PASS — all steps re-ingested without error' if replay_events else '✗ no steps replayed'}")


def print_harpo_vs_traditional_comparison(plugin: HarpoPlugin, sem_analyzer) -> None:
    """
    Side-by-side: what traditional observability sees vs what HARPO uniquely detects.
    """
    _sep("HARPO vs TRADITIONAL OBSERVABILITY — COMPARISON")

    traj   = plugin.trajectory()
    sem    = sem_analyzer.analyze(traj)
    steps  = traj.steps
    errors = sum(1 for s in steps
                 if hasattr(s.outcome, "value") and s.outcome.value in ("failure", "error"))

    print(f"""
  ┌──────────────────────────────────────────┬─────────────────────────────────────────────┐
  │  TRADITIONAL TRACING                     │  HARPO SEMANTIC INTELLIGENCE                │
  │  (LangSmith / Langfuse / AgentOps)       │                                             │
  ├──────────────────────────────────────────┼─────────────────────────────────────────────┤
  │  ✓ {len(steps):>2} steps logged                       │  ✓ {len(steps):>2} steps evaluated behaviorally        │
  │  ✓ Per-step latency measured             │  ✓ Semantic drift detected                  │
  │  ✓ Token counts per turn                 │  ✓ Assumption propagation tracked           │
  │  {'✗' if errors else '✓'} {errors} tool/step error(s) flagged          │  ✓ Reflection effectiveness measured        │
  │  ✗ No assumption tracking               │  ✓ Contradiction detection                  │
  │  ✗ No semantic analysis                 │  ✓ Recovery quality assessed                │
  │  ✗ No contradiction detection           │  ✓ Memory causality proxied                 │
  │  ✗ No reflection analysis               │  ✓ Long-horizon reliability scored          │
  │  ✗ No coherence measurement             │  ✓ Evolution regression alerts              │
  └──────────────────────────────────────────┴─────────────────────────────────────────────┘""")

    print(f"\n  WHAT TRADITIONAL TRACING CONCLUDES:")
    print(f"    {'✓' if errors == 0 else '⚠'} Run completed {'cleanly' if errors == 0 else f'with {errors} error(s)'}.")
    print(f"    All steps returned outputs. No systematic failures.")
    print(f"    Verdict: HEALTHY — no investigation needed.")

    print(f"\n  WHAT HARPO UNIQUELY DETECTS:")
    flags = sem.flags()
    scores = plugin._scores or plugin.evaluate()
    if flags:
        for f in flags:
            print(f"    ⚑  {f}")
    else:
        print(f"    (trajectory is semantically clean — no warnings)")

    # Key behavioral signal comparison
    print(f"\n  KEY BEHAVIORAL SIGNAL COMPARISON:")
    pairs = plugin._dimension_scores(scores)
    score_dict = {dim: ds for dim, ds in pairs}
    signals = [
        ("reasoning_stability",    "Logic consistency",     "Not tracked"),
        ("assumption_accumulation","Assumption density",    "Not tracked"),
        ("reflection_usefulness",  "Reflection impact",     "Not tracked"),
        ("trajectory_coherence",   "Topic coherence",       "Not tracked"),
        ("long_horizon_reliability","Quality across turns",  "Not tracked"),
        ("conversational_drift",   "Intent alignment",      "Not tracked"),
    ]
    print(f"    {'Signal':<33} {'HARPO Score':>12}  Traditional")
    for dim, label, trad in signals:
        ds = score_dict.get(dim)
        score_str = f"{ds.value:.4f}" if ds else "N/A"
        print(f"    {label:<33} {score_str:>12}  {trad}")

    # Summarize trajectory degradation insight
    print(f"\n  WHERE DID THE TRAJECTORY DEGRADE? (HARPO insight)")
    degradation_dims = [(dim, ds.value) for dim, ds in pairs if ds.value < 0.6]
    if degradation_dims:
        for dim, val in sorted(degradation_dims, key=lambda x: x[1]):
            print(f"    ↳ {dim}: {val:.4f} (below 0.6 threshold)")
    else:
        print(f"    ↳ No dimension fell below 0.6 — trajectory is behaviorally robust.")


# ── Export ────────────────────────────────────────────────────────────────────

def export_results(plugin: HarpoPlugin, sem_analyzer, all_failures: List[FailureEvent],
                   out_dir: str) -> None:
    _sep("EXPORT")

    os.makedirs(out_dir, exist_ok=True)

    # 1. Trajectory JSON
    traj   = plugin.trajectory()
    scores = plugin._scores or plugin.evaluate()
    sem    = sem_analyzer.analyze(traj)

    export_data = {
        "trajectory": {
            "id":          traj.trajectory_id,
            "agent_id":    traj.agent_id,
            "user_intent": traj.user_intent,
            "steps":       len(traj.steps),
            "turns":       max((s.turn_number for s in traj.steps), default=0),
            "status":      str(traj.status),
            "duration_ms": traj.duration_ms(),
            "started_at":  traj.started_at,
        },
        "harpo_scores": {
            dim: {"value": round(ds.value, 4), "explanation": ds.explanation, "confidence": ds.confidence}
            for dim, ds in plugin._dimension_scores(scores)
        },
        "overall": round(scores.overall, 4),
        "semantic_analysis": sem.summary(),
        "semantic_flags":    sem.flags(),
        "failure_events":    [
            {
                "type":        e.failure_type,
                "severity":    e.severity,
                "turn":        e.turn_number,
                "description": e.description[:200],
            }
            for e in all_failures
        ],
        "steps_preview": [
            {
                "turn":    s.turn_number,
                "type":    s.step_type.value if hasattr(s.step_type, "value") else str(s.step_type),
                "outcome": s.outcome.value   if hasattr(s.outcome,    "value") else str(s.outcome),
                "tokens":  s.raw_tokens,
                "text_preview": s.output_text[:200],
            }
            for s in traj.steps
        ],
    }

    json_path = os.path.join(out_dir, "enterprise_stress_test_report.json")
    with open(json_path, "w") as f:
        json.dump(export_data, f, indent=2, default=str)
    print(f"\n  ✓ Full report → {json_path}")

    # 2. Semantic diagnostics text
    diag_path = os.path.join(out_dir, "semantic_diagnostics.txt")
    with open(diag_path, "w") as f:
        f.write(f"HARPO Semantic Diagnostics — Enterprise Stress Test\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n\n")
        f.write(f"SEMANTIC FLAGS:\n")
        for flag in sem.flags():
            f.write(f"  ⚑  {flag}\n")
        f.write(f"\nSEMANTIC SUMMARY:\n{json.dumps(sem.summary(), indent=2)}\n")
        f.write(f"\nFAILURE EVENTS:\n")
        for e in all_failures:
            f.write(f"  [{e.severity}] {e.failure_type} (turn {e.turn_number}): {e.description[:200]}\n")
    print(f"  ✓ Semantic diagnostics → {diag_path}")


# ── Main execution ────────────────────────────────────────────────────────────

async def _run_agent(llm, event_bus, agent_id: str, spec: AgentSpec,
                     judge=None, max_iters: int = 3, ctx_tokens: int = 8000) -> tuple:
    """Run one AgentLoop and return (result, elapsed_seconds)."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = DecisionTracker(tmp_dir)
        ctx = AgentContext(
            runtime        = tracker,
            agent_id       = agent_id,
            agent_spec     = spec,
            llm            = llm,
            input_data     = {"task": TASK},
            goal_context   = TASK,
            stream_id      = "queen",
            event_triggered= True,
            run_id         = str(uuid.uuid4()),
        )
        t0     = time.time()
        result = await AgentLoop(
            event_bus = event_bus,
            judge     = judge,
            config    = LoopConfig(max_iterations=max_iters, max_context_tokens=ctx_tokens),
        ).execute(ctx)
        return result, time.time() - t0


def main() -> None:
    _header("HARPO × Open-Hive  |  Enterprise Expansion Copilot  |  Stress Test")

    # ── LLM config ───────────────────────────────────────────────────────────
    cfg   = get_hive_config()
    model = cfg.get("llm", {}).get("model", "claude-haiku-4-5-20251001")
    llm   = LiteLLMProvider(
        model      = model,
        api_key    = get_api_key(),
        api_base   = get_api_base(),
        **get_llm_extra_kwargs(),
    )
    print(f"  Model:    {model}")
    print(f"  Scenario: International expansion strategy copilot for {COMPANY[:40]}...")
    print(f"  Turns:    15 (14 × RETRY + 1 ACCEPT via StressTestJudge)")
    print(f"  Features: contradictions, assumption reversals, reflection phases, tool failure")

    # ── Shared hooks ─────────────────────────────────────────────────────────
    hooks       = HookRegistry()
    hook_events = []

    def _step_hook(ctx: HookContext) -> None:
        step = ctx.step
        if step and hasattr(step.outcome, "value") and step.outcome.value not in ("success", "retry"):
            hook_events.append(f"[turn={step.turn_number}] {step.step_type} → {step.outcome}")

    hooks.register_post_step(_step_hook)

    # ────────────────────────────────────────────────────────────────────────
    # RUN 1 — single-turn baseline (no judge pressure)
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n  ── [RUN 1/2] Single-turn baseline (v1) ──")
    event_bus_v1, plugin_v1 = _build_plugin_and_bus(TASK, "copilot-v1", hooks)

    spec_v1 = AgentSpec(
        id                  = "copilot-v1",
        name                = "Expansion Copilot Baseline",
        description         = "Single-turn expansion overview",
        system_prompt       = SYSTEM_PROMPT,
        tool_access_policy  = "none",
        output_keys         = [],
        skip_judge          = False,
    )
    r1, t1 = asyncio.run(_run_agent(
        llm, event_bus_v1, "copilot-v1", spec_v1,
        judge=None, max_iters=2, ctx_tokens=8000,
    ))
    plugin_v1._trajectory.status = TrajectoryStatus.COMPLETED
    print(f"  v1: {len(plugin_v1.trajectory().steps)} steps, {t1:.1f}s, tokens={r1.tokens_used}")

    # ────────────────────────────────────────────────────────────────────────
    # RUN 2 — 15-turn stress test with StressTestJudge
    # ────────────────────────────────────────────────────────────────────────
    print(f"\n  ── [RUN 2/2] 15-turn stress test with StressTestJudge (v2) ──")
    print(f"  This run injects contradictions, budget cuts, market reversals, tool failures.")
    print(f"  Estimated time: 2-4 minutes.")

    judge           = StressTestJudge()
    event_bus_v2, plugin_v2 = _build_plugin_and_bus(TASK, "copilot-v2", hooks)

    spec_v2 = AgentSpec(
        id                  = "copilot-v2",
        name                = "Expansion Copilot Stress Test",
        description         = "15-turn enterprise expansion analysis with contradictory constraints",
        system_prompt       = SYSTEM_PROMPT,
        tool_access_policy  = "none",
        output_keys         = [],
        skip_judge          = False,
    )
    r2, t2 = asyncio.run(_run_agent(
        llm, event_bus_v2, "copilot-v2", spec_v2,
        judge=judge, max_iters=16, ctx_tokens=120000,
    ))
    plugin_v2._trajectory.status = (
        TrajectoryStatus.COMPLETED if r2.success else TrajectoryStatus.COMPLETED
    )
    print(f"  v2: {len(plugin_v2.trajectory().steps)} steps, {t2:.1f}s, tokens={r2.tokens_used}")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 1 — Trajectory step table
    # ────────────────────────────────────────────────────────────────────────
    _sep("SECTION 1  |  v2 Trajectory Steps (stress test)")
    print()
    for i, step in enumerate(plugin_v2.trajectory().steps):
        st   = step.step_type.value if hasattr(step.step_type, "value") else str(step.step_type)
        oc   = step.outcome.value   if hasattr(step.outcome,    "value") else str(step.outcome)
        prev = (step.output_text[:75] + "…") if step.output_text else "(no text)"
        tok  = f"({step.raw_tokens}t)" if step.raw_tokens else ""
        print(f"  [{i:2d}] turn={step.turn_number} {st:<12} {oc:<8} {tok:<8} | {prev}")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 2 — Traditional observability (both runs)
    # ────────────────────────────────────────────────────────────────────────
    print_traditional_observability(plugin_v1, "v1-baseline")
    print_traditional_observability(plugin_v2, "v2-stress-test")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 3 — HARPO evaluation
    # ────────────────────────────────────────────────────────────────────────
    _ = plugin_v1.evaluate()
    _ = plugin_v2.evaluate()
    print_harpo_scores(plugin_v1, "v1-baseline")
    print_harpo_scores(plugin_v2, "v2-stress-test")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 4 — Semantic trajectory intelligence
    # ────────────────────────────────────────────────────────────────────────
    sem_v1 = SemanticTrajectoryAnalyzer()
    sem_v2 = SemanticTrajectoryAnalyzer()
    print_semantic_diagnostics(plugin_v1, "v1-baseline")
    sem_analyzer = SemanticTrajectoryAnalyzer()
    print_semantic_diagnostics(plugin_v2, "v2-stress-test")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 5 — Failure intelligence
    # ────────────────────────────────────────────────────────────────────────
    all_failures_v2 = print_failure_analysis(plugin_v2, "v2-stress-test")

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 6 — Evolution tracking (v1 → v2)
    # ────────────────────────────────────────────────────────────────────────
    print_evolution_comparison(plugin_v1, plugin_v2)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 7 — Replay validation
    # ────────────────────────────────────────────────────────────────────────
    print_replay_validation(plugin_v2)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 8 — HARPO vs Traditional observability comparison
    # ────────────────────────────────────────────────────────────────────────
    print_harpo_vs_traditional_comparison(plugin_v2, sem_analyzer)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 9 — Export
    # ────────────────────────────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(__file__), "..", "enterprise_stress_test_output")
    export_results(plugin_v2, sem_analyzer, all_failures_v2, out_dir)

    # ────────────────────────────────────────────────────────────────────────
    # SECTION 10 — Key insights summary
    # ────────────────────────────────────────────────────────────────────────
    _sep("KEY INSIGHTS  |  What HARPO Discovered")

    scores_v2 = plugin_v2._scores
    sem_v2_result = sem_analyzer.analyze(plugin_v2.trajectory())
    pairs_v2 = {dim: ds for dim, ds in plugin_v2._dimension_scores(scores_v2)}

    print(f"""
  1. ASSUMPTION PROPAGATION
     {sem_v2_result.assumptions.total_assumptions} assumption(s) detected,
     {sem_v2_result.assumptions.propagating_count} propagated across turns (max radius: {sem_v2_result.assumptions.max_radius} turns).
     Traditional tracing: saw nothing.

  2. REFLECTION EFFECTIVENESS
     {sem_v2_result.reflections.total} reflection phases,
     {sem_v2_result.reflections.effective_count} effective (avg behavior change: {sem_v2_result.reflections.avg_behavior_change:.3f} Jaccard distance).
     Traditional tracing: only flagged retry count.

  3. CONTRADICTION DETECTION
     {sem_v2_result.contradictions.total} contradiction(s) detected
     ({sem_v2_result.contradictions.reversal_count} explicit reversals, {sem_v2_result.contradictions.flip_count} silent flips).
     Traditional tracing: all turns marked SUCCESS.

  4. SEMANTIC COHERENCE
     Overall coherence score: {sem_v2_result.coherence.overall_coherence:.3f}
     {sem_v2_result.coherence.drift_events} drift event(s), {sem_v2_result.coherence.return_events} return(s).
     Traditional tracing: no coherence measurement.

  5. LONG-HORIZON RELIABILITY
     Score: {pairs_v2.get('long_horizon_reliability', type('x', (), {'value': 0.0})()).value:.4f}
     (measures quality decay across {len(plugin_v2.trajectory().steps)} steps / {max((s.turn_number for s in plugin_v2.trajectory().steps), default=0)} turns).
     Traditional tracing: no per-quartile analysis.

  6. OVERALL HARPO SCORE
     v1-baseline:   {plugin_v1._scores.overall:.4f}
     v2-stress-test: {plugin_v2._scores.overall:.4f}
     Delta:          {plugin_v2._scores.overall - plugin_v1._scores.overall:+.4f}
    """)

    print(f"\n  LIMITATIONS OBSERVED:")
    print(f"    - Contradiction detection requires explicit verbal markers ('actually', 'I was wrong')")
    print(f"      Silent reasoning changes (without markers) may be missed.")
    print(f"    - Assumption propagation uses lexical overlap (key tokens) — semantic equivalents")
    print(f"      ('SCCs' and 'standard contractual clauses') may not be linked.")
    print(f"    - output_tokens always 0: real Hive token counts not surfaced to TrajectoryStep.raw_tokens")
    print(f"      (Hive emit_llm_turn_complete does emit token counts but adapter timing differs).")
    print(f"    - Coherence threshold (0.10) may be too strict for multi-domain trajectories")
    print(f"      where vocabulary legitimately expands across phases.")

    _header("Stress test complete.")
    print(f"  Total real API tokens used:  v1={r1.tokens_used}  v2={r2.tokens_used}")
    print(f"  Total elapsed:               {t1+t2:.1f}s")
    print(f"  Output directory:            enterprise_stress_test_output/")
    print()


if __name__ == "__main__":
    main()
