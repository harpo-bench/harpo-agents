"""
HarpoPlugin — universal entry point for integrating HARPO into any agent.

Minimal usage
-------------
    # Open-Hive
    plugin = HarpoPlugin.for_hive(event_bus, user_intent="research topic X")
    # ... agent runs ...
    print(plugin.report())

    # LangGraph
    plugin, handler = HarpoPlugin.for_langgraph(user_intent="summarise docs")
    graph.invoke(input, config={"callbacks": [handler]})
    print(plugin.report())

    # Any runtime via string name
    import harpo
    plugin = harpo.attach(event_bus, adapter="hive", user_intent="...")
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from harpo.core.events import GenericAgentEvent, GenericEventType
from harpo.core.hooks import HookContext, HookRegistry, default_hooks
from harpo.core.schema import (
    AgentTrajectory,
    MemoryAccess,
    StepOutcome,
    StepType,
    ToolCall,
    TrajectoryScores,
    TrajectoryStatus,
    TrajectoryStep,
)
from harpo.trajectory.pipeline import TrajectoryEvaluator
from harpo.observability.realtime import TrajectoryMonitor


# ── GenericEventType → StepType mapping ─────────────────────────────────────

_EVENT_TO_STEP: dict[GenericEventType, StepType] = {
    GenericEventType.THINK:        StepType.THINK,
    GenericEventType.TOOL_USE:     StepType.TOOL_CALL,
    GenericEventType.OBSERVATION:  StepType.OBSERVATION,
    GenericEventType.RESPOND:      StepType.RESPONSE,
    GenericEventType.REFLECT:      StepType.REFLECTION,
    GenericEventType.RECOVER:      StepType.RECOVERY,
    GenericEventType.MEMORY_READ:  StepType.MEMORY_READ,
    GenericEventType.MEMORY_WRITE: StepType.MEMORY_WRITE,
    GenericEventType.HAND_OFF:     StepType.HANDOFF,
}


class HarpoPlugin:
    """
    Attach HARPO to any agent runtime with a single call.

    Parameters
    ----------
    agent_id     : human-readable label for the agent being evaluated
    user_intent  : task description (used for alignment scoring)
    evaluator    : optional pre-configured TrajectoryEvaluator
    monitor      : optional pre-configured TrajectoryMonitor
    hooks        : optional HookRegistry (defaults to module-level default_hooks)
    """

    def __init__(
        self,
        agent_id:    str = "agent",
        user_intent: str = "",
        evaluator:   Optional[TrajectoryEvaluator] = None,
        monitor:     Optional[TrajectoryMonitor]   = None,
        hooks:       Optional[HookRegistry]        = None,
    ) -> None:
        self._trajectory = AgentTrajectory(
            trajectory_id    = str(uuid.uuid4()),
            agent_id         = agent_id,
            user_intent      = user_intent,
            task_description = user_intent,
        )
        self._evaluator  = evaluator or TrajectoryEvaluator()
        self._monitor    = monitor or TrajectoryMonitor(self._trajectory.trajectory_id)
        self._hooks      = hooks or default_hooks
        self._step_index = 0
        self._scores: Optional[TrajectoryScores] = None

    # ── Factory helpers ──────────────────────────────────────────────────────

    @classmethod
    def for_hive(
        cls,
        event_bus: Any,
        user_intent: str = "",
        **kw,
    ) -> "HarpoPlugin":
        """Attach to a live Open-Hive EventBus."""
        from harpo.adapters.open_hive.adapter import HiveAdapter
        plugin = cls(agent_id="hive-agent", user_intent=user_intent, **kw)
        HiveAdapter(sink=plugin._ingest).attach(event_bus)
        return plugin

    @classmethod
    def for_langgraph(
        cls,
        user_intent: str = "",
        **kw,
    ) -> tuple["HarpoPlugin", Any]:
        """Return (plugin, LangChain BaseCallbackHandler). Phase 2."""
        raise NotImplementedError(
            "LangGraph adapter is planned for Phase 2. "
            "See src/adapters/langgraph/ for the stub."
        )

    @classmethod
    def for_crewai(
        cls,
        user_intent: str = "",
        **kw,
    ) -> tuple["HarpoPlugin", Callable]:
        """Return (plugin, step_callback). Phase 2."""
        raise NotImplementedError(
            "CrewAI adapter is planned for Phase 2. "
            "See src/adapters/crewai/ for the stub."
        )

    @classmethod
    def for_autogen(
        cls,
        user_intent: str = "",
        **kw,
    ) -> tuple["HarpoPlugin", Callable]:
        """Return (plugin, reply_func). Phase 2."""
        raise NotImplementedError(
            "AutoGen adapter is planned for Phase 2. "
            "See src/adapters/autogen/ for the stub."
        )

    @classmethod
    def from_adapter(
        cls,
        adapter_name: str,
        runtime: Any,
        user_intent: str = "",
        **kw,
    ) -> "HarpoPlugin":
        """Generic factory using the adapter registry."""
        from harpo.sdk.registry import get_adapter
        plugin = cls(agent_id=adapter_name, user_intent=user_intent, **kw)
        adapter = get_adapter(adapter_name, sink=plugin._ingest)
        adapter.attach(runtime)
        return plugin

    # ── Core evaluation interface ────────────────────────────────────────────

    def evaluate(self) -> TrajectoryScores:
        """Score the trajectory on all 10 behavioral dimensions."""
        self._scores = self._evaluator.evaluate(self._trajectory)
        ctx = HookContext(
            trajectory = self._trajectory,
            scores     = self._scores,
        )
        self._hooks.run_post_trajectory(ctx)
        return self._scores

    def monitor(self) -> TrajectoryMonitor:
        """Return the live TrajectoryMonitor for real-time metric access."""
        return self._monitor

    def trajectory(self) -> AgentTrajectory:
        """Return the accumulated trajectory (may still be in-progress)."""
        return self._trajectory

    def export(self, fmt: str = "json") -> dict:
        """
        Export trajectory + scores.

        fmt: "json" | "prometheus" | "otel"
        Only "json" is implemented in Phase 1.
        """
        scores = self._scores or self.evaluate()
        base = {
            "trajectory_id": self._trajectory.trajectory_id,
            "agent_id":      self._trajectory.agent_id,
            "user_intent":   self._trajectory.user_intent,
            "status":        str(self._trajectory.status),
            "steps":         len(self._trajectory.steps),
            "duration_ms":   self._trajectory.duration_ms(),
            "scores": {
                dim: {
                    "score":       round(ds.value, 4),
                    "confidence":  round(ds.confidence, 4),
                    "explanation": ds.explanation,
                }
                for dim, ds in self._dimension_scores(scores)
            },
            "overall": round(scores.overall, 4),
        }
        if fmt == "json":
            return base
        if fmt == "prometheus":
            # Return flat {metric_name: value} for push gateway
            return {f"harpo_{dim}_score": v["score"]
                    for dim, v in base["scores"].items()}
        # otel / others: return as-is for now
        return base

    def compare(self, other: "HarpoPlugin") -> dict:
        """Compare this trajectory against another."""
        a = self._scores or self.evaluate()
        b = other._scores or other.evaluate()
        deltas = {}
        for dim, ds_a in self._dimension_scores(a):
            ds_b_dict = {d: s for d, s in other._dimension_scores(b)}
            ds_b = ds_b_dict.get(dim)
            if ds_b:
                deltas[dim] = round(ds_b.value - ds_a.value, 4)
        return {
            "baseline":  self._trajectory.trajectory_id,
            "candidate": other._trajectory.trajectory_id,
            "deltas":    deltas,
            "overall_delta": round(b.overall - a.overall, 4),
        }

    def report(self) -> dict:
        """Human-readable summary dict."""
        scores = self._scores or self.evaluate()
        monitor_snap = self._monitor.snapshot()
        return {
            "trajectory_id": self._trajectory.trajectory_id,
            "agent":         self._trajectory.agent_id,
            "steps":         len(self._trajectory.steps),
            "status":        str(self._trajectory.status),
            "overall_score": round(scores.overall, 4),
            "dimensions": {
                dim: round(ds.value, 4)
                for dim, ds in self._dimension_scores(scores)
            },
            "live_metrics":  monitor_snap.get("metrics", {}),
            "alerts":        [],   # populated when ObservabilityBridge is wired
        }

    # ── Internal event ingestion ─────────────────────────────────────────────

    def _ingest(self, event: GenericAgentEvent) -> None:
        """Called by the adapter for every translated event."""
        # Update run status on lifecycle events
        if event.event_type == GenericEventType.RUN_END:
            self._trajectory.status = (
                TrajectoryStatus.COMPLETED if event.success
                else TrajectoryStatus.FAILED
            )
            self._trajectory.ended_at = event.timestamp
            return
        if event.event_type == GenericEventType.RUN_START:
            return

        step = self._to_step(event)
        if step is None:
            return

        self._trajectory.add_step(step)
        self._step_index += 1

        self._monitor.ingest(step)
        ctx = HookContext(trajectory=self._trajectory, step=step)
        self._hooks.run_post_step(ctx)

    def _to_step(self, event: GenericAgentEvent) -> Optional[TrajectoryStep]:
        step_type = _EVENT_TO_STEP.get(event.event_type)
        if step_type is None:
            return None

        outcome = StepOutcome.SUCCESS if event.success else (
            StepOutcome.RETRY  if event.event_type == GenericEventType.REFLECT
            else StepOutcome.FAILURE
        )

        tool_call: Optional[ToolCall] = None
        if event.tool_call:
            tc = event.tool_call
            tool_call = ToolCall(
                name       = tc.tool_name,
                arguments  = tc.arguments,
                result     = tc.result,
                error      = tc.error,
                latency_ms = tc.duration_ms,
            )

        mem: Optional[MemoryAccess] = None
        if event.memory:
            m = event.memory
            mem = MemoryAccess(
                operation       = m.operation,
                key             = m.key or "",
                value           = m.content_summary,
                hit             = m.hit,
                relevance_score = m.relevance_score,
            )

        return TrajectoryStep(
            trajectory_id = self._trajectory.trajectory_id,
            turn_number   = event.turn_number,
            step_index    = self._step_index,
            step_type     = step_type,
            outcome       = outcome,
            input_text    = "",
            output_text   = event.text_output,
            raw_tokens    = event.tokens,
            timestamp     = event.timestamp,
            latency_ms    = event.latency_ms,
            tool_call     = tool_call,
            memory_access = mem,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _dimension_scores(scores: TrajectoryScores):
        """Yield (dimension_name, DimensionScore) pairs."""
        fields = [
            ("reasoning_stability",    scores.reasoning_stability),
            ("conversational_drift",   scores.conversational_drift),
            ("memory_utility",         scores.memory_utility),
            ("assumption_accumulation", scores.assumption_accumulation),
            ("recovery_ability",       scores.recovery_ability),
            ("collaboration_quality",  scores.collaboration_quality),
            ("reflection_usefulness",  scores.reflection_usefulness),
            ("long_horizon_reliability", scores.long_horizon_reliability),
            ("trajectory_coherence",   scores.trajectory_coherence),
            ("user_aligned_quality",   scores.user_aligned_quality),
        ]
        return fields
