"""
Hive EventType → HARPO GenericEventType mapping table.

This module is the single source of truth for the Open-Hive translation.
It is intentionally separated from the adapter so the mapping can be
inspected, tested, and extended without touching the subscription logic.
"""

from __future__ import annotations

from typing import Optional

# Lazy import — Hive is only needed if you actually call attach()
try:
    from framework.host.event_bus import EventType as HiveEventType
    HIVE_AVAILABLE = True
except ImportError:
    HiveEventType = None        # type: ignore
    HIVE_AVAILABLE = False

from harpo.core.events import GenericEventType


# Maps each Hive EventType string value → GenericEventType (or None = skip)
HIVE_TO_GENERIC: dict[str, Optional[GenericEventType]] = {
    # LLM reasoning turn (text comes from llm_text_delta snapshots, not this event)
    "llm_turn_complete":        GenericEventType.THINK,
    # Text streaming deltas — captured internally, not emitted as GenericAgentEvent.
    # llm_text_delta: worker streams; client_output_delta: queen/interactive streams.
    "llm_text_delta":           None,
    "client_output_delta":      None,

    # Tool use
    "tool_call_started":        None,              # skip — wait for completed
    "tool_call_completed":      GenericEventType.TOOL_USE,

    # Judge decisions
    # verdict is inspected at runtime: ACCEPT→RESPOND, RETRY→REFLECT, ESCALATE→REFLECT(fail)
    "judge_verdict":            None,              # handled inline in adapter

    # Retry / recovery
    "node_retry":               GenericEventType.RECOVER,

    # Context management (compaction = implicit memory read)
    "context_compacted":        GenericEventType.MEMORY_READ,

    # Sub-agent reports (async observation from a worker)
    "subagent_report":          GenericEventType.OBSERVATION,

    # Escalation = handoff to human / queen
    "escalation_requested":     GenericEventType.HAND_OFF,

    # Run lifecycle — high-level orchestrator events
    "execution_started":        GenericEventType.RUN_START,
    "execution_completed":      GenericEventType.RUN_END,
    "execution_failed":         GenericEventType.RUN_END,

    # AgentLoop lifecycle — emitted directly by AgentLoop
    "node_loop_started":        GenericEventType.RUN_START,
    "node_loop_completed":      GenericEventType.RUN_END,

    # Doom-loop / stall detection → error signal
    "node_tool_doom_loop":      GenericEventType.ERROR,
    "node_stalled":             GenericEventType.ERROR,

    # Stream health signals — map to REFLECT (agent needs to adjust)
    "stream_nudge_sent":        GenericEventType.REFLECT,
    "tool_call_replay_detected": GenericEventType.REFLECT,
}


# EventTypes the adapter subscribes to:
# - non-None mappings emit GenericAgentEvents
# - judge_verdict and llm_text_delta are handled inline (text accumulation)
SUBSCRIBED_EVENT_TYPES = [
    k for k, v in HIVE_TO_GENERIC.items()
    if v is not None or k in ("judge_verdict", "llm_text_delta", "client_output_delta")
]


def resolve_judge_verdict(verdict_action: str) -> tuple[GenericEventType, bool]:
    """
    Return (generic_event_type, success_flag) for a JUDGE_VERDICT event.

    ACCEPT  → RESPOND   (success)
    RETRY   → REFLECT   (not success — agent revising)
    ESCALATE→ REFLECT   (not success — escalating)
    CONTINUE→ THINK     (success — keep going)
    """
    v = verdict_action.upper()
    if v == "ACCEPT":
        return GenericEventType.RESPOND, True
    if v == "CONTINUE":
        return GenericEventType.THINK, True
    return GenericEventType.REFLECT, False
