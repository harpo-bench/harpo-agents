"""
HARPO Open-Hive Adapter

Subscribes to Hive's EventBus and translates AgentEvents into
GenericAgentEvents. Zero changes required to Hive source code.

Usage
-----
from harpo.adapters.open_hive import HiveAdapter
from harpo.sdk import HarpoPlugin

plugin = HarpoPlugin.for_hive(event_bus, user_intent="research AI safety")
# plugin is now live — evaluates as the task runs
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Callable, Optional

from harpo.adapters.base import BaseAdapter
from harpo.core.events import (
    GenericAgentEvent,
    GenericEventType,
    GenericMemoryAccess,
    GenericToolCall,
)
from .event_map import HIVE_TO_GENERIC, SUBSCRIBED_EVENT_TYPES, resolve_judge_verdict


class HiveAdapter(BaseAdapter):
    """
    Translates Hive AgentEvents → GenericAgentEvents.

    Tracks turn count internally: every LLM_TURN_COMPLETE increments the turn.
    """

    def __init__(self, sink: Callable[[GenericAgentEvent], None],
                 agent_id: str = "hive-agent",
                 run_id: str = "") -> None:
        super().__init__(sink)
        self._agent_id = agent_id
        self._run_id   = run_id or str(uuid.uuid4())
        self._turn     = 0
        # Accumulates streamed text from LLM_TEXT_DELTA events
        self._text_snapshot: str = ""
        # Token accumulator from deltas (some Hive builds send token count per delta)
        self._token_accumulator: int = 0
        # Per-turn sequence counter for within-turn event ordering
        self._seq_in_turn: int = 0
        # Last emitted event_id for parent lineage tracking
        self._last_event_id: Optional[str] = None

    # ── Public ──────────────────────────────────────────────────

    def attach(self, event_bus: Any) -> None:
        """
        Subscribe to the Hive EventBus.

        Works with both the real Hive EventBus (uses HiveEventType enum values)
        and MockHiveEventBus / any bus that accepts plain string event types.
        Hive itself does not need to be importable for mock/test usage.
        """
        try:
            from framework.host.event_bus import EventType as HiveEventType
            subscribed = [getattr(HiveEventType, et.upper(), None)
                          for et in SUBSCRIBED_EVENT_TYPES]
            subscribed = [et for et in subscribed if et is not None]
        except ImportError:
            # Hive not on path — use raw string values (works with MockHiveEventBus)
            subscribed = list(SUBSCRIBED_EVENT_TYPES)

        event_bus.subscribe(event_types=subscribed, handler=self._handle)

    # ── Internal ─────────────────────────────────────────────────

    def _handle(self, event: Any) -> None:
        self._emit(event)

    def _to_generic(self, event: Any) -> Optional[GenericAgentEvent]:
        etype_str: str = str(event.type).replace("EventType.", "").lower()
        data: dict = event.data if hasattr(event, "data") else {}
        ts: float = (event.timestamp.timestamp()
                     if hasattr(event.timestamp, "timestamp")
                     else time.time())
        node_id: str = event.node_id or ""

        # ── Text delta accumulator (streaming) ───────────────
        # llm_text_delta: worker streams; client_output_delta: queen/interactive streams.
        # Both have identical data structure: {content, snapshot, inner_turn}
        if etype_str in ("llm_text_delta", "client_output_delta"):
            self._text_snapshot = data.get("snapshot", data.get("content", ""))
            # Some Hive builds send incremental token counts on each delta
            delta_toks = (data.get("token_count") or data.get("tokens") or
                          data.get("output_tokens") or 0)
            if delta_toks:
                self._token_accumulator += int(delta_toks)
            return None

        # ── Turn counter ──────────────────────────────────────
        if etype_str == "llm_turn_complete":
            self._turn += 1
            self._seq_in_turn = 0

        # ── Judge verdict (context-dependent mapping) ─────────
        if etype_str == "judge_verdict":
            verdict = data.get("verdict", data.get("action", "ACCEPT"))
            gtype, success = resolve_judge_verdict(str(verdict))
            evt = GenericAgentEvent(
                agent_id         = self._agent_id,
                run_id           = self._run_id,
                event_type       = gtype,
                timestamp        = ts,
                turn_number      = self._turn,
                text_output      = data.get("feedback", data.get("reason", "")),
                success          = success,
                parent_event_id  = self._last_event_id,
                sequence_in_turn = self._seq_in_turn,
                raw              = data,
            )
            self._last_event_id = evt.event_id
            self._seq_in_turn  += 1
            return evt

        gtype = HIVE_TO_GENERIC.get(etype_str)
        if gtype is None:
            return None

        # ── Run lifecycle ────────────────────────────────────
        if gtype == GenericEventType.RUN_START:
            return GenericAgentEvent(
                agent_id   = self._agent_id,
                run_id     = self._run_id,
                event_type = gtype,
                timestamp  = ts,
                success    = True,
                raw        = data,
            )

        if gtype == GenericEventType.RUN_END:
            success = etype_str in ("execution_completed", "node_loop_completed")
            return GenericAgentEvent(
                agent_id   = self._agent_id,
                run_id     = self._run_id,
                event_type = gtype,
                timestamp  = ts,
                success    = success,
                error      = data.get("error") if not success else None,
                raw        = data,
            )

        # ── THINK (LLM turn) ─────────────────────────────────
        if gtype == GenericEventType.THINK:
            text = (self._text_snapshot
                    or data.get("text", data.get("output", "")))
            self._text_snapshot = ""

            # Resolve token count: try all known Hive field names, then use
            # the delta accumulator, then fall back to word-count estimate.
            tokens = (
                data.get("output_tokens")
                or data.get("token_count")
                or data.get("tokens")
                or data.get("completion_tokens")
                or data.get("total_tokens")
                or self._token_accumulator
                or (int(len(text.split()) * 1.3) if text else 0)
            )
            self._token_accumulator = 0  # reset accumulator after consumption

            evt = GenericAgentEvent(
                agent_id         = self._agent_id,
                run_id           = self._run_id,
                event_type       = gtype,
                timestamp        = ts,
                turn_number      = self._turn,
                text_output      = text,
                tokens           = int(tokens),
                latency_ms       = float(data.get("latency_ms", 0)),
                parent_event_id  = self._last_event_id,
                sequence_in_turn = self._seq_in_turn,
                raw              = data,
            )
            self._last_event_id = evt.event_id
            self._seq_in_turn  += 1
            return evt

        # ── TOOL_USE ─────────────────────────────────────────
        if gtype == GenericEventType.TOOL_USE:
            tc = GenericToolCall(
                tool_name   = data.get("tool_name", node_id),
                arguments   = data.get("arguments", data.get("tool_input", {})),
                result      = str(data.get("result", "")),
                error       = data.get("error"),
                duration_ms = float(data.get("duration_ms", data.get("latency_ms", 0))),
            )
            evt = GenericAgentEvent(
                agent_id         = self._agent_id,
                run_id           = self._run_id,
                event_type       = gtype,
                timestamp        = ts,
                turn_number      = self._turn,
                success          = tc.error is None,
                tool_call        = tc,
                latency_ms       = tc.duration_ms,
                parent_event_id  = self._last_event_id,
                sequence_in_turn = self._seq_in_turn,
                raw              = data,
            )
            self._last_event_id = evt.event_id
            self._seq_in_turn  += 1
            return evt

        # ── MEMORY_READ (context compaction) ─────────────────
        if gtype == GenericEventType.MEMORY_READ:
            mem = GenericMemoryAccess(
                operation       = "compact",
                hit             = True,
                relevance_score = 1.0,
                content_summary = data.get("summary", "context compacted"),
            )
            return GenericAgentEvent(
                agent_id    = self._agent_id,
                run_id      = self._run_id,
                event_type  = gtype,
                timestamp   = ts,
                turn_number = self._turn,
                memory      = mem,
                raw         = data,
            )

        # ── RECOVER ──────────────────────────────────────────
        if gtype == GenericEventType.RECOVER:
            return GenericAgentEvent(
                agent_id    = self._agent_id,
                run_id      = self._run_id,
                event_type  = gtype,
                timestamp   = ts,
                turn_number = self._turn,
                text_output = data.get("reason", ""),
                success     = False,
                raw         = data,
            )

        # ── HAND_OFF ─────────────────────────────────────────
        if gtype == GenericEventType.HAND_OFF:
            return GenericAgentEvent(
                agent_id    = self._agent_id,
                run_id      = self._run_id,
                event_type  = gtype,
                timestamp   = ts,
                turn_number = self._turn,
                text_output = data.get("message", ""),
                raw         = data,
            )

        # ── OBSERVATION (subagent report) ────────────────────
        if gtype == GenericEventType.OBSERVATION:
            return GenericAgentEvent(
                agent_id    = self._agent_id,
                run_id      = self._run_id,
                event_type  = gtype,
                timestamp   = ts,
                turn_number = self._turn,
                text_output = str(data.get("report", data.get("output", ""))),
                raw         = data,
            )

        # ── ERROR ────────────────────────────────────────────
        if gtype == GenericEventType.ERROR:
            return GenericAgentEvent(
                agent_id    = self._agent_id,
                run_id      = self._run_id,
                event_type  = gtype,
                timestamp   = ts,
                turn_number = self._turn,
                error       = data.get("error", etype_str),
                success     = False,
                raw         = data,
            )

        # ── Fallback ─────────────────────────────────────────
        return GenericAgentEvent(
            agent_id    = self._agent_id,
            run_id      = self._run_id,
            event_type  = gtype,
            timestamp   = ts,
            turn_number = self._turn,
            text_output = str(data),
            raw         = data,
        )
