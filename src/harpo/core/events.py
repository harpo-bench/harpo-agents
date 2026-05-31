"""
HARPO Universal Event Abstraction

All ecosystem adapters (Open-Hive, LangGraph, CrewAI, AutoGen, OpenHands, ...)
translate their native events into GenericAgentEvent. The evaluation pipeline
never sees ecosystem-specific types — only this module's types.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class GenericEventType(str, Enum):
    THINK        = "think"        # LLM reasoning output
    TOOL_USE     = "tool_use"     # Tool invocation
    OBSERVATION  = "observation"  # Tool result / external data returned
    RESPOND      = "respond"      # Final agent response to user / caller
    REFLECT      = "reflect"      # Self-evaluation / critique (RETRY / JUDGE trigger)
    RECOVER      = "recover"      # Retry after failure
    MEMORY_READ  = "memory_read"  # Memory retrieval / context compaction
    MEMORY_WRITE = "memory_write" # Memory store / update
    HAND_OFF     = "hand_off"     # Agent-to-agent delegation
    RUN_START    = "run_start"    # Execution begins
    RUN_END      = "run_end"      # Execution ends (success or failure)
    ERROR        = "error"        # Unrecoverable error


@dataclass
class GenericToolCall:
    """Normalised tool call payload — produced by adapter, consumed by pipeline."""
    tool_name:   str
    arguments:   Dict[str, Any]
    result:      str
    error:       Optional[str]  = None
    duration_ms: float          = 0.0


@dataclass
class GenericMemoryAccess:
    """Normalised memory operation — read, write, or compaction event."""
    operation:       str    # "read" | "write" | "compact"
    hit:             bool   = True
    relevance_score: float  = 1.0
    key:             Optional[str] = None
    content_summary: str    = ""


@dataclass
class GenericAgentEvent:
    """
    Universal event emitted by every adapter.

    The evaluation pipeline and observability layer operate exclusively on
    GenericAgentEvents — no ecosystem-specific types ever cross that boundary.
    """
    event_id:    str             = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id:    str             = ""
    run_id:      str             = ""
    event_type:  GenericEventType = GenericEventType.THINK
    timestamp:   float           = field(default_factory=time.time)
    turn_number: int             = 0
    text_output: str             = ""
    tokens:      int             = 0
    latency_ms:  float           = 0.0

    tool_call:   Optional[GenericToolCall]    = None
    memory:      Optional[GenericMemoryAccess] = None
    error:       Optional[str]               = None
    success:     bool                        = True

    # Event lineage — causal chain tracing
    # parent_event_id: the event_id that directly caused / triggered this event
    # (e.g. a TOOL_USE triggers an OBSERVATION; a REFLECT triggers a RECOVER)
    parent_event_id: Optional[str] = None
    # sequence_in_turn: 0-based position within the same turn_number
    # (resolves ordering when multiple events share the same timestamp)
    sequence_in_turn: int = 0

    # Original ecosystem event — kept for debugging; never read by the core
    raw: Dict[str, Any] = field(default_factory=dict)
