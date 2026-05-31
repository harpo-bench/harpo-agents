"""
Shared Memory Store

A versioned, observable shared memory system for multi-agent trajectories.
Every read/write/update/invalidate operation:
  1. Updates the in-memory store
  2. Emits a TrajectoryStep (MEMORY_READ or MEMORY_WRITE) into the agent's
     HARPO plugin — making memory a first-class observable event

Stale read detection:
  A read is "stale" when the version read is older than the current version
  (i.e., another agent updated the value since the reader last synced).

Usage in demo
-------------
    store = SharedMemoryStore()

    # Product Manager writes initial values
    store.write("budget", "$5M", agent_id="product-manager", plugin=pm_plugin)

    # Finance updates budget
    store.update("budget", "$2M", agent_id="finance-lead", plugin=fin_plugin)

    # Engineering reads — stale if it reads version 1 (= $5M)
    obj = store.read("budget", agent_id="engineering-lead",
                     plugin=eng_plugin, force_version=1)  # force stale read
    # obj.is_stale == True, obj.value == "$5M"
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.sdk.plugin import HarpoPlugin

# ── Memory record ─────────────────────────────────────────────────────────────

@dataclass
class MemoryRecord:
    """One version of a memory object."""
    key:        str
    value:      Any
    version:    int
    written_by: str
    operation:  str         # "write" | "update" | "invalidate"
    timestamp:  float       = field(default_factory=time.time)
    step_id:    str         = field(default_factory=lambda: str(uuid.uuid4()))

    def is_invalidated(self) -> bool:
        return self.operation == "invalidate"


# ── Read event log ────────────────────────────────────────────────────────────

@dataclass
class MemoryReadEvent:
    """Records one read operation against the store."""
    key:             str
    value_read:      Any
    version_read:    int
    current_version: int
    reader_agent:    str
    is_stale:        bool
    timestamp:       float = field(default_factory=time.time)
    step_id:         str   = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def version_lag(self) -> int:
        return self.current_version - self.version_read


# ── Store ─────────────────────────────────────────────────────────────────────

class SharedMemoryStore:
    """
    Versioned shared memory store with HARPO event emission.

    Thread-safety: NOT thread-safe. Demo agents run sequentially.
    """

    def __init__(self) -> None:
        # key → list of MemoryRecord (append-only history)
        self._store:     Dict[str, List[MemoryRecord]] = {}
        # all read events
        self._reads:     List[MemoryReadEvent]         = []
        # all write/update events (all MemoryRecord operations)
        self._writes:    List[MemoryRecord]            = []

    # ── Public API ────────────────────────────────────────────────────────────

    def write(self, key: str, value: Any, agent_id: str,
              plugin: Optional["HarpoPlugin"] = None) -> MemoryRecord:
        """Write a new key (version 1) or re-initialize an existing key."""
        version = 1
        if key in self._store:
            version = self._store[key][-1].version + 1

        record = MemoryRecord(
            key        = key,
            value      = value,
            version    = version,
            written_by = agent_id,
            operation  = "write" if version == 1 else "update",
        )
        self._store.setdefault(key, []).append(record)
        self._writes.append(record)

        self._emit_write(key, value, version, agent_id, "write", plugin)
        return record

    def update(self, key: str, new_value: Any, agent_id: str,
               plugin: Optional["HarpoPlugin"] = None) -> MemoryRecord:
        """Update an existing key to a new version."""
        current = self.current(key)
        new_version = (current.version + 1) if current else 1

        record = MemoryRecord(
            key        = key,
            value      = new_value,
            version    = new_version,
            written_by = agent_id,
            operation  = "update",
        )
        self._store.setdefault(key, []).append(record)
        self._writes.append(record)

        self._emit_write(key, new_value, new_version, agent_id, "update", plugin)
        return record

    def invalidate(self, key: str, agent_id: str,
                   plugin: Optional["HarpoPlugin"] = None) -> None:
        """Mark a key as invalidated (value no longer valid)."""
        current = self.current(key)
        if not current:
            return
        new_version = current.version + 1
        record = MemoryRecord(
            key        = key,
            value      = None,
            version    = new_version,
            written_by = agent_id,
            operation  = "invalidate",
        )
        self._store.setdefault(key, []).append(record)
        self._writes.append(record)
        self._emit_write(key, "[INVALIDATED]", new_version, agent_id, "invalidate", plugin)

    def read(self, key: str, agent_id: str,
             plugin: Optional["HarpoPlugin"] = None,
             force_version: Optional[int] = None) -> Optional[MemoryRecord]:
        """
        Read a key from the store.

        force_version: if set, return that specific version even if a newer
                       one exists — simulates a stale read (agent hasn't
                       received the latest update).
        """
        history = self._store.get(key, [])
        if not history:
            self._emit_miss(key, agent_id, plugin)
            return None

        current_record = history[-1]
        current_version = current_record.version

        if force_version is not None:
            # Find the specific version
            record = next((r for r in history if r.version == force_version), None)
            if record is None:
                record = history[-1]   # fallback to current
        else:
            record = current_record

        is_stale = record.version < current_version

        read_event = MemoryReadEvent(
            key             = key,
            value_read      = record.value,
            version_read    = record.version,
            current_version = current_version,
            reader_agent    = agent_id,
            is_stale        = is_stale,
        )
        self._reads.append(read_event)

        self._emit_read(key, record.value, record.version, current_version,
                        is_stale, agent_id, plugin)
        return record

    def current(self, key: str) -> Optional[MemoryRecord]:
        """Return the current (latest) version of a key."""
        history = self._store.get(key, [])
        return history[-1] if history else None

    def history(self, key: str) -> List[MemoryRecord]:
        """Return full version history for a key."""
        return list(self._store.get(key, []))

    # ── Query methods ─────────────────────────────────────────────────────────

    def stale_reads(self) -> List[MemoryReadEvent]:
        return [r for r in self._reads if r.is_stale]

    def all_reads(self) -> List[MemoryReadEvent]:
        return list(self._reads)

    def all_writes(self) -> List[MemoryRecord]:
        return list(self._writes)

    def all_keys(self) -> List[str]:
        return list(self._store.keys())

    def summary(self) -> dict:
        return {
            "keys":         self.all_keys(),
            "total_writes": len(self._writes),
            "total_reads":  len(self._reads),
            "stale_reads":  len(self.stale_reads()),
            "per_key": {
                key: {
                    "current_version": self.current(key).version if self.current(key) else 0,
                    "current_value":   str(self.current(key).value)[:60] if self.current(key) else None,
                    "history_len":     len(self.history(key)),
                }
                for key in self.all_keys()
            },
        }

    # ── HARPO event emission ──────────────────────────────────────────────────

    def _emit_write(self, key: str, value: Any, version: int,
                    agent_id: str, operation: str,
                    plugin: Optional["HarpoPlugin"]) -> None:
        if plugin is None:
            return
        from harpo.trajectory.schema import StepType, StepOutcome, MemoryAccess, TrajectoryStep
        step = TrajectoryStep(
            trajectory_id = plugin.trajectory().trajectory_id,
            turn_number   = len(plugin.trajectory().steps) // 4 + 1,
            step_index    = len(plugin.trajectory().steps),
            step_type     = StepType.MEMORY_WRITE,
            outcome       = StepOutcome.SUCCESS,
            input_text    = "",
            output_text   = f"[{operation.upper()}] {key} = {value!r} (version {version})",
            timestamp     = time.time(),
            agent_id      = agent_id,
            memory_access = MemoryAccess(
                operation       = operation,
                key             = key,
                value           = str(value),
                hit             = True,
                relevance_score = 1.0,
                version         = version,
                is_stale        = False,
                current_version = version,
            ),
        )
        plugin.trajectory().add_step(step)

    def _emit_read(self, key: str, value: Any, version: int,
                   current_version: int, is_stale: bool,
                   agent_id: str, plugin: Optional["HarpoPlugin"]) -> None:
        if plugin is None:
            return
        from harpo.trajectory.schema import StepType, StepOutcome, MemoryAccess, TrajectoryStep
        relevance = 0.2 if is_stale else 1.0
        stale_note = f" [STALE: v{version} < current v{current_version}]" if is_stale else ""
        step = TrajectoryStep(
            trajectory_id = plugin.trajectory().trajectory_id,
            turn_number   = len(plugin.trajectory().steps) // 4 + 1,
            step_index    = len(plugin.trajectory().steps),
            step_type     = StepType.MEMORY_READ,
            outcome       = StepOutcome.SUCCESS,
            input_text    = "",
            output_text   = f"[READ] {key} = {value!r} (v{version}){stale_note}",
            timestamp     = time.time(),
            agent_id      = agent_id,
            memory_access = MemoryAccess(
                operation       = "read",
                key             = key,
                value           = str(value),
                hit             = True,
                relevance_score = relevance,
                version         = version,
                is_stale        = is_stale,
                current_version = current_version,
            ),
        )
        plugin.trajectory().add_step(step)

    def _emit_miss(self, key: str, agent_id: str,
                   plugin: Optional["HarpoPlugin"]) -> None:
        if plugin is None:
            return
        from harpo.trajectory.schema import StepType, StepOutcome, MemoryAccess, TrajectoryStep
        step = TrajectoryStep(
            trajectory_id = plugin.trajectory().trajectory_id,
            turn_number   = len(plugin.trajectory().steps) // 4 + 1,
            step_index    = len(plugin.trajectory().steps),
            step_type     = StepType.MEMORY_READ,
            outcome       = StepOutcome.FAILURE,
            input_text    = "",
            output_text   = f"[MISS] {key} not found in memory store",
            timestamp     = time.time(),
            agent_id      = agent_id,
            memory_access = MemoryAccess(
                operation       = "read",
                key             = key,
                value           = None,
                hit             = False,
                relevance_score = 0.0,
            ),
        )
        plugin.trajectory().add_step(step)
