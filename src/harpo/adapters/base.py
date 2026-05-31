"""
HARPO BaseAdapter

Every ecosystem adapter subclasses this. The contract:
- attach(runtime) — subscribe to the runtime's event stream (zero changes to runtime)
- _to_generic(native_event) — translate one native event → GenericAgentEvent or None

Adapters call self._emit(native_event) from their handler; BaseAdapter handles
the translate-then-sink dance.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

from harpo.core.events import GenericAgentEvent


class BaseAdapter(ABC):
    """
    Attach to an agent runtime and feed GenericAgentEvents into HARPO.

    Parameters
    ----------
    sink : callable
        Called with every successfully translated GenericAgentEvent.
        Typically HarpoPlugin._ingest.
    """

    def __init__(self, sink: Callable[[GenericAgentEvent], None]) -> None:
        self._sink = sink

    @abstractmethod
    def attach(self, runtime: Any) -> None:
        """
        Subscribe to the runtime's event stream.
        Must not require changes to the runtime's source code.
        """

    @abstractmethod
    def _to_generic(self, native_event: Any) -> Optional[GenericAgentEvent]:
        """
        Translate one native event into GenericAgentEvent.
        Return None to silently skip events that don't map.
        """

    def _emit(self, native_event: Any) -> None:
        """Translate and forward to sink. Safe to call from any handler."""
        try:
            evt = self._to_generic(native_event)
        except Exception:
            return
        if evt is not None:
            self._sink(evt)
