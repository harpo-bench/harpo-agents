"""
Adapter registry — maps string names to adapter factory functions.

Ecosystem-specific imports are guarded so the SDK remains importable
even when only one adapter's runtime is installed.
"""

from __future__ import annotations

from typing import Any, Callable, Dict

from harpo.adapters.base import BaseAdapter
from harpo.core.events import GenericAgentEvent


# Registry: adapter_name → factory(sink) → BaseAdapter
_REGISTRY: Dict[str, Callable[..., BaseAdapter]] = {}


def register(name: str) -> Callable:
    """Decorator: register an adapter factory under a short name."""
    def _deco(factory_fn: Callable) -> Callable:
        _REGISTRY[name] = factory_fn
        return factory_fn
    return _deco


def get_adapter(name: str, sink: Callable[[GenericAgentEvent], None], **kw) -> BaseAdapter:
    """Instantiate an adapter by name. Raises KeyError for unknown names."""
    if name not in _REGISTRY:
        available = list(_REGISTRY.keys())
        raise KeyError(
            f"Unknown adapter '{name}'. Available: {available}. "
            "Ensure the adapter package is importable."
        )
    return _REGISTRY[name](sink=sink, **kw)


# ── Built-in registrations ───────────────────────────────────

@register("hive")
@register("open_hive")
def _hive_factory(sink, **kw):
    from harpo.adapters.open_hive.adapter import HiveAdapter
    return HiveAdapter(sink=sink, **kw)


@register("langgraph")
def _langgraph_factory(sink, **kw):
    try:
        from harpo.adapters.langgraph.adapter import LangGraphAdapter
        return LangGraphAdapter(sink=sink, **kw)
    except ImportError:
        raise ImportError("LangGraph adapter not yet implemented (Phase 2).")


@register("crewai")
def _crewai_factory(sink, **kw):
    try:
        from harpo.adapters.crewai.adapter import CrewAIAdapter
        return CrewAIAdapter(sink=sink, **kw)
    except ImportError:
        raise ImportError("CrewAI adapter not yet implemented (Phase 2).")


@register("autogen")
def _autogen_factory(sink, **kw):
    try:
        from harpo.adapters.autogen.adapter import AutoGenAdapter
        return AutoGenAdapter(sink=sink, **kw)
    except ImportError:
        raise ImportError("AutoGen adapter not yet implemented (Phase 2).")


@register("openhands")
def _openhands_factory(sink, **kw):
    try:
        from harpo.adapters.openhands.adapter import OpenHandsAdapter
        return OpenHandsAdapter(sink=sink, **kw)
    except ImportError:
        raise ImportError("OpenHands adapter not yet implemented (Phase 2).")
