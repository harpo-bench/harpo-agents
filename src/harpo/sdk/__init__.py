"""
HARPO SDK — universal plugin API.

    import harpo

    # Attach to Open-Hive
    plugin = harpo.attach(event_bus, adapter="hive", user_intent="research X")

    # After agent run completes:
    print(plugin.report())
    print(plugin.export("json"))
"""

from .plugin import HarpoPlugin
from .registry import get_adapter, register


def attach(runtime, *, adapter: str, user_intent: str = "", **kw) -> HarpoPlugin:
    """
    One-line integration.

    Parameters
    ----------
    runtime      : ecosystem runtime object (EventBus, graph, crew, ...)
    adapter      : short name — "hive", "langgraph", "crewai", "autogen", "openhands"
    user_intent  : task description for alignment scoring
    **kw         : passed through to the adapter factory

    Returns
    -------
    HarpoPlugin  : live plugin, already subscribed to the runtime
    """
    plugin = HarpoPlugin(agent_id=adapter, user_intent=user_intent)
    adapter_instance = get_adapter(adapter, sink=plugin._ingest, **kw)
    adapter_instance.attach(runtime)
    return plugin


__all__ = ["HarpoPlugin", "attach", "get_adapter", "register"]
