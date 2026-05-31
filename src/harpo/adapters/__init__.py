"""
HARPO Ecosystem Adapters

Each sub-package translates a specific agent framework's native events into
GenericAgentEvents consumed by the universal evaluation pipeline.

Available adapters
------------------
open_hive  — Open-Hive EventBus (Phase 1, fully implemented)
langgraph  — LangChain / LangGraph callbacks (Phase 2)
crewai     — CrewAI step_callback (Phase 2)
autogen    — AutoGen register_reply (Phase 2)
openhands  — OpenHands Action/Observation (Phase 2)

All adapters subclass BaseAdapter from harpo.adapters.base.
"""

from .base import BaseAdapter

__all__ = ["BaseAdapter"]
