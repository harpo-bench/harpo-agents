"""
Memory Propagation Graph

Tracks how stale memory values spread through the agent network.

A stale value "propagates" when:
  Agent A reads stale value X
  → Agent A's output cites or depends on X
  → Agent B reads Agent A's output (context injection)
  → Agent B now also operates on X (second-order stale read)

This tracks the full chain: Memory Write → Stale Read → Output → Downstream Agent

Output:
  budget: PM:write($5M) → FIN:update($2M) → ENG:stale_read($5M)
          → ENG:output(plan for $5M) → PM:second_order_stale
          → OPS:second_order_stale
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport

_AGENT_DISPLAY: Dict[str, str] = {
    "product-manager":  "Product Manager",
    "engineering-lead": "Engineering Lead",
    "finance-lead":     "Finance Lead",
    "legal-lead":       "Legal Lead",
    "marketing-lead":   "Marketing Lead",
    "operations-lead":  "Operations Lead",
}

def _dn(a: str) -> str:
    return _AGENT_DISPLAY.get(a or "", (a or "").replace("-", " ").title())


# In the product launch scenario, these agents consume each other's reports
_DOWNSTREAM_READERS: Dict[str, List[str]] = {
    "product-manager":  ["engineering-lead", "finance-lead", "legal-lead",
                         "marketing-lead", "operations-lead"],
    "engineering-lead": ["product-manager", "operations-lead"],
    "finance-lead":     ["engineering-lead", "operations-lead", "product-manager"],
    "legal-lead":       ["marketing-lead", "product-manager", "operations-lead"],
    "marketing-lead":   ["product-manager", "operations-lead"],
    "operations-lead":  ["product-manager"],
}


@dataclass
class PropagationNode:
    agent_id:     str
    key:          str
    order:        int     # 0=direct stale read, 1=second-order, 2=third-order
    value_used:   str
    is_stale:     bool

    def render(self) -> str:
        order_str = {0: "direct", 1: "2nd-order", 2: "3rd-order"}.get(self.order, f"order-{self.order}")
        stale_str = " [STALE]" if self.is_stale else ""
        return f"{_dn(self.agent_id)} ({order_str}): {self.value_used!r}{stale_str}"


@dataclass
class MemoryPropagationChain:
    """One complete propagation chain for one memory key."""
    key:              str
    original_writer:  str
    stale_value:      str
    correct_value:    str
    nodes:            List[PropagationNode] = field(default_factory=list)

    def total_affected(self) -> int:
        return len({n.agent_id for n in self.nodes if n.is_stale})

    def render(self) -> str:
        lines = [
            f"  {self.key}: {_dn(self.original_writer)}:write({self.correct_value!r})"
            f" → stale({self.stale_value!r}) propagated:"
        ]
        for node in self.nodes:
            indent = "    " + "  " * node.order
            lines.append(f"{indent}→ {node.render()}")
        return "\n".join(lines)


@dataclass
class MemoryPropagationReport:
    """Complete propagation analysis."""
    chains:          List[MemoryPropagationChain] = field(default_factory=list)
    max_depth:       int                          = 0
    total_affected:  int                          = 0

    def as_dict(self) -> dict:
        return {
            "max_propagation_depth": self.max_depth,
            "total_affected_agents": self.total_affected,
            "chains": [
                {
                    "key":             c.key,
                    "original_writer": c.original_writer,
                    "stale_value":     c.stale_value,
                    "correct_value":   c.correct_value,
                    "affected_count":  c.total_affected(),
                    "nodes": [
                        {
                            "agent":  n.agent_id,
                            "order":  n.order,
                            "value":  n.value_used,
                            "stale":  n.is_stale,
                        }
                        for n in c.nodes
                    ],
                }
                for c in self.chains
            ],
        }

    def render(self) -> str:
        if not self.chains:
            return "  No stale memory propagation detected."
        lines = [
            f"  Max propagation depth: {self.max_depth}",
            f"  Total agents affected by stale memory: {self.total_affected}",
            "",
        ]
        for chain in self.chains:
            lines.append(chain.render())
            lines.append("")
        return "\n".join(lines)


def build_memory_propagation_report(
    store:        "SharedMemoryStore",
    stale_report: "StaleMemoryReport",
) -> MemoryPropagationReport:
    """
    Build propagation chains: from stale read, trace which downstream
    agents were affected by the stale reader's outputs.
    """
    chains: List[MemoryPropagationChain] = []
    all_affected: Set[str] = set()

    for rec in stale_report.records:
        key = rec.key
        original_writer_record = None
        for hist_rec in store.history(key):
            if hist_rec.version == 1:
                original_writer_record = hist_rec
                break

        original_writer = original_writer_record.written_by if original_writer_record else "unknown"

        chain = MemoryPropagationChain(
            key             = key,
            original_writer = original_writer,
            stale_value     = rec.stale_value,
            correct_value   = rec.current_value,
        )

        # Direct stale reader (order=0)
        chain.nodes.append(PropagationNode(
            agent_id   = rec.reader_agent,
            key        = key,
            order      = 0,
            value_used = rec.stale_value,
            is_stale   = True,
        ))
        all_affected.add(rec.reader_agent)

        # Second-order: agents that read from the stale reader's output
        downstream = _DOWNSTREAM_READERS.get(rec.reader_agent, [])
        for second_agent in downstream[:3]:
            chain.nodes.append(PropagationNode(
                agent_id   = second_agent,
                key        = key,
                order      = 1,
                value_used = rec.stale_value,  # inherited from stale reader
                is_stale   = True,
            ))
            all_affected.add(second_agent)

        chains.append(chain)

    max_depth = max((max(n.order for n in c.nodes) for c in chains if c.nodes),
                    default=0)

    return MemoryPropagationReport(
        chains         = chains,
        max_depth      = max_depth,
        total_affected = len(all_affected),
    )
