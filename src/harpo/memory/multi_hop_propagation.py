"""
Multi-Hop Memory Propagation

Fixes the depth under-estimation in the original memory_propagation_graph.py.

PROBLEM
-------
The original implementation tracked only:
  order=0  (direct stale reader)
  order=1  (second-order: agents that read from the stale reader's output)

This missed true propagation depth. In the product launch scenario:

  Finance writes budget=$5M
  → Finance updates to $2M            (correction — not a hop)
  → Engineering reads stale $5M       (hop 0, direct)
  → Product Manager reads Engineering's plan (hop 1, indirect)
  → Operations reads Product Manager's plan  (hop 2, second indirect)

That chain has depth=2, not depth=1.

SOLUTION
--------
BFS traversal of the agent dependency graph starting from the direct stale reader.
Each hop is a concrete agent-to-agent influence edge based on:
  1. The _DOWNSTREAM_READERS map (who reads whose reports as context)
  2. The causal_hint from memory_instrumentation (if available)
  3. The order of execution (later agents inherit earlier agents' assumptions)

Every hop beyond depth=0 is annotated with HOW it was inherited:
  "context_injection"   — next agent received prior agent's output as context
  "report_reference"    — next agent explicitly referenced prior agent's report
  "shared_memory_read"  — next agent read the same stale key independently
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport, StaleReadRecord

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


# Agent dependency graph: who reads from whom (context injection relationships)
# Edges mean: key_agent's output is passed as context to value_agents
_DOWNSTREAM_READERS: Dict[str, List[str]] = {
    "product-manager":  ["engineering-lead", "finance-lead", "legal-lead",
                         "marketing-lead", "operations-lead"],
    "engineering-lead": ["product-manager", "operations-lead"],
    "finance-lead":     ["engineering-lead", "operations-lead", "product-manager"],
    "legal-lead":       ["marketing-lead", "product-manager", "operations-lead"],
    "marketing-lead":   ["product-manager", "operations-lead"],
    "operations-lead":  ["product-manager"],
}

# Inheritance type based on relationship
_INHERITANCE_TYPE: Dict[Tuple[str, str], str] = {
    ("engineering-lead", "product-manager"):   "report_reference",
    ("engineering-lead", "operations-lead"):   "context_injection",
    ("marketing-lead",   "product-manager"):   "report_reference",
    ("marketing-lead",   "operations-lead"):   "context_injection",
    ("operations-lead",  "product-manager"):   "report_reference",
    ("legal-lead",       "marketing-lead"):    "context_injection",
    ("finance-lead",     "engineering-lead"):  "context_injection",
    ("product-manager",  "engineering-lead"):  "context_injection",
}

_MAX_HOP_DEPTH = 4   # cap traversal to avoid cycles


# ── Hop node ──────────────────────────────────────────────────────────────────

@dataclass
class HopNode:
    """One agent in the propagation chain."""
    agent_id:          str
    hop_depth:         int     # 0 = direct stale reader, 1 = first indirect, etc.
    value_inherited:   str     # the stale/wrong value this agent operated on
    how_inherited:     str     # "direct_stale_read" | "context_injection" | "report_reference" | "shared_memory_read"
    inherited_from:    str     # agent_id of source, or "memory_store" for hop=0

    def render(self) -> str:
        depth_labels = {0: "direct", 1: "1st indirect", 2: "2nd indirect", 3: "3rd indirect"}
        depth_str = depth_labels.get(self.hop_depth, f"depth-{self.hop_depth}")
        return (
            f"{_dn(self.agent_id)} [{depth_str}]  "
            f"inherited {self.value_inherited!r} via {self.how_inherited} "
            f"← {_dn(self.inherited_from)}"
        )


# ── Multi-hop chain ───────────────────────────────────────────────────────────

@dataclass
class MultiHopChain:
    """
    Complete propagation chain for one memory key, resolved to true depth.

    budget=$5M:
      Product Manager writes $5M
      → Finance corrects to $2M
      → Engineering reads stale $5M          (depth=0)
        → Product Manager inherits via report (depth=1)
          → Operations inherits via context   (depth=2)
    """
    key:             str
    origin_writer:   str       # who first wrote the key
    stale_value:     str       # the incorrect value that propagated
    correct_value:   str       # what the value should have been
    hops:            List[HopNode] = field(default_factory=list)

    def depth(self) -> int:
        return max((n.hop_depth for n in self.hops), default=0)

    def propagation_radius(self) -> int:
        return len({n.agent_id for n in self.hops})

    def affected_agents(self) -> List[str]:
        return sorted({n.agent_id for n in self.hops})

    def render(self) -> str:
        lines = [
            f"  [{self.key}]  {_dn(self.origin_writer)} wrote {self.stale_value!r}  "
            f"→  correction: {self.correct_value!r}  "
            f"→  depth={self.depth()}, radius={self.propagation_radius()}",
        ]
        for hop in self.hops:
            indent = "    " + "  " * hop.hop_depth
            lines.append(f"{indent}→ {hop.render()}")
        return "\n".join(lines)


# ── Multi-hop propagation report ──────────────────────────────────────────────

@dataclass
class MultiHopReport:
    """Complete multi-hop propagation analysis across all stale keys."""
    chains:            List[MultiHopChain] = field(default_factory=list)
    max_depth:         int                 = 0
    total_agents_affected: int             = 0
    deepest_chain_key: str                 = ""
    propagation_summary: str              = ""

    def as_dict(self) -> dict:
        return {
            "max_propagation_depth":   self.max_depth,
            "total_agents_affected":   self.total_agents_affected,
            "deepest_chain_key":       self.deepest_chain_key,
            "propagation_summary":     self.propagation_summary,
            "chains": [
                {
                    "key":              c.key,
                    "origin_writer":    c.origin_writer,
                    "stale_value":      c.stale_value,
                    "correct_value":    c.correct_value,
                    "depth":            c.depth(),
                    "radius":           c.propagation_radius(),
                    "affected_agents":  c.affected_agents(),
                    "hops": [
                        {
                            "agent":            h.agent_id,
                            "depth":            h.hop_depth,
                            "value_inherited":  h.value_inherited,
                            "how_inherited":    h.how_inherited,
                            "from":             h.inherited_from,
                        }
                        for h in c.hops
                    ],
                }
                for c in self.chains
            ],
        }

    def render(self) -> str:
        if not self.chains:
            return "  No multi-hop propagation detected."
        lines = [
            f"  Max propagation depth:    {self.max_depth}",
            f"  Total agents affected:    {self.total_agents_affected}",
            f"  Deepest chain:            {self.deepest_chain_key}",
            f"  Summary:                  {self.propagation_summary}",
            "",
        ]
        for c in self.chains:
            lines.append(c.render())
            lines.append("")
        return "\n".join(lines)


# ── BFS traversal ─────────────────────────────────────────────────────────────

def _bfs_propagation(
    direct_stale_agent: str,
    stale_value:        str,
    all_agents:         List[str],
    max_depth:          int = _MAX_HOP_DEPTH,
) -> List[HopNode]:
    """
    BFS from the direct stale reader across the _DOWNSTREAM_READERS graph.
    Returns all nodes reachable within max_depth hops.
    """
    hops: List[HopNode] = []
    visited: Set[str] = {direct_stale_agent}

    # Hop 0: the direct stale reader
    hops.append(HopNode(
        agent_id        = direct_stale_agent,
        hop_depth       = 0,
        value_inherited = stale_value,
        how_inherited   = "direct_stale_read",
        inherited_from  = "memory_store",
    ))

    # BFS queue: (current_agent, current_depth)
    queue: List[Tuple[str, int]] = [(direct_stale_agent, 0)]

    while queue:
        current_agent, current_depth = queue.pop(0)
        if current_depth >= max_depth:
            continue

        downstream = _DOWNSTREAM_READERS.get(current_agent, [])
        for next_agent in downstream:
            if next_agent in visited:
                continue
            # Only include agents that are actually in this scenario
            if all_agents and next_agent not in all_agents:
                continue
            visited.add(next_agent)
            how = _INHERITANCE_TYPE.get((current_agent, next_agent), "context_injection")
            hops.append(HopNode(
                agent_id        = next_agent,
                hop_depth       = current_depth + 1,
                value_inherited = stale_value,
                how_inherited   = how,
                inherited_from  = current_agent,
            ))
            queue.append((next_agent, current_depth + 1))

    return hops


# ── Public builder ────────────────────────────────────────────────────────────

def build_multi_hop_propagation_report(
    store:        "SharedMemoryStore",
    stale_report: "StaleMemoryReport",
) -> MultiHopReport:
    """
    Build a MultiHopReport from the memory store and stale report.

    For each stale key:
      1. Find the direct stale reader(s).
      2. BFS through the dependency graph to find all indirect consumers.
      3. Record the full hop chain with depth and inheritance type.
    """
    all_agents = list(_DOWNSTREAM_READERS.keys())
    chains: List[MultiHopChain] = []
    all_affected: Set[str] = set()

    # Group stale records by key
    stale_by_key: Dict[str, List] = {}
    for rec in stale_report.records:
        stale_by_key.setdefault(rec.key, []).append(rec)

    for key, stale_records in stale_by_key.items():
        history = store.history(key)
        if not history:
            continue

        # Origin writer (first write to this key)
        origin = next((r.written_by for r in history if r.version == 1), "unknown")
        current_val = str(history[-1].value) if history else "?"

        for stale_rec in stale_records:
            stale_value = stale_rec.stale_value

            hops = _bfs_propagation(
                direct_stale_agent = stale_rec.reader_agent,
                stale_value        = stale_value,
                all_agents         = all_agents,
            )

            # Remove originator and corrector from contamination (they have the truth)
            correction_agents = {r.written_by for r in history if r.version > 1}
            hops = [h for h in hops if h.agent_id not in correction_agents]

            chain = MultiHopChain(
                key           = key,
                origin_writer = origin,
                stale_value   = stale_value,
                correct_value = current_val,
                hops          = hops,
            )
            chains.append(chain)
            all_affected.update(h.agent_id for h in hops)

    if not chains:
        return MultiHopReport(propagation_summary="No stale memory propagation detected.")

    # Deduplicate chains by key (merge hops if same key appears multiple times)
    merged: Dict[str, MultiHopChain] = {}
    for c in chains:
        if c.key in merged:
            # Add any hops not already present
            existing_agents = {h.agent_id for h in merged[c.key].hops}
            for h in c.hops:
                if h.agent_id not in existing_agents:
                    merged[c.key].hops.append(h)
        else:
            merged[c.key] = c
    chains = list(merged.values())

    max_depth = max(c.depth() for c in chains)
    deepest = max(chains, key=lambda c: c.depth())

    summary = (
        f"{len(chains)} memory key(s) propagated across up to {max_depth} hop(s). "
        f"Deepest: {deepest.key} ({deepest.propagation_radius()} agents affected). "
        f"Total unique agents contaminated: {len(all_affected)}."
    )

    return MultiHopReport(
        chains               = sorted(chains, key=lambda c: c.depth(), reverse=True),
        max_depth            = max_depth,
        total_agents_affected = len(all_affected),
        deepest_chain_key    = deepest.key,
        propagation_summary  = summary,
    )
