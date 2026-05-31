"""
Memory Lineage Graph

Builds a directed graph of memory operations: write → update → read.
Every edge represents causal influence: writing an outdated value
that was then read by a downstream agent created the stale-memory failure.

Format
------
For each key, the lineage shows the full write/update/read history:

  budget:
    PM:write($5M, v1) → ENG:read($5M, v1, CURRENT)
                      → FIN:update($2M, v2)
                           → ENG:read($5M, v1, STALE ← uses old version)
                           → MKT:read($2M, v2, CURRENT)

Output structures
-----------------
  MemoryLineageNode — one operation (write/update/read) on one key
  MemoryLineageEdge — causal link between two operations
  MemoryLineageReport — complete graph + narrative for each key
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore, MemoryRecord, MemoryReadEvent

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


@dataclass
class MemoryLineageNode:
    key:         str
    agent_id:    str
    operation:   str    # "write" | "update" | "read" | "invalidate"
    value:       str
    version:     int
    is_stale:    bool   = False
    timestamp:   float  = 0.0

    def render(self) -> str:
        stale = " [STALE]" if self.is_stale else ""
        return f"{_dn(self.agent_id)}:{self.operation}({self.value!r}, v{self.version}){stale}"


@dataclass
class MemoryKeyLineage:
    """Complete lineage for one memory key."""
    key:      str
    nodes:    List[MemoryLineageNode] = field(default_factory=list)

    @property
    def writes(self) -> List[MemoryLineageNode]:
        return [n for n in self.nodes if n.operation in ("write", "update")]

    @property
    def reads(self) -> List[MemoryLineageNode]:
        return [n for n in self.nodes if n.operation == "read"]

    @property
    def stale_reads(self) -> List[MemoryLineageNode]:
        return [n for n in self.nodes if n.is_stale]

    def render(self) -> str:
        lines = [f"  {self.key}:"]
        for node in self.nodes:
            sym = {"write": "✏", "update": "✎", "read": "📖", "invalidate": "✗"}.get(
                node.operation, "·"
            )
            stale = " ← STALE (outdated version used)" if node.is_stale else ""
            lines.append(
                f"    {sym}  {_dn(node.agent_id):20s}  {node.operation:10s}"
                f"  v{node.version}  {str(node.value)[:35]}{stale}"
            )
        return "\n".join(lines)


@dataclass
class MemoryLineageReport:
    """Complete memory lineage across all keys."""
    key_lineages: Dict[str, MemoryKeyLineage] = field(default_factory=dict)
    stale_read_count: int = 0
    total_reads:      int = 0
    total_writes:     int = 0

    def as_dict(self) -> dict:
        return {
            "total_writes":    self.total_writes,
            "total_reads":     self.total_reads,
            "stale_reads":     self.stale_read_count,
            "keys": {
                key: {
                    "writes":     len(lin.writes),
                    "reads":      len(lin.reads),
                    "stale_reads": len(lin.stale_reads),
                    "nodes":      [
                        {
                            "agent":     n.agent_id,
                            "operation": n.operation,
                            "value":     str(n.value)[:60],
                            "version":   n.version,
                            "is_stale":  n.is_stale,
                        }
                        for n in lin.nodes
                    ],
                }
                for key, lin in self.key_lineages.items()
            },
        }

    def render(self) -> str:
        lines = [
            "  MEMORY LINEAGE GRAPH",
            f"  Total writes: {self.total_writes}  "
            f"reads: {self.total_reads}  "
            f"stale: {self.stale_read_count}",
            "",
        ]
        for key, lin in self.key_lineages.items():
            lines.append(lin.render())
            lines.append("")
        return "\n".join(lines)


def build_memory_lineage_report(store: "SharedMemoryStore") -> MemoryLineageReport:
    """Build a MemoryLineageReport from a SharedMemoryStore."""
    key_lineages: Dict[str, MemoryKeyLineage] = {}

    # Build write nodes per key (sorted by version)
    for key in store.all_keys():
        lin = MemoryKeyLineage(key=key)
        for rec in store.history(key):
            lin.nodes.append(MemoryLineageNode(
                key       = key,
                agent_id  = rec.written_by,
                operation = rec.operation,
                value     = str(rec.value)[:40],
                version   = rec.version,
                is_stale  = False,
                timestamp = rec.timestamp,
            ))
        key_lineages[key] = lin

    # Interleave read events in chronological order
    for read_ev in store.all_reads():
        lin = key_lineages.get(read_ev.key)
        if lin is None:
            continue
        # Insert at chronological position
        node = MemoryLineageNode(
            key       = read_ev.key,
            agent_id  = read_ev.reader_agent,
            operation = "read",
            value     = str(read_ev.value_read)[:40],
            version   = read_ev.version_read,
            is_stale  = read_ev.is_stale,
            timestamp = read_ev.timestamp,
        )
        # Find insertion point based on timestamp
        inserted = False
        for i, existing in enumerate(lin.nodes):
            if existing.timestamp > read_ev.timestamp:
                lin.nodes.insert(i, node)
                inserted = True
                break
        if not inserted:
            lin.nodes.append(node)

    stale = sum(
        len(lin.stale_reads) for lin in key_lineages.values()
    )
    return MemoryLineageReport(
        key_lineages     = key_lineages,
        stale_read_count = stale,
        total_reads      = len(store.all_reads()),
        total_writes     = len(store.all_writes()),
    )
