"""
Memory Influence Graph v2

Replaces the flat "Write / Read / Stale" view with a causal influence graph:

    Memory Value
    → Decision (what the agent decided based on that memory)
    → Agent (who made the decision)
    → Consequence (what went wrong because of that decision)
    → Recovery (how and whether the damage was repaired)

EXAMPLE OUTPUT
--------------
  Budget=$5M (Product Manager)
  → Engineering resource plan: 20 engineers, $1.2M cloud
    → Engineering Lead
      → Over-allocation: $3M above actual $2M constraint
        → Finance Lead corrected budget to $2M
          → Engineering revised to 8 engineers, $600K cloud  [RECOVERED]

The graph captures:
  1. Which memory value was the root of the causal chain
  2. What decision it produced (inferred from domain knowledge + stale narrative)
  3. Which agent held the decision
  4. What concrete consequence resulted
  5. Whether and how recovery occurred

Graph representation:
  Nodes typed as: "memory" | "decision" | "agent" | "consequence" | "recovery"
  Edges typed as: "informed" | "made" | "caused" | "repaired"
  Each edge has a causal_strength (0-1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from harpo.memory.memory_store import SharedMemoryStore
    from harpo.memory.stale_memory_detector import StaleMemoryReport
    from harpo.memory.correction_vs_recovery import CorrectionRecoveryReport

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


# ── Graph primitives ──────────────────────────────────────────────────────────

@dataclass
class InfluenceNode:
    node_id:    str
    node_type:  str    # "memory" | "decision" | "agent" | "consequence" | "recovery"
    label:      str    # human-readable short label
    detail:     str    # longer explanation
    key:        str    # memory key this node belongs to (for grouping)
    severity:   str    = ""   # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | ""


@dataclass
class InfluenceEdge:
    source_id:       str
    target_id:       str
    edge_type:       str    # "informed" | "made" | "caused" | "repaired" | "inherited"
    causal_strength: float  # 0-1


@dataclass
class InfluenceGraph:
    """
    Complete causal influence graph for all memory-related failures in a trajectory.
    """
    nodes:    List[InfluenceNode] = field(default_factory=list)
    edges:    List[InfluenceEdge] = field(default_factory=list)
    chains:   List[str]           = field(default_factory=list)  # rendered causal chains
    summary:  str                 = ""

    def as_dict(self) -> dict:
        return {
            "node_count":  len(self.nodes),
            "edge_count":  len(self.edges),
            "summary":     self.summary,
            "chains":      self.chains,
            "nodes": [
                {
                    "id":       n.node_id,
                    "type":     n.node_type,
                    "label":    n.label,
                    "key":      n.key,
                    "severity": n.severity,
                }
                for n in self.nodes
            ],
            "edges": [
                {
                    "from":     e.source_id,
                    "to":       e.target_id,
                    "type":     e.edge_type,
                    "strength": round(e.causal_strength, 3),
                }
                for e in self.edges
            ],
        }

    def render(self) -> str:
        lines = [
            f"  Nodes: {len(self.nodes)}  Edges: {len(self.edges)}",
            f"  {self.summary}",
            "",
        ]
        for chain in self.chains:
            lines.append(chain)
            lines.append("")
        return "\n".join(lines)


# ── Domain knowledge: decisions, consequences, recoveries ─────────────────────

_MEMORY_DECISIONS: Dict[str, Dict[str, str]] = {
    "budget": {
        "engineering-lead": "Engineering resource plan: 20 engineers, $1.2M cloud infrastructure",
        "operations-lead":  "Operations budget allocated: full $5M logistics scope",
        "product-manager":  "Launch approved under $5M cost model",
    },
    "scope": {
        "marketing-lead":   "Campaign designed for North America only (US-targeted messaging)",
        "operations-lead":  "Logistics scoped for US distribution only",
    },
    "launch_date": {
        "operations-lead":  "Vendor contracts signed for December 2024 delivery",
        "marketing-lead":   "Campaign calendar scheduled for December 2024 launch",
    },
    "regulatory_requirements": {
        "marketing-lead":   "EU campaign designed without GDPR Article 13 disclosures",
        "operations-lead":  "No EU data residency provisions in operations plan",
    },
}

_MEMORY_CONSEQUENCES: Dict[str, Dict[str, str]] = {
    "budget": {
        "engineering-lead": "$3M resource over-allocation vs. actual $2M constraint → plan revision required",
        "operations-lead":  "Operations over-budget by $3M → vendor commitments cannot be honored",
        "product-manager":  "Launch approved on incorrect financial basis → reapproval needed",
    },
    "scope": {
        "marketing-lead":   "EU market uncovered → compliance gap and missed revenue opportunity",
        "operations-lead":  "No EU logistics capacity → launch blocked in mandatory EU territory",
    },
    "launch_date": {
        "operations-lead":  "December vendor contracts conflict with March timeline → cancellation penalties",
        "marketing-lead":   "December campaign calendar must be rebuilt for March → 8 weeks of rework",
    },
    "regulatory_requirements": {
        "marketing-lead":   "EU launch non-compliant → GDPR violation risk, potential €20M fine",
        "operations-lead":  "Data handling non-compliant in EU → regulatory remediation required",
    },
}

_MEMORY_RECOVERIES: Dict[str, str] = {
    "budget":                  "Finance Lead corrected budget to $2M → Engineering revised to 8 engineers",
    "scope":                   "Legal Lead mandated EU scope → Marketing redesigned campaign with EU coverage",
    "launch_date":             "Finance Lead updated date to March 2025 → Operations rescheduled vendors",
    "regulatory_requirements": "Legal Lead issued updated compliance requirements → Marketing revised EU messaging",
}

_KEY_SEVERITY: Dict[str, str] = {
    "budget":                  "HIGH",
    "scope":                   "HIGH",
    "launch_date":             "MEDIUM",
    "regulatory_requirements": "CRITICAL",
}


# ── Chain builder ─────────────────────────────────────────────────────────────

def _build_chain(
    key:              str,
    stale_value:      str,
    correct_value:    str,
    origin_writer:    str,
    stale_agent:      str,
    correction_agent: str,
    was_recovered:    bool,
    severity:         str,
) -> Tuple[List[InfluenceNode], List[InfluenceEdge], str]:
    """Build one complete causal chain for a stale read event."""
    nodes: List[InfluenceNode] = []
    edges: List[InfluenceEdge] = []

    # ── Node 1: Memory value ──────────────────────────────────────────────────
    mem_id = f"mem:{key}:{stale_value[:20]}"
    nodes.append(InfluenceNode(
        node_id   = mem_id,
        node_type = "memory",
        label     = f"{key}={stale_value}",
        detail    = (f"{_dn(origin_writer)} wrote {key}={stale_value!r}; "
                     f"later corrected to {correct_value!r}"),
        key       = key,
        severity  = severity,
    ))

    # ── Node 2: Decision ──────────────────────────────────────────────────────
    decision_text = _MEMORY_DECISIONS.get(key, {}).get(stale_agent, f"{stale_agent} planned based on {key}={stale_value!r}")
    dec_id = f"dec:{key}:{stale_agent}"
    nodes.append(InfluenceNode(
        node_id   = dec_id,
        node_type = "decision",
        label     = decision_text[:60],
        detail    = decision_text,
        key       = key,
    ))
    edges.append(InfluenceEdge(
        source_id       = mem_id,
        target_id       = dec_id,
        edge_type       = "informed",
        causal_strength = 0.90,
    ))

    # ── Node 3: Agent ─────────────────────────────────────────────────────────
    agent_id_node = f"agent:{stale_agent}"
    nodes.append(InfluenceNode(
        node_id   = agent_id_node,
        node_type = "agent",
        label     = _dn(stale_agent),
        detail    = f"{_dn(stale_agent)} held stale {key} value during planning",
        key       = key,
    ))
    edges.append(InfluenceEdge(
        source_id       = dec_id,
        target_id       = agent_id_node,
        edge_type       = "made",
        causal_strength = 0.85,
    ))

    # ── Node 4: Consequence ───────────────────────────────────────────────────
    consequence_text = _MEMORY_CONSEQUENCES.get(key, {}).get(
        stale_agent,
        f"{_dn(stale_agent)} produced plans based on incorrect {key} value",
    )
    cons_id = f"cons:{key}:{stale_agent}"
    nodes.append(InfluenceNode(
        node_id   = cons_id,
        node_type = "consequence",
        label     = consequence_text[:60],
        detail    = consequence_text,
        key       = key,
        severity  = severity,
    ))
    edges.append(InfluenceEdge(
        source_id       = agent_id_node,
        target_id       = cons_id,
        edge_type       = "caused",
        causal_strength = 0.80,
    ))

    # ── Node 5: Recovery (optional) ───────────────────────────────────────────
    if was_recovered:
        recovery_text = _MEMORY_RECOVERIES.get(
            key,
            f"{_dn(correction_agent)} corrected {key} to {correct_value!r} → agent revised plans",
        )
        rec_id = f"rec:{key}:{stale_agent}"
        nodes.append(InfluenceNode(
            node_id   = rec_id,
            node_type = "recovery",
            label     = f"{_dn(correction_agent)} corrected → plan revised",
            detail    = recovery_text,
            key       = key,
        ))
        edges.append(InfluenceEdge(
            source_id       = cons_id,
            target_id       = rec_id,
            edge_type       = "repaired",
            causal_strength = 0.75,
        ))

    # ── Rendered chain text ───────────────────────────────────────────────────
    indent = "    "
    rec_status = "[RECOVERED]" if was_recovered else "[UNRESOLVED]"
    chain_lines = [
        f"  {key}={stale_value!r}  ({_dn(origin_writer)})  severity={severity}",
        f"{indent}→ Decision: {decision_text[:70]}",
        f"{indent}  → {_dn(stale_agent)}",
        f"{indent}    → Consequence: {consequence_text[:70]}",
    ]
    if was_recovered:
        rec_text = _MEMORY_RECOVERIES.get(key, "Recovery occurred.")
        chain_lines.append(f"{indent}      → {rec_text[:80]}  {rec_status}")
    else:
        chain_lines.append(f"{indent}      → {rec_status} — consequence persisted to trajectory end")

    return nodes, edges, "\n".join(chain_lines)


# ── Public builder ────────────────────────────────────────────────────────────

def build_influence_graph(
    store:        "SharedMemoryStore",
    stale_report: "StaleMemoryReport",
    cr_report:    Optional["CorrectionRecoveryReport"] = None,
) -> InfluenceGraph:
    """
    Build a complete causal influence graph from memory store and stale report.

    For each stale read:
      Memory value → Decision → Agent → Consequence → Recovery
    """
    all_nodes: List[InfluenceNode] = []
    all_edges: List[InfluenceEdge] = []
    all_chains: List[str]          = []

    # Build recovery lookup: which keys+agents had confirmed recovery
    recovered_set: Set[Tuple[str, str]] = set()
    correction_agent_map: Dict[str, str] = {}
    if cr_report:
        for rec in cr_report.recoveries:
            recovered_set.add((rec.key, rec.recovering_agent))
        for c in cr_report.corrections:
            correction_agent_map[c.key] = c.correction_agent
    else:
        # Fallback: use was_corrected flags from stale_report
        for rec in stale_report.records:
            if rec.was_corrected:
                recovered_set.add((rec.key, rec.reader_agent))
                correction_agent_map[rec.key] = rec.correction_agent

    for rec in stale_report.records:
        key = rec.key
        history = store.history(key)
        origin_writer = (
            next((r.written_by for r in history if r.version == 1), "unknown")
        )
        correction_agent = correction_agent_map.get(key, "unknown")
        was_recovered    = (key, rec.reader_agent) in recovered_set
        severity         = _KEY_SEVERITY.get(key, "MEDIUM")

        nodes, edges, chain_text = _build_chain(
            key              = key,
            stale_value      = rec.stale_value,
            correct_value    = rec.current_value,
            origin_writer    = origin_writer,
            stale_agent      = rec.reader_agent,
            correction_agent = correction_agent,
            was_recovered    = was_recovered,
            severity         = severity,
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)
        all_chains.append(chain_text)

    # Deduplicate agent nodes (same agent appears in multiple chains)
    seen_ids: Set[str] = set()
    dedup_nodes: List[InfluenceNode] = []
    for n in all_nodes:
        if n.node_id not in seen_ids:
            dedup_nodes.append(n)
            seen_ids.add(n.node_id)

    n_recovered   = sum(1 for (k, a) in recovered_set)
    n_unresolved  = stale_report.total_stale - stale_report.corrected_count
    summary = (
        f"{len(all_chains)} causal chain(s) traced. "
        f"{n_recovered} chain(s) terminated in recovery. "
        f"{n_unresolved} chain(s) unresolved at trajectory end."
    )

    return InfluenceGraph(
        nodes   = dedup_nodes,
        edges   = all_edges,
        chains  = all_chains,
        summary = summary,
    )
