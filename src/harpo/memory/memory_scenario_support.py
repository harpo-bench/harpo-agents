"""
Memory Scenario Support — Memory Lineage Graph

Builds a directed lineage graph of memory operations across a multi-agent
trajectory.  Tracks:

  Source (writer)  → Content   → Consumer (reader)  → Outcome

Classifies each edge as:
  BENEFICIAL  — reader used the content to make a correct decision
  HARMFUL     — reader used stale/incorrect content, reinforcing a bad assumption
  NEUTRAL     — content read but no detectable downstream effect
  STALE       — source was later contradicted; reader used it before correction

This makes memory a first-class causal signal rather than an absent one.

Memory Lineage Graph format
---------------------------
  Nodes: agents (as publishers and consumers)
  Edges: InferredMemoryEvent (from memory_instrumentation.py) enriched with
         outcome classification

Example output
--------------
  Memory Write:  Security Analyst → [breach timeline: 03:12 UTC]
  Memory Read:   Compliance Agent ← [breach timeline: 03:12 UTC]
  Result:        Incorrect GDPR deadline (stale timeline consumed)
  Impact:        HARMFUL (stale reuse)

  Memory Write:  Forensics Agent → [breach timeline: 21:43 UTC correction]
  Memory Read:   Incident Commander ← [corrected timeline]
  Result:        Correct GDPR reasoning restored
  Impact:        BENEFICIAL (corrective retrieval)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory
    from harpo.memory.memory_instrumentation import InferredMemoryEvent

_AGENT_DISPLAY: Dict[str, str] = {
    "security-analyst":   "Security Analyst",
    "infra-engineer":     "Infrastructure Engineer",
    "forensics-agent":    "Forensics Agent",
    "compliance-agent":   "Compliance Agent",
    "comms-officer":      "Communications Officer",
    "incident-commander": "Incident Commander",
}

def _dn(a: str) -> str:
    return _AGENT_DISPLAY.get(a or "", (a or "").replace("-", " ").title())


@dataclass
class MemoryEdge:
    """One directed memory operation: writer → content → reader."""
    source_agent:   str           # who wrote the memory
    consumer_agent: str           # who read it
    turn_written:   int
    turn_read:      int
    content_hint:   str           # brief description of what was read
    impact:         str           # "BENEFICIAL" | "HARMFUL" | "STALE" | "NEUTRAL"
    outcome:        str           # one sentence

    @property
    def is_harmful(self) -> bool:
        return self.impact in ("HARMFUL", "STALE")

    @property
    def is_beneficial(self) -> bool:
        return self.impact == "BENEFICIAL"

    def render(self) -> str:
        arrow = "→" if not self.is_harmful else "⚠"
        return (
            f"  [{self.impact}]  {_dn(self.source_agent)} {arrow} {_dn(self.consumer_agent)}\n"
            f"     Content: {self.content_hint}\n"
            f"     Outcome: {self.outcome}"
        )


@dataclass
class MemoryLineageGraph:
    """Complete memory lineage for a trajectory."""
    edges:               List[MemoryEdge] = field(default_factory=list)
    beneficial_count:    int = 0
    harmful_count:       int = 0
    stale_count:         int = 0
    neutral_count:       int = 0
    net_memory_impact:   str = "NEUTRAL"   # "BENEFICIAL" | "HARMFUL" | "NEUTRAL"

    def harmful_edges(self) -> List[MemoryEdge]:
        return [e for e in self.edges if e.is_harmful]

    def beneficial_edges(self) -> List[MemoryEdge]:
        return [e for e in self.edges if e.is_beneficial]

    def render(self) -> str:
        if not self.edges:
            return "  No memory operations detected."
        lines = [f"  Memory Lineage [{self.net_memory_impact}]:",
                 f"  Beneficial: {self.beneficial_count}  "
                 f"Harmful/Stale: {self.harmful_count + self.stale_count}  "
                 f"Neutral: {self.neutral_count}", ""]
        for edge in self.edges:
            lines.append(edge.render())
            lines.append("")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "net_impact":     self.net_memory_impact,
            "beneficial":     self.beneficial_count,
            "harmful":        self.harmful_count,
            "stale":          self.stale_count,
            "neutral":        self.neutral_count,
            "edges": [
                {
                    "source":   e.source_agent,
                    "consumer": e.consumer_agent,
                    "impact":   e.impact,
                    "outcome":  e.outcome,
                }
                for e in self.edges
            ],
        }


def _classify_content(source_agent: str, consumer_agent: str,
                       is_stale: bool, causal_hint: str,
                       overlap: float) -> tuple:
    """Return (content_hint, impact, outcome)."""
    # Domain-aware content classification
    _AGENT_DOMAIN_CONTENT: Dict[str, str] = {
        "security-analyst":   "breach timeline and attack vector assessment",
        "infra-engineer":     "infrastructure scope and network analysis",
        "forensics-agent":    "forensic timeline and host compromise evidence",
        "compliance-agent":   "regulatory notification requirements",
        "comms-officer":      "stakeholder communication strategy",
        "incident-commander": "incident synthesis and response coordination",
    }

    content_hint = _AGENT_DOMAIN_CONTENT.get(source_agent,
                                              f"{_dn(source_agent)}'s analysis")

    if causal_hint == "correction":
        impact  = "BENEFICIAL"
        outcome = (f"{_dn(consumer_agent)} received corrective information from "
                   f"{_dn(source_agent)}, potentially improving downstream reasoning.")
    elif is_stale:
        impact  = "STALE"
        outcome = (f"{_dn(consumer_agent)} read {_dn(source_agent)}'s report "
                   f"which contained claims later contradicted. "
                   f"Stale content may have reinforced incorrect assumptions.")
    elif causal_hint == "reinforcement":
        impact  = "HARMFUL"
        outcome = (f"{_dn(consumer_agent)}'s reading of {_dn(source_agent)}'s report "
                   f"reinforced an active unverified assumption (overlap={overlap:.2f}).")
    else:
        impact  = "NEUTRAL"
        outcome = (f"{_dn(consumer_agent)} integrated {_dn(source_agent)}'s report "
                   f"into its context (overlap={overlap:.2f}, no active assumption match).")

    return content_hint, impact, outcome


def build_memory_lineage(
    traj: "AgentTrajectory",
    analysis: Any = None,
) -> MemoryLineageGraph:
    """
    Build a MemoryLineageGraph from a trajectory.

    Uses memory_instrumentation.infer_memory_operations() to get inferred
    memory events, then enriches them with root-cause context when available.
    """
    try:
        from harpo.memory.memory_instrumentation import infer_memory_operations
        inferred_events = infer_memory_operations(traj)
    except Exception:
        return MemoryLineageGraph()

    if not inferred_events:
        return MemoryLineageGraph()

    # Filter: only cross-agent reads (not self-reads or writes alone)
    cross_agent_reads = [
        e for e in inferred_events
        if e.operation == "read" and e.source_agent != e.reader_agent
    ]

    # Get root cause context if available
    rc_report = None
    if analysis:
        try:
            from harpo.forensics.root_cause_engine import build_root_causes
            rc_report = build_root_causes(analysis, traj)
        except Exception:
            pass

    # For each root cause, find which agents were affected → stale reads
    stale_source_agents: Set[str] = set()
    if rc_report:
        for rc in rc_report.root_causes:
            if rc.resolution == "UNRESOLVED" or rc.damage_score >= 0.3:
                stale_source_agents.add(rc.origin_agent)

    edges: List[MemoryEdge] = []

    for ev in cross_agent_reads:
        # Override is_stale if we know the source agent introduced a significant error
        is_stale = ev.is_stale or (ev.source_agent in stale_source_agents)
        # If the read includes correction signals, override to beneficial
        if ev.causal_hint == "correction":
            is_stale = False

        content_hint, impact, outcome = _classify_content(
            ev.source_agent, ev.reader_agent, is_stale, ev.causal_hint, ev.overlap_ratio
        )

        edges.append(MemoryEdge(
            source_agent   = ev.source_agent,
            consumer_agent = ev.reader_agent,
            turn_written   = 0,   # not available from inferred events
            turn_read      = ev.reader_turn,
            content_hint   = content_hint,
            impact         = impact,
            outcome        = outcome,
        ))

    bc = sum(1 for e in edges if e.impact == "BENEFICIAL")
    hc = sum(1 for e in edges if e.impact == "HARMFUL")
    sc = sum(1 for e in edges if e.impact == "STALE")
    nc = sum(1 for e in edges if e.impact == "NEUTRAL")

    if hc + sc > bc + 1:
        net = "HARMFUL"
    elif bc > hc + sc:
        net = "BENEFICIAL"
    else:
        net = "NEUTRAL"

    return MemoryLineageGraph(
        edges             = edges,
        beneficial_count  = bc,
        harmful_count     = hc,
        stale_count       = sc,
        neutral_count     = nc,
        net_memory_impact = net,
    )
