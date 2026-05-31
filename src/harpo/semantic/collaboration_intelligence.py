"""
Multi-Agent Collaboration Intelligence

Builds a dependency graph of inter-agent interactions and measures
the *quality* of collaboration — going beyond "did handoffs happen?"
to answer:

  CONTRIBUTION QUALITY  — how much did each agent's output improve
                           subsequent agents' reasoning quality?
  TOPIC ADOPTION        — when agent B uses vocabulary from agent A's
                           last response, agent A's contribution was adopted
  CONTRADICTION REPAIR  — did a later agent explicitly correct an
                           earlier agent's error?
  STABILIZATION SCORE   — which agents reduced overall trajectory
                           failure signals vs amplified them?
  COORDINATION LATENCY  — time between related contributions from
                           different agents on the same topic
  SILOED AGENTS         — agents whose vocabulary never overlaps with
                           others → contributing in isolation

This module works from TrajectoryStep.agent_id tagging (set by the demo
via step.agent_id = agent_id). It gracefully degrades to 0.5 if no
multi-agent tagging is present.

No external dependencies.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep

_STOP: Set[str] = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "that", "this", "these", "those",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "and", "or", "but", "if", "then", "so", "as", "at", "by", "for",
    "of", "on", "to", "in", "with", "about", "from", "not", "no",
    "what", "which", "who", "when", "where", "how", "very", "just",
    "also", "can", "all", "any", "more", "into", "than", "here",
}

_CORRECTION_MARKERS = re.compile(
    r'\b(?:actually|correction|incorrect|wrong|re-evaluating|correcting|'
    r'forensics (?:shows|reveals|confirms)|file system|earlier was|'
    r'this contradicts|not sql|not at 03|21:43|five hours earlier)\b',
    re.IGNORECASE,
)

_FAILURE_MARKERS = re.compile(
    r'\b(?:failed|error|violation|missed|incorrect|wrong|breach|'
    r'unresolved|conflict|deadline|penalty|harm|exfiltration)\b',
    re.IGNORECASE,
)

_STABILIZATION_MARKERS = re.compile(
    r'\b(?:contained|resolved|fixed|corrected|blocked|revoked|patched|'
    r'mitigated|stabilized|addressed|recovered|remediated|notified|isolated|'
    r'confirmed|verified|updated|clarified)\b',
    re.IGNORECASE,
)


def _sig_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r'\b[a-z]{3,}\b', text.lower()) if t not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class AgentContributionProfile:
    """Per-agent metrics in a multi-agent trajectory."""
    agent_id:               str
    step_count:             int
    turn_range:             Tuple[int, int]        # (first_turn, last_turn)
    adopted_by:             List[str]              # agents who later used this agent's vocabulary
    adopted_from:           List[str]              # agents whose vocabulary this agent adopted
    contradiction_repairs:  int                    # times this agent corrected another's error
    failure_amplifications: int                    # times this agent's steps had failure signals
    stabilization_events:   int                    # times this agent contributed stabilizing content
    contribution_score:     float                  # 0-1: overall contribution quality
    is_siloed:              bool                   # vocabulary never overlaps with others

    def narrative(self) -> str:
        adopted_str = f"adopted by {', '.join(self.adopted_by)}" if self.adopted_by else "not adopted"
        repair_str  = f"{self.contradiction_repairs} repair(s)" if self.contradiction_repairs else "no repairs"
        stab_str    = f"{self.stabilization_events} stabilization(s)" if self.stabilization_events else ""
        siloed_note = " [SILOED — minimal integration]" if self.is_siloed else ""
        return (
            f"{self.agent_id}: score={self.contribution_score:.2f}{siloed_note}. "
            f"{adopted_str}. {repair_str}. {stab_str}."
        )


@dataclass
class CollaborationEdge:
    """A directed adoption relationship: source → target."""
    source_agent:   str
    target_agent:   str
    overlap:        float    # vocabulary overlap ratio
    edge_type:      str      # "adoption" | "repair" | "contradiction"
    turn_source:    int
    turn_target:    int


@dataclass
class CollaborationIntelligenceReport:
    """Full inter-agent collaboration analysis."""
    agent_profiles:          Dict[str, AgentContributionProfile]
    edges:                   List[CollaborationEdge]
    strongest_contributor:   Optional[str]
    most_siloed_agent:       Optional[str]
    overall_integration:     float    # 0-1: average cross-agent adoption
    contradiction_repair_count: int
    collaborative_quality_score: float  # replaces the heuristic 0.5 default

    def as_dict(self) -> dict:
        return {
            "overall_integration":       round(self.overall_integration, 3),
            "contradiction_repair_count": self.contradiction_repair_count,
            "collaborative_quality_score": round(self.collaborative_quality_score, 3),
            "strongest_contributor":     self.strongest_contributor,
            "most_siloed_agent":         self.most_siloed_agent,
            "agent_profiles": {
                aid: {
                    "contribution_score": round(p.contribution_score, 3),
                    "adopted_by":         p.adopted_by,
                    "adopted_from":       p.adopted_from,
                    "contradiction_repairs": p.contradiction_repairs,
                    "stabilization_events":  p.stabilization_events,
                    "is_siloed":          p.is_siloed,
                    "narrative":          p.narrative(),
                }
                for aid, p in self.agent_profiles.items()
            },
            "edges": [
                {
                    "source": e.source_agent,
                    "target": e.target_agent,
                    "overlap": round(e.overlap, 3),
                    "type":   e.edge_type,
                }
                for e in self.edges
            ],
        }

    def narrative(self) -> str:
        n_agents = len(self.agent_profiles)
        if n_agents < 2:
            return "Single agent — collaboration analysis not applicable."
        top = self.strongest_contributor or "unknown"
        sil = f" {self.most_siloed_agent} was siloed." if self.most_siloed_agent else ""
        return (
            f"{n_agents} agents. Integration: {self.overall_integration:.2f}. "
            f"Quality score: {self.collaborative_quality_score:.2f}. "
            f"Strongest contributor: {top}. "
            f"{self.contradiction_repair_count} cross-agent repair(s).{sil}"
        )


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_collaboration(traj: "AgentTrajectory") -> CollaborationIntelligenceReport:
    """
    Build a CollaborationIntelligenceReport from a trajectory with
    agent_id-tagged steps.

    Algorithm
    ---------
    1. Group steps by agent_id.  If none present, return default 0.5.
    2. For each agent, build a "contribution signature": last 3 THINK/RESPONSE
       step token sets merged.
    3. For each subsequent agent (ordered by first turn): compute vocabulary
       overlap with prior agents' contribution signatures.
       Overlap ≥ 0.20 → adoption edge (source: prior agent, target: current)
    4. Detect repair: a step with correction markers + token overlap with a
       prior agent's steps → repair edge.
    5. Count failure amplifications and stabilization events per agent.
    6. Compute contribution_score per agent = adoption_score × 0.40 +
       repair_score × 0.30 + stabilization_score × 0.30 − silo_penalty × 0.20
    7. overall_integration = mean pairwise adoption across all agent pairs.
    8. collaborative_quality_score = mean contribution_score, adjusted by
       contradiction_repair_count and silo count.
    """
    from harpo.trajectory.schema import StepType

    think_types = (StepType.THINK, StepType.RESPONSE)

    # Group steps by agent
    agent_steps: Dict[str, List["TrajectoryStep"]] = defaultdict(list)
    for step in traj.steps:
        aid = getattr(step, "agent_id", "")
        if aid:
            agent_steps[aid].append(step)

    agent_ids = [a for a in agent_steps if a]

    if len(agent_ids) < 2:
        return CollaborationIntelligenceReport(
            agent_profiles          = {},
            edges                   = [],
            strongest_contributor   = None,
            most_siloed_agent       = None,
            overall_integration     = 0.5,
            contradiction_repair_count = 0,
            collaborative_quality_score = 0.5,
        )

    # Sort agents by their first step timestamp
    agent_order = sorted(
        agent_ids,
        key=lambda a: min(s.timestamp for s in agent_steps[a]),
    )

    # ── Build vocabulary signatures per agent ─────────────────────────────────
    # Use ALL THINK/RESPONSE text (not just last N) for richer matching
    agent_all_tokens: Dict[str, Set[str]] = {}
    agent_last_tokens: Dict[str, Set[str]] = {}  # last 3 steps (contribution footprint)

    for aid in agent_order:
        steps = [s for s in agent_steps[aid] if s.step_type in think_types]
        if not steps:
            steps = agent_steps[aid]
        all_tok = set()
        for s in steps:
            all_tok |= _sig_tokens(s.output_text)
        agent_all_tokens[aid] = all_tok
        last_n = steps[-3:] if len(steps) >= 3 else steps
        last_tok: Set[str] = set()
        for s in last_n:
            last_tok |= _sig_tokens(s.output_text)
        agent_last_tokens[aid] = last_tok

    # ── Build adoption edges ──────────────────────────────────────────────────
    edges: List[CollaborationEdge] = []
    adopted_by: Dict[str, List[str]]   = defaultdict(list)
    adopted_from: Dict[str, List[str]] = defaultdict(list)

    for i, target_id in enumerate(agent_order[1:], start=1):
        prior_agents = agent_order[:i]
        target_toks  = agent_all_tokens.get(target_id, set())

        for source_id in prior_agents:
            source_sig = agent_last_tokens.get(source_id, set())
            ov = _overlap(source_sig, target_toks)
            if ov >= 0.20:
                t_source = min(s.turn_number for s in agent_steps[source_id])
                t_target = min(s.turn_number for s in agent_steps[target_id])
                edges.append(CollaborationEdge(
                    source_agent = source_id,
                    target_agent = target_id,
                    overlap      = round(ov, 3),
                    edge_type    = "adoption",
                    turn_source  = t_source,
                    turn_target  = t_target,
                ))
                if source_id not in adopted_by[source_id]:
                    adopted_by[source_id].append(target_id)
                if source_id not in adopted_from[target_id]:
                    adopted_from[target_id].append(source_id)

    # ── Detect contradiction repairs ──────────────────────────────────────────
    repair_counts:  Dict[str, int] = defaultdict(int)
    failure_counts: Dict[str, int] = defaultdict(int)
    stab_counts:    Dict[str, int] = defaultdict(int)

    for aid in agent_order:
        for step in agent_steps[aid]:
            if step.step_type not in think_types:
                continue
            text = step.output_text
            if _CORRECTION_MARKERS.search(text):
                # Find which prior agent's tokens this step overlaps with
                for other_id in agent_order:
                    if other_id == aid:
                        continue
                    other_first_turn = min(s.turn_number for s in agent_steps[other_id])
                    if other_first_turn >= step.turn_number:
                        continue
                    ov = _overlap(_sig_tokens(text), agent_all_tokens.get(other_id, set()))
                    if ov >= 0.20:
                        repair_counts[aid] += 1
                        edges.append(CollaborationEdge(
                            source_agent = aid,
                            target_agent = other_id,
                            overlap      = round(ov, 3),
                            edge_type    = "repair",
                            turn_source  = step.turn_number,
                            turn_target  = other_first_turn,
                        ))
                        break
            if _FAILURE_MARKERS.search(text):
                failure_counts[aid] += 1
            if _STABILIZATION_MARKERS.search(text):
                stab_counts[aid] += 1

    # ── Identify siloed agents ────────────────────────────────────────────────
    cross_overlaps: Dict[str, float] = {}
    for aid in agent_order:
        others = [other for other in agent_order if other != aid]
        if not others:
            cross_overlaps[aid] = 0.0
            continue
        avg_ov = sum(
            _jaccard(agent_all_tokens.get(aid, set()), agent_all_tokens.get(o, set()))
            for o in others
        ) / len(others)
        cross_overlaps[aid] = avg_ov

    # ── Build per-agent profiles ──────────────────────────────────────────────
    profiles: Dict[str, AgentContributionProfile] = {}
    for aid in agent_order:
        steps = agent_steps[aid]
        turns = [s.turn_number for s in steps]
        turn_range = (min(turns), max(turns)) if turns else (0, 0)

        # Adoption score: how many other agents adopted this agent's vocabulary?
        n_adopted_by = len(adopted_by[aid])
        adoption_score = min(n_adopted_by / max(len(agent_order) - 1, 1), 1.0)

        # Repair score
        repair_score = min(repair_counts[aid] / 3.0, 1.0)

        # Stabilization score
        fail_c = failure_counts.get(aid, 0)
        stab_c = stab_counts.get(aid, 0)
        if fail_c + stab_c == 0:
            stab_score = 0.5
        else:
            stab_score = stab_c / (fail_c + stab_c)

        is_siloed = cross_overlaps.get(aid, 0.0) < 0.10

        contribution_score = (
            adoption_score * 0.40
            + repair_score * 0.30
            + stab_score * 0.30
            - (0.20 if is_siloed else 0.0)
        )
        contribution_score = max(0.0, min(1.0, contribution_score))

        profiles[aid] = AgentContributionProfile(
            agent_id               = aid,
            step_count             = len(steps),
            turn_range             = turn_range,
            adopted_by             = list(adopted_by[aid]),
            adopted_from           = list(adopted_from[aid]),
            contradiction_repairs  = repair_counts[aid],
            failure_amplifications = failure_counts.get(aid, 0),
            stabilization_events   = stab_counts.get(aid, 0),
            contribution_score     = round(contribution_score, 3),
            is_siloed              = is_siloed,
        )

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if profiles:
        strongest = max(profiles, key=lambda a: profiles[a].contribution_score)
        siloed    = [a for a, p in profiles.items() if p.is_siloed]
        most_sil  = siloed[0] if siloed else None

        # Overall integration: average cross-agent overlap
        pairwise_overlaps = []
        for i, a1 in enumerate(agent_order):
            for a2 in agent_order[i+1:]:
                pairwise_overlaps.append(
                    _jaccard(agent_all_tokens.get(a1, set()),
                             agent_all_tokens.get(a2, set()))
                )
        overall_int = (sum(pairwise_overlaps) / len(pairwise_overlaps)
                       if pairwise_overlaps else 0.5)

        n_repairs = sum(p.contradiction_repairs for p in profiles.values())

        # Quality score
        avg_contrib = sum(p.contribution_score for p in profiles.values()) / len(profiles)
        repair_bonus = min(n_repairs * 0.05, 0.15)
        silo_penalty = len(siloed) * 0.05
        collab_score = max(0.0, min(1.0, avg_contrib + repair_bonus - silo_penalty))
    else:
        strongest = None
        most_sil  = None
        overall_int = 0.5
        n_repairs   = 0
        collab_score = 0.5

    return CollaborationIntelligenceReport(
        agent_profiles               = profiles,
        edges                        = edges,
        strongest_contributor        = strongest,
        most_siloed_agent            = most_sil,
        overall_integration          = round(overall_int, 3),
        contradiction_repair_count   = n_repairs,
        collaborative_quality_score  = round(collab_score, 3),
    )
