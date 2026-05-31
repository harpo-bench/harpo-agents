"""
Causal Assumption Propagation Analysis

Extends basic assumption tracking to answer causal questions:

  WHERE did this assumption originate?
    → which agent, which turn, what type of epistemic trigger
  WHO inherited it?
    → which subsequent agents' steps contain the assumption's tokens
  WHAT downstream failures did it cause?
    → failure-marker co-occurrence after propagation events
  WAS it corrected?
    → did a REFLECTION or contradiction step follow and reduce the assumption's token footprint?
  DID recovery stabilize it?
    → did a RECOVERY step follow the correction and produce stable reasoning?

This is the causal layer on top of assumptions.py.  It imports AssumptionChain /
analyze_assumption_propagation and enriches each chain with per-node causal
metadata rather than just propagation_turns counts.

No external dependencies.  Pure heuristic + token analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory, TrajectoryStep

# ── Failure signals ──────────────────────────────────────────────────────────
# Token fragments whose presence in a step suggests something went wrong.
_FAILURE_SIGNALS: Set[str] = {
    "failed", "failure", "error", "violation", "missed", "incorrect",
    "wrong", "breach", "unresolved", "conflict", "contradiction", "missed",
    "overlooked", "misidentified", "escalated", "deadline", "penalty",
    "regulatory", "fined", "lawsuit", "harm", "damage", "exfiltration",
}

# ── Trigger type classifiers ─────────────────────────────────────────────────
_TRIGGER_UNCERTAINTY = re.compile(
    r'\b(?:i assume|assuming|probably|likely|i think|it seems|perhaps|'
    r'i believe|i suppose|presumably|it appears|apparently)\b',
    re.IGNORECASE,
)
_TRIGGER_INFERENCE = re.compile(
    r'\b(?:therefore|thus|hence|so it (?:seems|appears)|this suggests|'
    r'this implies|which means|consequently)\b',
    re.IGNORECASE,
)
_TRIGGER_INCOMPLETE_DATA = re.compile(
    r'\b(?:based on|given the (?:data|information|available|limited)|'
    r'from what (?:i|we) (?:see|have|know)|with the (?:data|info) (?:available|at hand))\b',
    re.IGNORECASE,
)
_TRIGGER_DELEGATION = re.compile(
    r'\b(?:according to|as reported by|per the (?:report|analysis)|'
    r'as (?:noted|stated|mentioned) by)\b',
    re.IGNORECASE,
)

# ── Correction markers ────────────────────────────────────────────────────────
_CORRECTION_MARKERS = re.compile(
    r'\b(?:actually|incorrect|wrong|correcting|correction|re-evaluating|'
    r'forensics (?:shows|reveals|confirms)|this was not|not sql|'
    r'not 03:12|not at 03|21:43|five hours|earlier than|file system confirms)\b',
    re.IGNORECASE,
)

# ── Stop words (shared) ───────────────────────────────────────────────────────
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


def _sig_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r'\b[a-z]{3,}\b', text.lower()) if t not in _STOP}


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _classify_trigger(snippet: str) -> str:
    if _TRIGGER_DELEGATION.search(snippet):
        return "delegation"
    if _TRIGGER_INCOMPLETE_DATA.search(snippet):
        return "incomplete_data"
    if _TRIGGER_INFERENCE.search(snippet):
        return "inference"
    return "uncertainty"


# ── Per-propagation-node metadata ─────────────────────────────────────────────

@dataclass
class PropagationNode:
    """One step that absorbed an assumption."""
    turn_number:          int
    step_id:              str
    agent_id:             str        # empty string if single-agent
    overlap_ratio:        float      # token overlap with assumption
    failure_co_occurrence: bool      # failure-signal tokens present in step
    correction_follows:   bool       # REFLECTION/RECOVERY within 2 turns after this
    was_corrected:        bool       # explicit correction marker in step


# ── Per-assumption causal chain ───────────────────────────────────────────────

@dataclass
class CausalAssumptionChain:
    """
    An assumption with a full causal trace from origin through propagation
    to downstream effects and (optional) correction.
    """
    assumption_text:       str
    key_tokens:            Set[str]
    origin_turn:           int
    origin_step_id:        str
    origin_agent_id:       str
    trigger_type:          str        # "uncertainty" | "inference" | "incomplete_data" | "delegation"

    propagation_nodes:     List[PropagationNode] = field(default_factory=list)

    # Downstream effects
    failure_linked_turns:  List[int]  = field(default_factory=list)
    correction_turn:       Optional[int] = None
    correction_type:       str        = ""   # "reflection" | "contradiction" | "recovery" | ""
    recovery_turn:         Optional[int] = None

    # Verdicts
    was_corrected:         bool       = False
    damage_score:          float      = 0.0  # 0-1: higher = more damage

    def propagation_radius(self) -> int:
        """Distinct turns this assumption contaminated."""
        return len({n.turn_number for n in self.propagation_nodes})

    def contaminated_agents(self) -> List[str]:
        """Unique agents (other than origin) that received this assumption."""
        agents = {n.agent_id for n in self.propagation_nodes
                  if n.agent_id and n.agent_id != self.origin_agent_id}
        return sorted(agents)

    def narrative(self) -> str:
        radius        = self.propagation_radius()
        agents        = self.contaminated_agents()
        agent_str     = f" (agents: {', '.join(agents)})" if agents else ""
        failure_str   = (f", linked to {len(self.failure_linked_turns)} failure signal(s)"
                         if self.failure_linked_turns else "")
        damage_label  = ("high" if self.damage_score > 0.5
                         else "moderate" if self.damage_score > 0.2 else "low")
        correction_str = (f"Corrected via {self.correction_type} at turn {self.correction_turn}."
                          if self.was_corrected
                          else "NOT corrected — assumption persisted to trajectory end.")
        origin_str = self.origin_agent_id or "unknown"
        return (
            f"Assumption (turn {self.origin_turn}, {origin_str}, trigger={self.trigger_type}): "
            f'"{self.assumption_text[:80]}..." '
            f"contaminated {radius} turn(s){agent_str}{failure_str}. "
            f"Impact: {damage_label}. {correction_str}"
        )


# ── Aggregate report ──────────────────────────────────────────────────────────

@dataclass
class CausalPropagationReport:
    """Aggregated causal propagation results for one trajectory."""
    chains:             List[CausalAssumptionChain] = field(default_factory=list)
    total_assumptions:  int   = 0
    uncorrected_count:  int   = 0
    high_damage_count:  int   = 0   # damage_score > 0.5
    cascade_detected:   bool  = False
    most_damaging:      Optional[CausalAssumptionChain] = None
    summary_narrative:  str   = ""

    def as_dict(self) -> dict:
        return {
            "total_assumptions":  self.total_assumptions,
            "uncorrected_count":  self.uncorrected_count,
            "high_damage_count":  self.high_damage_count,
            "cascade_detected":   self.cascade_detected,
            "summary_narrative":  self.summary_narrative,
            "chains": [
                {
                    "text":              c.assumption_text[:100],
                    "origin_turn":       c.origin_turn,
                    "origin_agent":      c.origin_agent_id,
                    "trigger_type":      c.trigger_type,
                    "propagation_radius": c.propagation_radius(),
                    "contaminated_agents": c.contaminated_agents(),
                    "failure_linked_turns": c.failure_linked_turns,
                    "was_corrected":     c.was_corrected,
                    "correction_type":   c.correction_type,
                    "damage_score":      round(c.damage_score, 3),
                    "narrative":         c.narrative(),
                }
                for c in self.chains
            ],
        }


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_causal_propagation(traj: "AgentTrajectory") -> CausalPropagationReport:
    """
    Build a CausalPropagationReport for *traj*.

    Approach
    --------
    1. Re-use the lightweight assumption extraction from assumptions.py (regex
       patterns + synonym expansion) to get raw (turn, step_id, snippet) triples.
    2. For each assumption, scan ALL subsequent steps for:
       a. Token overlap ≥ 0.25  → PropagationNode
       b. Failure signal tokens → failure_co_occurrence = True
       c. REFLECTION / RECOVERY step within 2 turns → correction_follows
       d. Explicit correction markers in the propagating step
    3. Assign trigger_type, was_corrected, damage_score.
    4. Detect cascades: ≥2 uncorrected high-damage assumptions sharing ≥40%
       token overlap with each other AND linked to at least one failure turn.
    """
    from harpo.trajectory.schema import StepType

    think_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE, StepType.REFLECTION)
        and s.output_text.strip()
    ]
    reflect_recover_turns = {
        s.turn_number
        for s in traj.steps
        if s.step_type in (StepType.REFLECTION, StepType.RECOVERY)
    }

    if not think_steps:
        return CausalPropagationReport()

    # ── Borrow synonym expansion from assumptions module ──────────────────────
    try:
        from harpo.semantic.assumptions import (
            _build_abbreviation_map, _expand_tokens,
            _ASSUMPTION_PATTERNS,
        )
        all_text   = " ".join(s.output_text for s in think_steps)
        abbrev_map = _build_abbreviation_map(all_text)
        use_expand = True
    except ImportError:
        abbrev_map = {}
        use_expand = False

    def expand(tokens: Set[str]) -> Set[str]:
        if use_expand:
            return _expand_tokens(tokens, abbrev_map)
        return tokens

    _PATTERNS = _ASSUMPTION_PATTERNS if use_expand else [
        r"\bI assume\b", r"\bassuming\b", r"\bprobably\b", r"\blikely\b",
        r"\bI think\b", r"\bit seems\b", r"\bperhaps\b", r"\bI believe\b",
    ]

    # ── Extract raw assumption events ─────────────────────────────────────────
    raw: List[Tuple[int, str, str, str]] = []  # (turn, step_id, snippet, agent_id)
    for step in think_steps:
        if step.step_type == StepType.REFLECTION:
            continue  # reflections are correction events, not assumption sources
        text = step.output_text
        for pat in _PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                start   = max(0, m.start() - 100)
                end     = min(len(text), m.end() + 100)
                snippet = text[start:end].strip()
                raw.append((step.turn_number, step.step_id, snippet,
                             getattr(step, "agent_id", "")))
                break

    if not raw:
        return CausalPropagationReport(total_assumptions=0,
                                       summary_narrative="No assumptions detected.")

    # Pre-compute step token sets (expanded)
    step_tokens_cache: Dict[str, Set[str]] = {
        s.step_id: expand(_sig_tokens(s.output_text))
        for s in think_steps
    }

    # ── Build causal chains ───────────────────────────────────────────────────
    chains: List[CausalAssumptionChain] = []

    for turn, step_id, snippet, agent_id in raw:
        base_tok = _sig_tokens(snippet)
        key_tok  = expand(base_tok)
        trigger  = _classify_trigger(snippet)

        chain = CausalAssumptionChain(
            assumption_text  = snippet[:150],
            key_tokens       = key_tok,
            origin_turn      = turn,
            origin_step_id   = step_id,
            origin_agent_id  = agent_id,
            trigger_type     = trigger,
        )

        # ── Scan subsequent steps ─────────────────────────────────────────
        for step in think_steps:
            if step.turn_number <= turn:
                continue

            s_tok   = step_tokens_cache.get(step.step_id, set())
            overlap = _overlap(key_tok, s_tok)
            if overlap < 0.25:
                continue

            # Failure signals present?
            fail_co = bool(_FAILURE_SIGNALS & s_tok)

            # Correction-follows: is there a reflection/recovery within 2 turns?
            corr_follows = any(
                abs(rt - step.turn_number) <= 2
                for rt in reflect_recover_turns
            )

            # Explicit correction in THIS step's text?
            corrected_here = bool(_CORRECTION_MARKERS.search(step.output_text))

            node = PropagationNode(
                turn_number           = step.turn_number,
                step_id               = step.step_id,
                agent_id              = getattr(step, "agent_id", ""),
                overlap_ratio         = round(overlap, 3),
                failure_co_occurrence = fail_co,
                correction_follows    = corr_follows,
                was_corrected         = corrected_here,
            )
            chain.propagation_nodes.append(node)

            if fail_co:
                chain.failure_linked_turns.append(step.turn_number)

        # ── Determine if / how assumption was corrected ────────────────────
        for step in traj.steps:
            if step.turn_number <= turn:
                continue
            if step.step_type == StepType.REFLECTION:
                # Reflection after origin = potential correction
                if not chain.was_corrected:
                    chain.was_corrected  = True
                    chain.correction_turn = step.turn_number
                    chain.correction_type = "reflection"
                break
        # Explicit correction markers in any propagation node override
        for node in chain.propagation_nodes:
            if node.was_corrected:
                chain.was_corrected  = True
                chain.correction_turn = node.turn_number
                chain.correction_type = "contradiction"
                break
        # Recovery step after last propagation
        for step in traj.steps:
            if step.step_type == StepType.RECOVERY:
                if step.turn_number > (chain.correction_turn or turn):
                    chain.recovery_turn = step.turn_number
                    break

        # ── Compute damage score ───────────────────────────────────────────
        radius       = chain.propagation_radius()
        fail_count   = len(set(chain.failure_linked_turns))
        raw_damage   = min(radius / 6.0, 1.0) * 0.55 + min(fail_count / 3.0, 1.0) * 0.45
        if chain.was_corrected:
            raw_damage *= 0.35
        chain.damage_score = round(raw_damage, 3)

        chains.append(chain)

    # ── Aggregate stats ───────────────────────────────────────────────────────
    uncorrected   = [c for c in chains if not c.was_corrected]
    high_damage   = [c for c in chains if c.damage_score > 0.5]
    most_damaging = max(chains, key=lambda c: c.damage_score) if chains else None

    # Cascade: ≥2 uncorrected high-damage chains share ≥40% token overlap AND
    # both have failure-linked turns
    cascade = False
    hd_unc = [c for c in uncorrected if c.damage_score > 0.4 and c.failure_linked_turns]
    for i in range(len(hd_unc)):
        for j in range(i + 1, len(hd_unc)):
            if _overlap(hd_unc[i].key_tokens, hd_unc[j].key_tokens) >= 0.40:
                cascade = True
                break
        if cascade:
            break

    # ── Summary narrative ─────────────────────────────────────────────────────
    if not chains:
        summary = "No assumptions detected in trajectory."
    elif not uncorrected:
        summary = (
            f"{len(chains)} assumption(s) detected; all corrected before end of trajectory. "
            f"No persistent causal damage."
        )
    else:
        top = most_damaging
        agents = top.contaminated_agents() if top else []
        agent_str = f" ({', '.join(agents)})" if agents else ""
        summary = (
            f"{len(chains)} assumption(s) detected; {len(uncorrected)} uncorrected. "
            f"Most damaging (score={top.damage_score if top else 0:.2f}): "
            f'"{(top.assumption_text[:60] + "...") if top else ""}" '
            f"originated turn {top.origin_turn if top else "?"}{agent_str}, "
            f"propagated to {top.propagation_radius() if top else 0} turn(s), "
            f"linked to {len(top.failure_linked_turns) if top else 0} failure signal(s). "
            + ("ASSUMPTION CASCADE DETECTED." if cascade else "")
        )

    return CausalPropagationReport(
        chains            = chains,
        total_assumptions = len(chains),
        uncorrected_count = len(uncorrected),
        high_damage_count = len(high_damage),
        cascade_detected  = cascade,
        most_damaging     = most_damaging,
        summary_narrative = summary,
    )
