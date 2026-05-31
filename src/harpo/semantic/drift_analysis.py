"""
Silent Drift Intelligence

Detects cases where an agent's behavior silently shifts from its established
objective without any verbal acknowledgment — the most insidious failure mode
in long-horizon autonomous agents.

Five drift types:
  OBJECTIVE_DRIFT      — goal tokens from the original task disappear from reasoning
  PRIORITY_SHIFT       — a secondary concern (PR, legal, stakeholder) overtakes the primary goal
  ATTENTION_COLLAPSE   — key technical/factual entities stop appearing across ≥3 consecutive turns
  GOAL_MUTATION        — the framing of the core problem changes measurably
  COORDINATION_DRIFT   — (multi-agent) agents diverge from the shared initial objective

For each event this module reports:
  • WHY it occurred (dominant new token cluster that replaced objective tokens)
  • WHICH turn it started and when it was first detectable
  • WHETHER recovery happened (objective tokens returned within 3 turns)

Unlike contradiction.py (logical consistency) and assumptions.py (epistemic
uncertainty), drift_analysis.py measures *gradual directional change* in what
the agent is attending to across the arc of a trajectory.

No external dependencies.  Pure token + sliding-window heuristics.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

# ── Constants ─────────────────────────────────────────────────────────────────
_OBJECTIVE_WINDOW     = 2    # first N THINK steps used to establish "objective tokens"
_DRIFT_THRESHOLD      = 0.15 # core-overlap below this triggers objective drift
_COLLAPSE_WINDOW      = 3    # N consecutive turns an entity must be absent → collapse
_RECOVERY_WINDOW      = 3    # turns after drift within which return is counted as recovery
_MIN_ENTITY_LEN       = 5    # minimum char length for attention entities
_PRIORITY_SHIFT_RATIO = 1.8  # secondary cluster must be N× larger than objective cluster

# Tokens signalling external pressure / priority hijacking
_PRESSURE_TOKENS: Set[str] = {
    "stakeholder", "shareholder", "board", "legal", "compliance", "regulatory",
    "lawyer", "media", "press", "optics", "reputation", "brand", "pr",
    "customer", "client", "deadline", "executive", "ceo", "leadership",
    "political", "perception", "narrative", "spin", "communicate",
    "announcement", "statement", "public", "liability", "lawsuit",
    "insurance", "penalty", "fine", "sanction",
}

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
    "need", "get", "use", "make", "take", "look", "work", "good",
    "want", "give", "know", "see", "say", "come", "think", "first",
    "time", "way", "been", "has", "like", "just", "over", "such",
    "also", "back", "after", "between", "through", "each",
}


class DriftType(str, Enum):
    OBJECTIVE_DRIFT    = "objective_drift"
    PRIORITY_SHIFT     = "priority_shift"
    ATTENTION_COLLAPSE = "attention_collapse"
    GOAL_MUTATION      = "goal_mutation"
    COORDINATION_DRIFT = "coordination_drift"


@dataclass
class DriftEvent:
    """A single detected drift occurrence."""
    drift_type:        DriftType
    turn_start:        int              # when the drift baseline was set
    turn_detected:     int              # first turn where drift was observable
    objective_overlap: float            # objective token overlap at detection time
    signal_tokens:     Set[str]         # tokens that disappeared or were displaced
    new_tokens:        Set[str]         # tokens that replaced them
    pressure_triggered: bool            # new_tokens include external pressure signals
    trigger_evidence:  str              # brief snippet from the turn where drift peaked
    recovery_detected: bool             = False
    recovery_turn:     Optional[int]    = None
    agent_id:          str              = ""   # for coordination drift

    def why(self) -> str:
        """One-sentence causal explanation."""
        if self.pressure_triggered:
            pressure = sorted(self.new_tokens & _PRESSURE_TOKENS)[:3]
            return (f"Drift triggered by external pressure signals "
                    f"({', '.join(pressure)}), displacing core objective tokens.")
        if self.new_tokens:
            top = sorted(self.new_tokens, key=len, reverse=True)[:3]
            return (f"Attention shifted to new topic cluster "
                    f"({', '.join(top)}), causing objective token displacement.")
        return "Objective tokens dropped below coherence threshold with no replacement cluster."

    def recovery_note(self) -> str:
        if self.recovery_detected:
            return f"Recovered at turn {self.recovery_turn}."
        return "No recovery detected — drift persisted to trajectory end."

    def narrative(self) -> str:
        agent_str = f" [{self.agent_id}]" if self.agent_id else ""
        return (
            f"{self.drift_type.value.upper()}{agent_str}: detected turn {self.turn_detected} "
            f"(objective overlap={self.objective_overlap:.2f}). "
            f"{self.why()} {self.recovery_note()}"
        )


@dataclass
class DriftReport:
    """Aggregated drift analysis for one trajectory."""
    events:                    List[DriftEvent] = field(default_factory=list)
    objective_drift_detected:  bool             = False
    attention_collapse_turns:  List[int]        = field(default_factory=list)
    drift_onset_turn:          Optional[int]    = None   # earliest drift event
    recovery_rate:             float            = 0.0    # fraction of drifts that recovered
    overall_drift_score:       float            = 0.0    # 0=stable 1=severe
    objective_token_profile:   List[Tuple[int, float]] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "total_drift_events":      len(self.events),
            "objective_drift":         self.objective_drift_detected,
            "attention_collapse_turns": self.attention_collapse_turns,
            "drift_onset_turn":        self.drift_onset_turn,
            "recovery_rate":           round(self.recovery_rate, 3),
            "overall_drift_score":     round(self.overall_drift_score, 3),
            "events": [
                {
                    "type":             e.drift_type.value,
                    "turn_detected":    e.turn_detected,
                    "objective_overlap": round(e.objective_overlap, 3),
                    "pressure_triggered": e.pressure_triggered,
                    "recovery":         e.recovery_detected,
                    "recovery_turn":    e.recovery_turn,
                    "agent_id":         e.agent_id,
                    "narrative":        e.narrative(),
                }
                for e in self.events
            ],
        }

    def narrative(self) -> str:
        if not self.events:
            return "No drift detected. Trajectory maintained objective alignment throughout."
        n_obj  = sum(1 for e in self.events if e.drift_type == DriftType.OBJECTIVE_DRIFT)
        n_prio = sum(1 for e in self.events if e.drift_type == DriftType.PRIORITY_SHIFT)
        n_coll = sum(1 for e in self.events if e.drift_type == DriftType.ATTENTION_COLLAPSE)
        n_rec  = sum(1 for e in self.events if e.recovery_detected)
        parts  = []
        if n_obj:
            parts.append(f"{n_obj} objective drift(s)")
        if n_prio:
            parts.append(f"{n_prio} priority shift(s)")
        if n_coll:
            parts.append(f"{n_coll} attention collapse(s)")
        parts_str = ", ".join(parts)
        onset     = f" (onset: turn {self.drift_onset_turn})" if self.drift_onset_turn else ""
        rec_str   = f"{n_rec}/{len(self.events)} recovered" if self.events else ""
        return (
            f"Drift detected: {parts_str}{onset}. "
            f"Overall drift score: {self.overall_drift_score:.2f}. {rec_str}."
        )


# ── Internal helpers ──────────────────────────────────────────────────────────

def _sig_tokens(text: str, min_len: int = 3) -> Set[str]:
    return {
        t for t in re.findall(r'\b[a-z]{%d,}\b' % min_len, text.lower())
        if t not in _STOP
    }


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _dominant_cluster(text: str, exclude: Set[str]) -> Set[str]:
    """Top-10 tokens by frequency that are NOT in the exclude set."""
    tokens = [t for t in re.findall(r'\b[a-z]{4,}\b', text.lower())
              if t not in _STOP and t not in exclude]
    if not tokens:
        return set()
    counts = Counter(tokens)
    top    = {tok for tok, _ in counts.most_common(10)}
    return top


def _is_significant_entity(token: str) -> bool:
    """Heuristic: technical / domain-specific terms worth tracking."""
    return len(token) >= _MIN_ENTITY_LEN and token not in _STOP


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_drift(traj: "AgentTrajectory") -> DriftReport:
    """
    Analyse *traj* for silent drift events across all five drift types.

    Returns a DriftReport with per-event narratives and an overall_drift_score.
    """
    from harpo.trajectory.schema import StepType

    think_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE)
        and len(s.output_text.split()) >= 8
    ]

    if len(think_steps) < 4:
        return DriftReport(
            overall_drift_score=0.0,
        )

    # Pre-compute token sets per step
    step_toks: List[Set[str]] = [_sig_tokens(s.output_text) for s in think_steps]
    step_turns: List[int] = [s.turn_number for s in think_steps]

    # ── Build objective token baseline from first _OBJECTIVE_WINDOW steps ─────
    seed_n = min(_OBJECTIVE_WINDOW, len(think_steps))
    objective_tokens: Set[str] = set()
    for i in range(seed_n):
        objective_tokens |= step_toks[i]

    # Also capture attention entities (longer tokens from the task description)
    task_text = traj.task_description + " " + traj.user_intent
    attention_entities: Set[str] = {
        t for t in _sig_tokens(task_text, min_len=_MIN_ENTITY_LEN)
    }
    # Add entities from first 2 steps
    for i in range(min(2, len(think_steps))):
        attention_entities |= {t for t in step_toks[i] if len(t) >= _MIN_ENTITY_LEN}

    events: List[DriftEvent] = []
    obj_profile: List[Tuple[int, float]] = []   # (turn, overlap_with_objective)

    # ── Pass 1 — objective drift + priority shift ─────────────────────────────
    consecutive_drift = 0
    prev_was_drift    = False
    for i in range(seed_n, len(think_steps)):
        s    = think_steps[i]
        toks = step_toks[i]
        turn = step_turns[i]

        overlap = _jaccard(toks, objective_tokens)
        obj_profile.append((turn, round(overlap, 4)))

        if overlap < _DRIFT_THRESHOLD:
            # Objective drift
            new_cluster    = _dominant_cluster(s.output_text, objective_tokens)
            pressure_hit   = bool(new_cluster & _PRESSURE_TOKENS)
            lost_tokens    = objective_tokens - toks
            snippet        = s.output_text[:120].replace("\n", " ")

            events.append(DriftEvent(
                drift_type         = DriftType.OBJECTIVE_DRIFT,
                turn_start         = step_turns[seed_n - 1],
                turn_detected      = turn,
                objective_overlap  = overlap,
                signal_tokens      = lost_tokens,
                new_tokens         = new_cluster,
                pressure_triggered = pressure_hit,
                trigger_evidence   = snippet,
                agent_id           = getattr(s, "agent_id", ""),
            ))
            consecutive_drift += 1

            # Priority shift: secondary cluster much larger than objective footprint
            obj_footprint = len(objective_tokens & toks)
            sec_footprint = len(new_cluster - objective_tokens)
            if sec_footprint >= obj_footprint * _PRIORITY_SHIFT_RATIO and new_cluster:
                events.append(DriftEvent(
                    drift_type         = DriftType.PRIORITY_SHIFT,
                    turn_start         = step_turns[seed_n - 1],
                    turn_detected      = turn,
                    objective_overlap  = overlap,
                    signal_tokens      = objective_tokens - toks,
                    new_tokens         = new_cluster,
                    pressure_triggered = pressure_hit,
                    trigger_evidence   = snippet,
                    agent_id           = getattr(s, "agent_id", ""),
                ))
        else:
            # Check recovery: was there a drift in the previous _RECOVERY_WINDOW turns?
            for ev in events:
                if (not ev.recovery_detected
                        and ev.drift_type in (DriftType.OBJECTIVE_DRIFT, DriftType.PRIORITY_SHIFT)
                        and abs(turn - ev.turn_detected) <= _RECOVERY_WINDOW):
                    ev.recovery_detected = True
                    ev.recovery_turn     = turn
            consecutive_drift = 0

        prev_was_drift = overlap < _DRIFT_THRESHOLD

    # ── Pass 2 — attention collapse ───────────────────────────────────────────
    collapse_turns: List[int] = []
    if attention_entities:
        absent_count: Dict[str, int] = {e: 0 for e in attention_entities}
        for i in range(seed_n, len(think_steps)):
            s    = think_steps[i]
            toks = step_toks[i]
            turn = step_turns[i]
            for entity in attention_entities:
                if entity not in toks:
                    absent_count[entity] += 1
                else:
                    absent_count[entity] = 0
            # If any entity is absent for ≥ _COLLAPSE_WINDOW consecutive steps
            collapsing = {e for e, cnt in absent_count.items() if cnt >= _COLLAPSE_WINDOW}
            if collapsing:
                snippet = s.output_text[:120].replace("\n", " ")
                events.append(DriftEvent(
                    drift_type         = DriftType.ATTENTION_COLLAPSE,
                    turn_start         = step_turns[max(0, i - _COLLAPSE_WINDOW)],
                    turn_detected      = turn,
                    objective_overlap  = _jaccard(toks, objective_tokens),
                    signal_tokens      = collapsing,
                    new_tokens         = set(),
                    pressure_triggered = False,
                    trigger_evidence   = snippet,
                    agent_id           = getattr(s, "agent_id", ""),
                ))
                collapse_turns.append(turn)
                # Reset so we don't re-fire every step
                for e in collapsing:
                    absent_count[e] = 0

    # ── Pass 3 — goal mutation (framing change) ───────────────────────────────
    # Compare token signature of first 2 steps vs last 2 steps
    if len(think_steps) >= 6:
        early_toks = step_toks[0] | step_toks[1]
        late_toks  = step_toks[-2] | step_toks[-1]
        mutation   = 1.0 - _jaccard(early_toks, late_toks)
        # Only flag if mutation is substantial AND not covered by objective drift already
        obj_drift_turns = {e.turn_detected for e in events
                           if e.drift_type == DriftType.OBJECTIVE_DRIFT}
        if mutation > 0.55 and not obj_drift_turns:
            new_cluster = late_toks - early_toks
            events.append(DriftEvent(
                drift_type         = DriftType.GOAL_MUTATION,
                turn_start         = step_turns[0],
                turn_detected      = step_turns[-1],
                objective_overlap  = _jaccard(late_toks, objective_tokens),
                signal_tokens      = early_toks - late_toks,
                new_tokens         = new_cluster,
                pressure_triggered = bool(new_cluster & _PRESSURE_TOKENS),
                trigger_evidence   = think_steps[-1].output_text[:120].replace("\n", " "),
                agent_id           = "",
            ))

    # ── Pass 4 — coordination drift (multi-agent) ─────────────────────────────
    agent_ids = {getattr(s, "agent_id", "") for s in think_steps} - {""}
    if len(agent_ids) >= 2:
        # Per-agent: compare their last-2 steps against the shared objective
        agent_last_steps: Dict[str, List[Set[str]]] = {}
        for i, s in enumerate(think_steps):
            aid = getattr(s, "agent_id", "")
            if aid:
                agent_last_steps.setdefault(aid, []).append(step_toks[i])

        for aid, tok_list in agent_last_steps.items():
            if len(tok_list) < 2:
                continue
            late_agent_toks = tok_list[-1] | tok_list[-2]
            coord_overlap   = _jaccard(late_agent_toks, objective_tokens)
            if coord_overlap < _DRIFT_THRESHOLD:
                new_cl = late_agent_toks - objective_tokens
                events.append(DriftEvent(
                    drift_type         = DriftType.COORDINATION_DRIFT,
                    turn_start         = step_turns[seed_n - 1],
                    turn_detected      = step_turns[-1],
                    objective_overlap  = coord_overlap,
                    signal_tokens      = objective_tokens - late_agent_toks,
                    new_tokens         = new_cl,
                    pressure_triggered = bool(new_cl & _PRESSURE_TOKENS),
                    trigger_evidence   = "",
                    agent_id           = aid,
                ))

    # ── Deduplicate events at same (type, turn) ───────────────────────────────
    seen: set = set()
    unique_events: List[DriftEvent] = []
    for ev in events:
        key = (ev.drift_type, ev.turn_detected, ev.agent_id)
        if key not in seen:
            seen.add(key)
            unique_events.append(ev)
    events = unique_events

    # ── Aggregate ─────────────────────────────────────────────────────────────
    obj_drift = any(e.drift_type == DriftType.OBJECTIVE_DRIFT for e in events)
    onset     = min((e.turn_detected for e in events), default=None)
    n_rec     = sum(1 for e in events if e.recovery_detected)
    rec_rate  = n_rec / len(events) if events else 0.0

    # Drift score: weighted by event type severity and non-recovery
    type_weights = {
        DriftType.OBJECTIVE_DRIFT:    0.40,
        DriftType.PRIORITY_SHIFT:     0.30,
        DriftType.ATTENTION_COLLAPSE: 0.15,
        DriftType.GOAL_MUTATION:      0.25,
        DriftType.COORDINATION_DRIFT: 0.20,
    }
    raw_score = sum(type_weights.get(e.drift_type, 0.10) * (0.5 if e.recovery_detected else 1.0)
                    for e in events)
    drift_score = round(min(raw_score, 1.0), 4)

    return DriftReport(
        events                   = events,
        objective_drift_detected = obj_drift,
        attention_collapse_turns = collapse_turns,
        drift_onset_turn         = onset,
        recovery_rate            = round(rec_rate, 3),
        overall_drift_score      = drift_score,
        objective_token_profile  = obj_profile,
    )
