"""
Contradiction Detection

Finds self-contradictions in agent reasoning using four signals:
1. Reversal markers — explicit self-correction language
2. Plan flips — "I will X" followed by "I will not X"
3. Negation flips — "X is Y" followed by "X is not Y"
4. Stance reversals — same entity recommended/rejected across distant turns
   (silent structural flip: no verbal marker needed)

No external dependencies. All heuristic + pattern matching.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

# Phrases that signal the agent is correcting something it said before
_REVERSAL_MARKERS = [
    "actually,", "actually ", "wait,", "i was wrong", "to correct myself",
    "my mistake", "correction:", "i should note", "re-evaluating",
    "upon reflection,", "i made an error", "let me correct",
    "i need to reconsider", "that's incorrect", "i apologize",
    "i misspoke", "contrary to", "however earlier", "but i said",
    "correcting my earlier", "i realize i", "i realize that",
    "this contradicts", "contradicts my", "contradicts our",
    "revising my earlier", "updating my prior", "i was incorrect",
    "this was incorrect", "previous assessment was", "earlier estimate was wrong",
    "not 03:12", "not sql", "no longer believe",
]

# Positive stance patterns (entity is recommended / affirmed)
_POSITIVE_STANCES = [
    r'\b(?:recommend|proceed|approve|prioritize|implement|adopt|expand|launch)\s+(?:with\s+)?(\w+)',
    r'\b(\w+)\s+(?:is|are|will be)\s+(?:viable|feasible|recommended|approved|prioritized|preferred)',
    r'\b(\w+)\s+(?:should|must|needs to)\s+(?:be\s+)?(?:implemented|adopted|prioritized|approved)',
]

# Negative stance patterns (entity is rejected / withdrawn)
_NEGATIVE_STANCES = [
    r'\b(?:reject|abandon|withdraw|deprioritize|pause|defer|cancel)\s+(?:from\s+)?(\w+)',
    r'\b(\w+)\s+(?:is|are|will be)\s+(?:not viable|infeasible|rejected|paused|deferred|cancelled)',
    r'\b(?:not|no longer|cannot)\s+(?:recommend|proceed|support|approve)\s+(?:with\s+)?(\w+)',
]

# Common boilerplate words that produce false-positive stances
_STANCE_STOP: Set[str] = {
    "the", "this", "that", "it", "we", "our", "their", "all",
    "any", "some", "most", "more", "very", "also", "just",
    "now", "then", "here", "there", "what", "which", "how",
}


@dataclass
class ContradictionEvent:
    turn_a:     int
    turn_b:     int
    step_id_a:  str
    step_id_b:  str
    kind:       str   # "reversal_marker" | "plan_flip" | "negation_flip" | "stance_reversal"
    snippet_a:  str = ""
    snippet_b:  str = ""


@dataclass
class ContradictionResult:
    contradictions:  List[ContradictionEvent] = field(default_factory=list)
    reversal_count:  int = 0
    flip_count:      int = 0
    affected_turns:  List[int] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.contradictions)

    def severity(self) -> float:
        """0-1 severity. Silent flips penalized more than explicit reversals."""
        if not self.contradictions:
            return 0.0
        return min(self.flip_count * 0.20 + self.reversal_count * 0.10, 1.0)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _extract_stance_entities(text: str, patterns: List[str]) -> Set[str]:
    """Return normalised entity tokens mentioned in a stance pattern match."""
    entities: Set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, text, re.IGNORECASE):
            token = m.group(1).lower().strip()
            if len(token) >= 3 and token not in _STANCE_STOP:
                entities.add(token)
    return entities


def _dedup_events(events: List[ContradictionEvent]) -> List[ContradictionEvent]:
    """Remove duplicate (turn_a, turn_b, kind) events."""
    seen: Set[Tuple] = set()
    out: List[ContradictionEvent] = []
    for e in events:
        key = (e.turn_a, e.turn_b, e.kind, e.snippet_a[:20])
        if key not in seen:
            seen.add(key)
            out.append(e)
    return out


# ── Main detector ─────────────────────────────────────────────────────────────

def detect_contradictions(traj: "AgentTrajectory") -> ContradictionResult:
    """
    Detect contradictions in THINK and RESPONSE steps.

    Returns a ContradictionResult with individual events and aggregate severity.
    """
    from harpo.trajectory.schema import StepType

    think_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE)
        and s.output_text.strip()
    ]
    if len(think_steps) < 2:
        return ContradictionResult()

    events: List[ContradictionEvent] = []
    reversal_count = 0
    flip_count = 0

    # ── Pass 1 — reversal markers ────────────────────────────────
    for step in think_steps:
        text_lower = step.output_text.lower()
        for marker in _REVERSAL_MARKERS:
            if marker in text_lower:
                prior = [s for s in think_steps if s.timestamp < step.timestamp]
                if prior:
                    events.append(ContradictionEvent(
                        turn_a=prior[-1].turn_number,
                        turn_b=step.turn_number,
                        step_id_a=prior[-1].step_id,
                        step_id_b=step.step_id,
                        kind="reversal_marker",
                        snippet_a="",
                        snippet_b=f"…{marker}…",
                    ))
                    reversal_count += 1
                break

    # ── Pass 2 — plan flips ("I will X" → "I will not X"), all pairs ────────
    # Extend search window: compare up to 5 steps ahead, not just adjacent
    for i in range(len(think_steps) - 1):
        s_a = think_steps[i]
        text_a = s_a.output_text
        will_do = re.findall(r'\bI will\s+(\w+)', text_a, re.IGNORECASE)
        if not will_do:
            continue
        for j in range(i + 1, min(i + 6, len(think_steps))):
            s_b = think_steps[j]
            text_b = s_b.output_text
            for verb in will_do[:8]:
                if re.search(rf'\bI will not\s+{re.escape(verb)}\b', text_b, re.IGNORECASE):
                    events.append(ContradictionEvent(
                        turn_a=s_a.turn_number,
                        turn_b=s_b.turn_number,
                        step_id_a=s_a.step_id,
                        step_id_b=s_b.step_id,
                        kind="plan_flip",
                        snippet_a=f"I will {verb}",
                        snippet_b=f"I will not {verb}",
                    ))
                    flip_count += 1
                    break

    # ── Pass 3 — negation flips ("X is Y" → "X is not Y"), all pairs ────────
    for i in range(len(think_steps) - 1):
        s_a = think_steps[i]
        text_a = s_a.output_text.lower()
        is_pairs = re.findall(r'\b(\w{5,})\s+is\s+(\w{3,})\b', text_a)
        if not is_pairs:
            continue
        for j in range(i + 1, min(i + 6, len(think_steps))):
            s_b = think_steps[j]
            text_b = s_b.output_text
            for subj, pred in is_pairs[:5]:
                negated = re.search(
                    rf'\b{re.escape(subj)}\s+is\s+(?:not|never|no longer)\b',
                    text_b,
                    re.IGNORECASE,
                )
                if negated:
                    events.append(ContradictionEvent(
                        turn_a=s_a.turn_number,
                        turn_b=s_b.turn_number,
                        step_id_a=s_a.step_id,
                        step_id_b=s_b.step_id,
                        kind="negation_flip",
                        snippet_a=f"{subj} is {pred}",
                        snippet_b=negated.group(0),
                    ))
                    flip_count += 1
                    break

    # ── Pass 4 — stance reversals (silent, structural, across distant turns) ─
    # Track which entities get positive / negative stances per step.
    # If the same entity gets opposite stances across steps that are ≥2 turns
    # apart, flag it as a silent structural flip.
    step_positive: List[Tuple[int, str, str]] = []   # (turn, step_id, entity)
    step_negative: List[Tuple[int, str, str]] = []

    for step in think_steps:
        pos = _extract_stance_entities(step.output_text, _POSITIVE_STANCES)
        neg = _extract_stance_entities(step.output_text, _NEGATIVE_STANCES)
        for e in pos:
            step_positive.append((step.turn_number, step.step_id, e))
        for e in neg:
            step_negative.append((step.turn_number, step.step_id, e))

    # Find entities that appear in both stance lists with a gap ≥ 2 turns
    pos_by_entity: Dict[str, List[Tuple[int, str]]] = {}
    neg_by_entity: Dict[str, List[Tuple[int, str]]] = {}
    for turn, sid, ent in step_positive:
        pos_by_entity.setdefault(ent, []).append((turn, sid))
    for turn, sid, ent in step_negative:
        neg_by_entity.setdefault(ent, []).append((turn, sid))

    for entity in set(pos_by_entity) & set(neg_by_entity):
        for (t_pos, sid_pos) in pos_by_entity[entity]:
            for (t_neg, sid_neg) in neg_by_entity[entity]:
                if abs(t_neg - t_pos) >= 2:
                    # Stance reversal: earlier positive → later negative OR vice versa
                    t_a, sid_a = min((t_pos, sid_pos), (t_neg, sid_neg))
                    t_b, sid_b = max((t_pos, sid_pos), (t_neg, sid_neg))
                    label_a = "recommend" if t_a == t_pos else "reject"
                    label_b = "reject"    if t_b == t_neg else "recommend"
                    events.append(ContradictionEvent(
                        turn_a=t_a,
                        turn_b=t_b,
                        step_id_a=sid_a,
                        step_id_b=sid_b,
                        kind="stance_reversal",
                        snippet_a=f"{entity}: {label_a}",
                        snippet_b=f"{entity}: {label_b}",
                    ))
                    flip_count += 1

    events = _dedup_events(events)
    affected_turns = sorted({e.turn_b for e in events})
    return ContradictionResult(
        contradictions=events,
        reversal_count=reversal_count,
        flip_count=flip_count,
        affected_turns=affected_turns,
    )
