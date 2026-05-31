"""
Memory Causality Analysis

Tracks whether memory retrieval and storage events played a causal role in
trajectory degradation or stabilization — distinguishing:

  REINFORCEMENT    — a MEMORY_READ step retrieved content that overlaps ≥40%
                     with a propagating assumption's key tokens
                     → memory made the assumption more entrenched

  CORRECTION       — a MEMORY_READ step retrieved content that overlaps ≥40%
                     with *correction marker* tokens (contradicting the assumption)
                     → memory corrected a faulty assumption

  STALE_REUSE      — a MEMORY_READ step reused content from early in the
                     trajectory (or from task context) in a later turn where
                     situational facts have demonstrably changed (context drift
                     score > 0.5 between the retrieval time and current turn)
                     → harmful retrieval: outdated evidence re-applied

  ASSUMPTION_STORAGE — a MEMORY_WRITE step stored text that contains assumption
                       pattern markers, persisting an unverified claim
                       → assumption embedded in persistent memory

  NEUTRAL          — memory event present but no causal signal detected

Key questions answered:
  • Which specific memory operations reinforced faulty assumptions?
  • Which corrected them?
  • Was harmful information stored that would propagate into future turns?
  • What is the net memory causality (harmful / beneficial / neutral)?

No external dependencies.  Heuristic token-overlap analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

# ── Shared token helpers ──────────────────────────────────────────────────────
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

_CORRECTION_TOKENS: Set[str] = {
    "incorrect", "wrong", "false", "contradiction", "contradicts", "error",
    "actually", "corrected", "revised", "updated", "recalculate", "reconsider",
    "forensics", "confirmed", "verified", "evidence", "disproves",
}

_ASSUMPTION_MARKERS = re.compile(
    r'\b(?:i assume|assuming|probably|likely|i think|it seems|'
    r'perhaps|i believe|i suppose|presumably|it appears|apparently)\b',
    re.IGNORECASE,
)

_STALENESS_MARKERS = re.compile(
    r'\b(?:earlier|previous|initial|original|at first|we thought|'
    r'based on earlier|previously|no longer|has changed|update:|'
    r'new information|new evidence|correction)\b',
    re.IGNORECASE,
)

# Memory reinforcement / correction thresholds
_REINFORCE_OVERLAP = 0.35   # assumption tokens that re-appear in memory content
_CORRECT_OVERLAP   = 0.30   # correction tokens that appear in memory content
_STALE_DRIFT_TURNS = 4      # memory at turn T used at turn T+N where N > this


def _sig_tokens(text: str) -> Set[str]:
    return {t for t in re.findall(r'\b[a-z]{3,}\b', text.lower()) if t not in _STOP}


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class MemoryCausalEvent:
    """One memory operation with its causal role annotated."""
    step_id:               str
    turn_number:           int
    operation:             str      # "read" | "write" | "compact"
    causal_role:           str      # "reinforcement"|"correction"|"stale_reuse"|
                                    # "assumption_storage"|"neutral"
    memory_content_snippet: str     # first 120 chars of memory access content/key
    linked_assumption_text: str     # nearest active assumption (empty if neutral)
    overlap_score:         float    # token overlap driving the classification
    impact_description:    str      # one-sentence narrative

    def is_harmful(self) -> bool:
        return self.causal_role in ("reinforcement", "stale_reuse", "assumption_storage")

    def is_beneficial(self) -> bool:
        return self.causal_role == "correction"


@dataclass
class MemoryCausalReport:
    """Aggregated memory causality results for one trajectory."""
    events:                  List[MemoryCausalEvent] = field(default_factory=list)
    reinforcement_count:     int   = 0
    correction_count:        int   = 0
    stale_reuse_count:       int   = 0
    assumption_storage_count: int  = 0
    neutral_count:           int   = 0
    net_causality:           str   = "neutral"   # "harmful" | "beneficial" | "neutral"
    harmful_turn_range:      Tuple[Optional[int], Optional[int]] = (None, None)

    def as_dict(self) -> dict:
        return {
            "total_memory_events":      len(self.events),
            "reinforcement_count":      self.reinforcement_count,
            "correction_count":         self.correction_count,
            "stale_reuse_count":        self.stale_reuse_count,
            "assumption_storage_count": self.assumption_storage_count,
            "neutral_count":            self.neutral_count,
            "net_causality":            self.net_causality,
            "harmful_turn_range":       list(self.harmful_turn_range),
            "events": [
                {
                    "step_id":        e.step_id,
                    "turn":           e.turn_number,
                    "operation":      e.operation,
                    "causal_role":    e.causal_role,
                    "overlap_score":  round(e.overlap_score, 3),
                    "impact":         e.impact_description,
                }
                for e in self.events
            ],
        }

    def narrative(self) -> str:
        if not self.events:
            return "No memory operations detected — memory causality not applicable."
        parts = []
        if self.reinforcement_count:
            parts.append(f"{self.reinforcement_count} reinforcement(s)")
        if self.correction_count:
            parts.append(f"{self.correction_count} correction(s)")
        if self.stale_reuse_count:
            parts.append(f"{self.stale_reuse_count} stale reuse(s)")
        if self.assumption_storage_count:
            parts.append(f"{self.assumption_storage_count} assumption storage event(s)")
        harm_note = ""
        if self.harmful_turn_range[0] is not None:
            harm_note = (f" Harmful memory window: turns "
                         f"{self.harmful_turn_range[0]}-{self.harmful_turn_range[1]}.")
        return (
            f"Memory causality [{self.net_causality.upper()}]: "
            + (", ".join(parts) or "all neutral")
            + "."
            + harm_note
        )


# ── Main analyzer ─────────────────────────────────────────────────────────────

def _from_inferred(inferred) -> "MemoryCausalEvent":
    """Convert an InferredMemoryEvent to a MemoryCausalEvent."""
    return MemoryCausalEvent(
        step_id                = inferred.reader_step_id,
        turn_number            = inferred.reader_turn,
        operation              = inferred.operation,
        causal_role            = inferred.causal_hint,
        memory_content_snippet = (
            ", ".join(sorted(inferred.source_tokens & inferred.reader_tokens)[:10])
        ),
        linked_assumption_text = "",
        overlap_score          = inferred.overlap_ratio,
        impact_description     = inferred.retrieval_summary,
    )


def analyze_memory_causality(traj: "AgentTrajectory") -> MemoryCausalReport:
    """
    Scan *traj* for MEMORY_READ / MEMORY_WRITE steps and classify each
    as reinforcement, correction, stale_reuse, assumption_storage, or neutral.

    Fallback (no explicit memory steps): uses memory_instrumentation to infer
    memory operations from cross-agent vocabulary overlap.  This covers
    frameworks that pass context via text injection rather than explicit events.

    The analysis proceeds in three passes:

    Pass 1 — Build the active assumption pool from THINK steps.
    Pass 2 — For each memory event: overlap with assumptions → reinforcement;
             correction markers → correction; stale content → stale_reuse.
    Pass 3 — MEMORY_WRITE with assumption markers → assumption_storage.
    """
    from harpo.trajectory.schema import StepType

    steps = traj.steps
    if not steps:
        return MemoryCausalReport()

    # ── Collect assumption pool ───────────────────────────────────────────────
    # Try to import the causal propagation chains for richer assumption data.
    assumption_pool: List[Tuple[int, Set[str], str]] = []  # (turn, tokens, text)
    try:
        from harpo.semantic.assumptions import (
            analyze_assumption_propagation,
            _build_abbreviation_map, _expand_tokens,
        )
        all_think_text = " ".join(
            s.output_text for s in steps
            if s.step_type in (StepType.THINK, StepType.RESPONSE)
        )
        abbrev_map = _build_abbreviation_map(all_think_text)
        apr        = analyze_assumption_propagation(traj)
        for chain in apr.chains:
            tok = _expand_tokens(chain.key_tokens, abbrev_map)
            assumption_pool.append((chain.introduced_turn, tok, chain.text[:100]))
    except Exception:
        # Fallback: scan for assumption patterns directly
        _ASSUMPTION_PATTERNS = [
            r"\bI assume\b", r"\bassuming\b", r"\bprobably\b", r"\blikely\b",
            r"\bI think\b", r"\bit seems\b", r"\bI believe\b",
        ]
        for s in steps:
            if s.step_type not in (StepType.THINK, StepType.RESPONSE):
                continue
            for pat in _ASSUMPTION_PATTERNS:
                m = re.search(pat, s.output_text, re.IGNORECASE)
                if m:
                    start   = max(0, m.start() - 80)
                    end     = min(len(s.output_text), m.end() + 80)
                    snippet = s.output_text[start:end]
                    assumption_pool.append((s.turn_number, _sig_tokens(snippet), snippet[:100]))
                    break

    # ── Identify memory steps ─────────────────────────────────────────────────
    mem_steps = [
        s for s in steps
        if s.step_type in (StepType.MEMORY_READ, StepType.MEMORY_WRITE)
    ]

    if not mem_steps:
        # ── Fallback: infer memory operations from cross-agent overlap ─────────
        try:
            from harpo.memory.memory_instrumentation import infer_memory_operations
            inferred = infer_memory_operations(traj)
            if inferred:
                events = [_from_inferred(e) for e in inferred]
                # Enrich with linked assumption text
                for ev, inf_ev in zip(events, inferred):
                    active = [atext for (at, atoks, atext) in assumption_pool
                              if at <= inf_ev.reader_turn]
                    if active and ev.causal_role in ("reinforcement", "stale_reuse"):
                        ev.linked_assumption_text = active[-1]

                rc  = sum(1 for e in events if e.causal_role == "reinforcement")
                cc  = sum(1 for e in events if e.causal_role == "correction")
                sc  = sum(1 for e in events if e.causal_role == "stale_reuse")
                ac  = 0
                nc  = sum(1 for e in events if e.causal_role == "neutral")
                harmful  = [e for e in events if e.is_harmful()]
                benef    = [e for e in events if e.is_beneficial()]
                net = ("harmful" if len(harmful) > len(benef) + 1
                       else "beneficial" if len(benef) > len(harmful)
                       else "neutral")
                harm_turns = [e.turn_number for e in harmful]
                harm_range: Tuple[Optional[int], Optional[int]] = (
                    (min(harm_turns), max(harm_turns)) if harm_turns else (None, None)
                )
                return MemoryCausalReport(
                    events=events,
                    reinforcement_count=rc,
                    correction_count=cc,
                    stale_reuse_count=sc,
                    assumption_storage_count=ac,
                    neutral_count=nc,
                    net_causality=net,
                    harmful_turn_range=harm_range,
                )
        except Exception:
            pass
        return MemoryCausalReport(net_causality="neutral", neutral_count=0)

    events: List[MemoryCausalEvent] = []

    # Track previously-seen memory content (for stale-reuse detection)
    prior_mem_content: List[Tuple[int, Set[str]]] = []  # (turn, tokens)

    for step in mem_steps:
        op = step.step_type.value   # "memory_read" | "memory_write"

        # Extract memory content from MemoryAccess or output_text
        mem_text = ""
        if step.memory_access:
            ma = step.memory_access
            if ma.value:
                mem_text = str(ma.value)
            elif ma.key:
                mem_text = str(ma.key)
        mem_text = mem_text or step.output_text or step.input_text
        mem_toks = _sig_tokens(mem_text)

        turn = step.turn_number

        # ── Find active assumptions at this turn ──────────────────────────────
        active_assumptions = [
            (at, atoks, atext)
            for (at, atoks, atext) in assumption_pool
            if at <= turn
        ]

        best_role  = "neutral"
        best_overlap = 0.0
        best_assumption_text = ""

        # ── Reinforcement: memory overlaps with active assumption ─────────────
        for (_, atoks, atext) in active_assumptions:
            ov = _overlap(mem_toks, atoks)
            if ov >= _REINFORCE_OVERLAP and ov > best_overlap:
                best_role             = "reinforcement"
                best_overlap          = ov
                best_assumption_text  = atext

        # ── Correction: memory contains correction tokens ─────────────────────
        corr_ov = _overlap(mem_toks, _CORRECTION_TOKENS)
        if corr_ov >= _CORRECT_OVERLAP:
            if corr_ov > best_overlap or best_role == "neutral":
                best_role             = "correction"
                best_overlap          = corr_ov
                best_assumption_text  = active_assumptions[-1][2] if active_assumptions else ""

        # ── Stale reuse: same content retrieved again after context shifted ────
        if op == "memory_read" and prior_mem_content:
            for (prior_turn, prior_toks) in prior_mem_content:
                if turn - prior_turn >= _STALE_DRIFT_TURNS:
                    stale_ov = _jaccard(mem_toks, prior_toks)
                    if stale_ov >= 0.55:  # substantially same content retrieved again
                        if best_role == "neutral" or best_role == "reinforcement":
                            best_role    = "stale_reuse"
                            best_overlap = stale_ov
                        break

        # ── Assumption storage: WRITE of assumption-pattern text ──────────────
        if op == "memory_write" and _ASSUMPTION_MARKERS.search(mem_text):
            if best_role == "neutral":
                best_role = "assumption_storage"
                best_assumption_text = mem_text[:100]

        # ── Build impact description ──────────────────────────────────────────
        if best_role == "reinforcement":
            desc = (f"Memory retrieval (overlap={best_overlap:.2f}) reinforced active assumption: "
                    f'"{best_assumption_text[:60]}..."')
        elif best_role == "correction":
            desc = (f"Memory retrieval contained correction signals (overlap={best_overlap:.2f}), "
                    f"potentially countering active assumptions.")
        elif best_role == "stale_reuse":
            desc = (f"Memory retrieval reused content from an earlier context window "
                    f"(similarity={best_overlap:.2f}), potentially outdated.")
        elif best_role == "assumption_storage":
            desc = (f"Memory write stored unverified assumption text: "
                    f'"{best_assumption_text[:60]}..."')
        else:
            desc = "Memory operation with no detectable causal signal."

        events.append(MemoryCausalEvent(
            step_id                = step.step_id,
            turn_number            = turn,
            operation              = op.replace("memory_", ""),
            causal_role            = best_role,
            memory_content_snippet = mem_text[:120],
            linked_assumption_text = best_assumption_text,
            overlap_score          = round(best_overlap, 3),
            impact_description     = desc,
        ))

        prior_mem_content.append((turn, mem_toks))

    # ── Aggregate ─────────────────────────────────────────────────────────────
    rc  = sum(1 for e in events if e.causal_role == "reinforcement")
    cc  = sum(1 for e in events if e.causal_role == "correction")
    sc  = sum(1 for e in events if e.causal_role == "stale_reuse")
    ac  = sum(1 for e in events if e.causal_role == "assumption_storage")
    nc  = sum(1 for e in events if e.causal_role == "neutral")

    harmful_events   = [e for e in events if e.is_harmful()]
    beneficial_events = [e for e in events if e.is_beneficial()]

    if len(harmful_events) > len(beneficial_events) + 1:
        net = "harmful"
    elif len(beneficial_events) > len(harmful_events):
        net = "beneficial"
    else:
        net = "neutral"

    harm_turns = [e.turn_number for e in harmful_events]
    harm_range: Tuple[Optional[int], Optional[int]] = (
        (min(harm_turns), max(harm_turns)) if harm_turns else (None, None)
    )

    return MemoryCausalReport(
        events                   = events,
        reinforcement_count      = rc,
        correction_count         = cc,
        stale_reuse_count        = sc,
        assumption_storage_count = ac,
        neutral_count            = nc,
        net_causality            = net,
        harmful_turn_range       = harm_range,
    )
