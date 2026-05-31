"""
Memory Instrumentation

Post-hoc inference of memory read/write operations from trajectory steps,
for frameworks that pass inter-agent context via text injection (judge
feedback, system prompts) rather than through explicit MEMORY_READ/WRITE
events.

In the HARPO incident response demo each agent receives prior agents'
reports as context injection.  No MEMORY_READ events are emitted, so
the memory causal analyzer sees nothing.  This module detects implicit
memory operations by measuring cross-agent vocabulary overlap:

  CONTEXT_INJECTION_READ  — agent B's first THINK step overlaps ≥ 0.30
                             with agent A's final output → implicit read
                             of A's report

  REPORT_WRITE            — an agent's final RESPONSE/THINK step contains
                             dense factual vocabulary (≥ 12 significant
                             tokens) → implicit write of conclusions into
                             shared context

  STALE_CONTEXT_READ      — the source agent's report was later contradicted
                             (contradiction marker in a subsequent step) but
                             the reader's step predates the correction
                             → stale retrieval

  SELF_REFERENCE_READ     — an agent's later step overlaps ≥ 0.50 with its
                             OWN earlier step → the agent is re-reading its
                             own prior conclusions (potential stale-loop)

For each inferred event this module assigns a preliminary causal_hint
(reinforcement / correction / stale_reuse / neutral) based on whether the
retrieved content overlaps with active assumptions and whether correction
markers are present in the reader step.

No external dependencies.
"""

from __future__ import annotations

import re
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

# Token thresholds
_READ_OVERLAP_THRESHOLD  = 0.28   # cross-agent overlap to infer a read
_WRITE_TOKEN_THRESHOLD   = 10     # min significant tokens in final step → write
_SELF_LOOP_THRESHOLD     = 0.55   # self-overlap ratio for stale-loop detection

# Correction signals (if reader step contains these, the retrieval may be corrective)
_CORRECTION_SIGNALS = re.compile(
    r'\b(?:actually|incorrect|wrong|correction|re-evaluating|'
    r'forensics (?:shows|reveals|confirms)|this was not|not sql|'
    r'not 03:12|not at 03|21:43|five hours|earlier than|'
    r'file system confirms|contradicts|revising)\b',
    re.IGNORECASE,
)

# Assumption signals in reader step (retrieval reinforced the assumption)
_ASSUMPTION_SIGNALS = re.compile(
    r'\b(?:based on|given the|as reported|according to|'
    r'from the (?:report|analysis|assessment)|'
    r'as (?:noted|stated) by|building on|following the)\b',
    re.IGNORECASE,
)

# Markdown / context-injection markers (strip these sections for cleaner analysis)
_MARKDOWN_HEAVY = re.compile(r'#{1,3}\s+\w+|---|\*\*\w|\|\s*\w+\s*\|')


def _sig_tokens(text: str) -> Set[str]:
    """Extract significant tokens, stripping markdown artifacts."""
    clean = _MARKDOWN_HEAVY.sub(" ", text)
    return {t for t in re.findall(r'\b[a-z]{3,}\b', clean.lower()) if t not in _STOP}


def _overlap(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _summary_sentence(retrieved_tokens: Set[str], source_agent: str, reader_agent: str) -> str:
    top = sorted(retrieved_tokens, key=len, reverse=True)[:5]
    return (
        f"{reader_agent} read {source_agent}'s report "
        f"(key topics: {', '.join(top[:3])})."
        if top else
        f"{reader_agent} read context from {source_agent}."
    )


@dataclass
class InferredMemoryEvent:
    """A memory operation inferred from cross-agent vocabulary analysis."""
    reader_agent:          str
    source_agent:          str          # "" = global/task context
    reader_turn:           int
    reader_step_id:        str
    operation:             str          # "read" | "write" | "self_read"
    event_subtype:         str          # "context_injection_read" | "report_write" |
                                        # "stale_context_read" | "self_reference_read"
    overlap_ratio:         float        # cross-agent vocabulary overlap
    source_tokens:         Set[str]     # vocabulary from the source step
    reader_tokens:         Set[str]     # vocabulary from the reader step
    is_stale:              bool         # source was later contradicted
    causal_hint:           str          # "reinforcement"|"correction"|"stale_reuse"|"neutral"
    retrieval_summary:     str          # one sentence

    def is_harmful(self) -> bool:
        return self.causal_hint in ("reinforcement", "stale_reuse")

    def is_beneficial(self) -> bool:
        return self.causal_hint == "correction"


def infer_memory_operations(traj: "AgentTrajectory") -> List[InferredMemoryEvent]:
    """
    Scan *traj* for implicit memory operations and return a list of
    InferredMemoryEvents.

    Algorithm
    ---------
    1. Group steps by agent_id, sorted by timestamp.
    2. For each agent, build a "publication signature" from their final
       THINK/RESPONSE step (what they wrote into shared context).
    3. For each agent, check the FIRST THINK step's vocabulary against
       all prior agents' publication signatures.
       Overlap ≥ _READ_OVERLAP_THRESHOLD → inferred CONTEXT_INJECTION_READ.
    4. If the source publication was contradicted in any subsequent step
       (before the reader's turn), mark the read as stale.
    5. Detect REPORT_WRITE events: final step of each agent with
       ≥ _WRITE_TOKEN_THRESHOLD significant tokens.
    6. Detect SELF_REFERENCE_READ: mid-trajectory THINK step with
       high overlap against the agent's own earlier step (agent looping).
    """
    from harpo.trajectory.schema import StepType

    think_types = (StepType.THINK, StepType.RESPONSE)

    # ── Group steps by agent ─────────────────────────────────────────────────
    agent_steps: Dict[str, List["TrajectoryStep"]] = {}
    for step in traj.steps:
        aid = getattr(step, "agent_id", "")
        if aid:
            agent_steps.setdefault(aid, []).append(step)

    # Determine agent order by first-step timestamp
    agent_order = sorted(
        agent_steps.keys(),
        key=lambda a: min(s.timestamp for s in agent_steps[a]),
    )

    if len(agent_order) < 2:
        return []

    # ── Build per-agent "publication signature" from final content step ───────
    agent_publication: Dict[str, Tuple[Set[str], int, str]] = {}  # → (tokens, turn, step_id)
    for aid in agent_order:
        content_steps = [s for s in agent_steps[aid] if s.step_type in think_types and s.output_text]
        if content_steps:
            last = content_steps[-1]
            agent_publication[aid] = (_sig_tokens(last.output_text), last.turn_number, last.step_id)

    # ── Identify correction turns per agent ──────────────────────────────────
    # If any step after an agent's publication contains correction signals,
    # the publication may have been contradicted.
    publication_contradicted: Dict[str, bool] = {aid: False for aid in agent_order}
    for step in traj.steps:
        if _CORRECTION_SIGNALS.search(step.output_text or ""):
            pub_turn = agent_publication.get(step.agent_id or "", (None, None, None))[1]
            for aid in agent_order:
                pub = agent_publication.get(aid)
                if pub and step.turn_number > pub[1]:
                    # Check if the correction overlaps with this agent's publication
                    corr_toks = _sig_tokens(step.output_text)
                    if _overlap(pub[0], corr_toks) >= 0.20:
                        publication_contradicted[aid] = True

    # ── Infer active assumption pool for causal_hint ──────────────────────────
    assumption_pool: Set[str] = set()
    try:
        from harpo.semantic.assumptions import analyze_assumption_propagation
        apr = analyze_assumption_propagation(traj)
        for chain in apr.chains:
            assumption_pool |= chain.key_tokens
    except Exception:
        pass

    # ── Main inference loop ───────────────────────────────────────────────────
    events: List[InferredMemoryEvent] = []

    for i, reader_id in enumerate(agent_order[1:], start=1):
        prior_agents = agent_order[:i]
        reader_steps = [s for s in agent_steps[reader_id] if s.step_type in think_types]
        if not reader_steps:
            continue

        first_step = reader_steps[0]
        first_toks = _sig_tokens(first_step.output_text)

        # ── Pass A: context injection reads from prior agents ─────────────────
        for source_id in prior_agents:
            pub = agent_publication.get(source_id)
            if not pub:
                continue
            src_toks, src_turn, src_step_id = pub

            # Only count if the publication happened BEFORE the reader started
            if src_turn >= first_step.turn_number:
                continue

            ov = _overlap(src_toks, first_toks)
            if ov < _READ_OVERLAP_THRESHOLD:
                continue

            is_stale = publication_contradicted.get(source_id, False)

            # Determine causal_hint
            has_correction = bool(_CORRECTION_SIGNALS.search(first_step.output_text))
            has_assumption  = bool(_ASSUMPTION_SIGNALS.search(first_step.output_text))
            assump_ov       = _overlap(src_toks, assumption_pool) if assumption_pool else 0.0

            if has_correction:
                hint = "correction"
            elif is_stale:
                hint = "stale_reuse"
            elif has_assumption and assump_ov >= 0.25:
                hint = "reinforcement"
            else:
                hint = "neutral"

            events.append(InferredMemoryEvent(
                reader_agent      = reader_id,
                source_agent      = source_id,
                reader_turn       = first_step.turn_number,
                reader_step_id    = first_step.step_id,
                operation         = "read",
                event_subtype     = "stale_context_read" if is_stale else "context_injection_read",
                overlap_ratio     = round(ov, 3),
                source_tokens     = src_toks,
                reader_tokens     = first_toks,
                is_stale          = is_stale,
                causal_hint       = hint,
                retrieval_summary = _summary_sentence(src_toks & first_toks, source_id, reader_id),
            ))

        # ── Pass B: report write from final step ──────────────────────────────
        last_step = reader_steps[-1]
        last_toks = _sig_tokens(last_step.output_text)
        if len(last_toks) >= _WRITE_TOKEN_THRESHOLD:
            events.append(InferredMemoryEvent(
                reader_agent      = reader_id,
                source_agent      = reader_id,
                reader_turn       = last_step.turn_number,
                reader_step_id    = last_step.step_id,
                operation         = "write",
                event_subtype     = "report_write",
                overlap_ratio     = 1.0,
                source_tokens     = last_toks,
                reader_tokens     = last_toks,
                is_stale          = False,
                causal_hint       = "neutral",
                retrieval_summary = (
                    f"{reader_id} wrote conclusions to shared context "
                    f"({len(last_toks)} key terms)."
                ),
            ))

        # ── Pass C: self-reference loop detection ─────────────────────────────
        if len(reader_steps) >= 3:
            for idx in range(2, len(reader_steps)):
                cur  = reader_steps[idx]
                cur_toks = _sig_tokens(cur.output_text)
                # Check against all earlier steps from same agent
                for j in range(max(0, idx - 3), idx):
                    prev = reader_steps[j]
                    prev_toks = _sig_tokens(prev.output_text)
                    ov = _overlap(cur_toks, prev_toks)
                    if ov >= _SELF_LOOP_THRESHOLD:
                        events.append(InferredMemoryEvent(
                            reader_agent      = reader_id,
                            source_agent      = reader_id,
                            reader_turn       = cur.turn_number,
                            reader_step_id    = cur.step_id,
                            operation         = "self_read",
                            event_subtype     = "self_reference_read",
                            overlap_ratio     = round(ov, 3),
                            source_tokens     = prev_toks,
                            reader_tokens     = cur_toks,
                            is_stale          = False,
                            causal_hint       = "neutral",
                            retrieval_summary = (
                                f"{reader_id} re-read its own prior step "
                                f"(turn {prev.turn_number}, overlap={ov:.2f})."
                            ),
                        ))
                        break   # one self-loop per step is enough

    return events
