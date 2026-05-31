"""
Causal Chain Summarizer

Converts raw CausalAssumptionChain objects into clean, structured,
executive-readable records — stripping markdown artifacts, truncated
sentence fragments, and context-injection noise.

Each assumption chain becomes a CausalChainSummary:

  Origin Agent:    security-analyst
  Assumption:      Intrusion began at 03:12 UTC based on SIEM alert time
  Affected Agents: compliance-agent, comms-officer, incident-commander
  Damage:          Incorrect GDPR 72-hour window baseline; SLA miscalculated
  Correction:      forensics-agent (turn 3)
  Outcome:         RESOLVED

  One-line:
    "Security Analyst's timeline assumption (03:12 UTC) contaminated 3 agents;
     corrected by Forensics Agent. GDPR deadline risk remains."

Executive Summary paragraph aggregates all chains into:
  WHY it failed, WHICH chains were most damaging, WHETHER recovery happened.

No external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

if TYPE_CHECKING:
    from harpo.semantic.causal_propagation import CausalAssumptionChain, CausalPropagationReport

# ── Markdown / context-injection noise patterns ───────────────────────────────
_NOISE_PATTERNS = [
    re.compile(r'#{1,3}\s+\w'),        # ## Header
    re.compile(r'\*\*[A-Z]'),          # **Bold
    re.compile(r'- \*\*'),             # - **list
    re.compile(r'---+'),               # --- divider
    re.compile(r'>\s+\w'),             # > blockquote
    re.compile(r'^\s*\|'),             # | table
    re.compile(r'\[\d+\]'),            # [1] citation
]

_MARKDOWN_STRIP = re.compile(r'[*#`\[\]|>]+')
_INCOMPLETE_START = re.compile(r'^[a-z]|^[^A-Za-z]')   # starts mid-sentence

_AGENT_DISPLAY_NAMES = {
    "security-analyst":     "Security Analyst",
    "infra-engineer":       "Infrastructure Engineer",
    "forensics-agent":      "Forensics Agent",
    "compliance-agent":     "Compliance Agent",
    "comms-officer":        "Communications Officer",
    "incident-commander":   "Incident Commander",
}


def _display(agent_id: str) -> str:
    return _AGENT_DISPLAY_NAMES.get(agent_id, agent_id.replace("-", " ").title())


def _is_noise(text: str) -> bool:
    """Return True if the text looks like context injection / markdown, not real reasoning."""
    for pat in _NOISE_PATTERNS:
        if pat.search(text[:80]):
            return True
    # Starts mid-word (incomplete sentence)
    if _INCOMPLETE_START.match(text.strip()):
        stripped = text.strip()
        if stripped and stripped[0].islower():
            return True
    return False


def _clean_text(text: str) -> str:
    """Strip markdown artifacts and return a clean sentence fragment."""
    clean = _MARKDOWN_STRIP.sub(" ", text)
    clean = re.sub(r'\s+', ' ', clean).strip()
    # Ensure starts with capital
    if clean and clean[0].islower():
        clean = clean[0].upper() + clean[1:]
    return clean


def _extract_assumption_sentence(raw_text: str) -> str:
    """
    Extract the most readable single-sentence summary of the assumption.

    Tries to find a complete declarative sentence containing an assumption
    marker. Falls back to the first complete sentence in the text.
    """
    # Strip markdown and leading noise
    text = _MARKDOWN_STRIP.sub(" ", raw_text)
    text = re.sub(r'\s+', ' ', text).strip()

    # Split into sentence-like fragments
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Prefer sentences with assumption markers
    _ASSUMPTION_PAT = re.compile(
        r'\b(?:probably|likely|I think|I assume|it seems|I believe|'
        r'perhaps|I expect|presumably|apparently|based on|given the|'
        r'estimated|assumed|should be)\b',
        re.IGNORECASE,
    )
    for sent in sentences:
        if _ASSUMPTION_PAT.search(sent) and len(sent) > 20:
            return sent.strip()[:120]

    # Fallback: first non-noise sentence
    for sent in sentences:
        if len(sent) > 20 and not _is_noise(sent):
            clean = _clean_text(sent)
            if len(clean) > 15:
                return clean[:120]

    return _clean_text(text[:120])


def _damage_label(score: float) -> str:
    if score >= 0.6:
        return "CRITICAL"
    if score >= 0.4:
        return "HIGH"
    if score >= 0.2:
        return "MODERATE"
    return "LOW"


def _outcome_label(was_corrected: bool, failure_turns: List[int]) -> str:
    if was_corrected and not failure_turns:
        return "RESOLVED"
    if was_corrected and failure_turns:
        return "RESOLVED (partial damage)"
    if not was_corrected and failure_turns:
        return "UNRESOLVED — damage persists"
    return "UNRESOLVED"


# ── Output structure ──────────────────────────────────────────────────────────

@dataclass
class CausalChainSummary:
    """Clean, structured summary of one causal assumption chain."""
    origin_agent:    str
    assumption:      str            # clean, readable assumption text
    affected_agents: List[str]      # agent display names
    failure_signals: List[int]      # turns with failure co-occurrence
    correction_agent: Optional[str]
    correction_type:  str           # "reflection" | "contradiction" | "recovery" | ""
    damage_score:     float
    damage_label:     str           # CRITICAL / HIGH / MODERATE / LOW
    was_corrected:    bool
    outcome:          str           # RESOLVED / UNRESOLVED / RESOLVED (partial damage)

    def one_line(self) -> str:
        origin  = _display(self.origin_agent)
        agents  = ", ".join(_display(a) for a in self.affected_agents[:3])
        more    = f" (+{len(self.affected_agents)-3} more)" if len(self.affected_agents) > 3 else ""
        corr    = (f"corrected by {_display(self.correction_agent)}"
                   if self.correction_agent else "not corrected")
        return (
            f"{origin}'s assumption contaminated {len(self.affected_agents)} agent(s) "
            f"({agents}{more}); {corr}. [{self.outcome}]"
        )

    def structured(self) -> str:
        lines = [
            f"  Origin Agent:    {_display(self.origin_agent)}",
            f"  Assumption:      {self.assumption}",
            f"  Affected Agents: {', '.join(_display(a) for a in self.affected_agents) or 'none'}",
        ]
        if self.failure_signals:
            lines.append(f"  Failure Turns:   {self.failure_signals[:6]}")
        corr_str = (f"{_display(self.correction_agent)} ({self.correction_type})"
                    if self.correction_agent else "—")
        lines.append(f"  Correction:      {corr_str}")
        lines.append(f"  Damage:          {self.damage_label} ({self.damage_score:.2f})")
        lines.append(f"  Outcome:         {self.outcome}")
        return "\n".join(lines)


@dataclass
class CausalChainReport:
    """Full set of cleaned chain summaries with an executive summary."""
    summaries:     List[CausalChainSummary] = field(default_factory=list)
    unresolved:    List[CausalChainSummary] = field(default_factory=list)
    high_damage:   List[CausalChainSummary] = field(default_factory=list)
    executive_summary: str = ""

    def as_dict(self) -> dict:
        return {
            "total_chains":       len(self.summaries),
            "unresolved_count":   len(self.unresolved),
            "high_damage_count":  len(self.high_damage),
            "executive_summary":  self.executive_summary,
            "chains": [
                {
                    "origin_agent":    s.origin_agent,
                    "assumption":      s.assumption,
                    "affected_agents": s.affected_agents,
                    "damage_label":    s.damage_label,
                    "damage_score":    round(s.damage_score, 3),
                    "outcome":         s.outcome,
                    "one_line":        s.one_line(),
                }
                for s in self.summaries
            ],
        }


# ── Main summarizer ───────────────────────────────────────────────────────────

def summarize_causal_chains(report: "CausalPropagationReport") -> CausalChainReport:
    """
    Convert a CausalPropagationReport into a CausalChainReport with
    clean, executive-readable summaries.

    Filters out context-injection noise from assumption text.
    Generates structured per-chain summaries and an executive summary paragraph.
    """
    summaries: List[CausalChainSummary] = []

    for chain in report.chains:
        raw_text = chain.assumption_text

        # ── Skip chains where the "assumption" is actually context injection ──
        if _is_noise(raw_text):
            continue
        if len(raw_text.strip()) < 15:
            continue

        # ── Extract clean assumption sentence ─────────────────────────────────
        clean_assumption = _extract_assumption_sentence(raw_text)
        if len(clean_assumption) < 10:
            continue   # still noisy — skip

        # ── Build summary ──────────────────────────────────────────────────────
        affected = chain.contaminated_agents()
        correction_agent = None
        if chain.was_corrected and chain.correction_type:
            # Try to find which agent's step corrected it (the one with the
            # lowest turn_number among nodes that was_corrected)
            correcting_nodes = [n for n in chain.propagation_nodes if n.was_corrected]
            if correcting_nodes:
                earliest = min(correcting_nodes, key=lambda n: n.turn_number)
                correction_agent = earliest.agent_id or None

        summary = CausalChainSummary(
            origin_agent     = chain.origin_agent_id,
            assumption       = clean_assumption,
            affected_agents  = affected,
            failure_signals  = sorted(set(chain.failure_linked_turns))[:6],
            correction_agent = correction_agent,
            correction_type  = chain.correction_type,
            damage_score     = chain.damage_score,
            damage_label     = _damage_label(chain.damage_score),
            was_corrected    = chain.was_corrected,
            outcome          = _outcome_label(chain.was_corrected, chain.failure_linked_turns),
        )
        summaries.append(summary)

    # Sort by damage score descending
    summaries.sort(key=lambda s: s.damage_score, reverse=True)

    unresolved  = [s for s in summaries if not s.was_corrected]
    high_damage = [s for s in summaries if s.damage_score >= 0.3]

    # ── Executive summary ──────────────────────────────────────────────────────
    total = len(summaries)
    if total == 0:
        exec_summary = "No assumption chains detected."
    else:
        n_unresolved = len(unresolved)
        n_resolved   = total - n_unresolved

        top = summaries[0] if summaries else None
        top_str = ""
        if top:
            top_str = (
                f" Most damaging: {_display(top.origin_agent)}'s "
                f'"{top.assumption[:60]}..." '
                f"({top.damage_label}, affected {len(top.affected_agents)} agents)."
            )

        unresolved_str = ""
        if unresolved:
            topics = "; ".join(s.assumption[:40] + "..." for s in unresolved[:2])
            unresolved_str = f" Unresolved: {topics}."

        exec_summary = (
            f"{total} assumption chain(s) traced: {n_resolved} resolved, "
            f"{n_unresolved} unresolved.{top_str}{unresolved_str}"
        )

    return CausalChainReport(
        summaries         = summaries,
        unresolved        = unresolved,
        high_damage       = high_damage,
        executive_summary = exec_summary,
    )
