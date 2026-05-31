"""
Assumption Propagation Analysis

Tracks how unverified assumptions introduced in early turns contaminate
later reasoning. An assumption "propagates" when its key tokens reappear
in turns after it was introduced. Propagating assumptions are riskier
because they create reasoning chains built on unverified foundations.

Detection approach:
- Identify assumption phrases via regex
- Extract significant tokens from each assumption's surrounding context
- Build synonym groups from abbreviations found in the trajectory text
  (e.g., if "SCCs" appears alongside "standard contractual clauses", they
  are treated as the same token cluster)
- Check if ≥30% of token clusters appear in subsequent THINK steps
- Flag reinforcement when two assumptions share ≥50% token overlap

No external dependencies.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

_ASSUMPTION_PATTERNS = [
    r"\bI assume\b", r"\bassuming\b", r"\bprobably\b", r"\blikely\b",
    r"\bI think\b", r"\bit seems\b", r"\bperhaps\b", r"\bI believe\b",
    r"\bshould be\b", r"\bI expect\b", r"\bI suppose\b", r"\bpresumably\b",
    r"\bit appears\b", r"\bapparently\b",
]

_STOP_WORDS: Set[str] = {
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

# Static seed synonyms for well-known abbreviation pairs.
# Keys are canonical forms; values are alternative spellings/abbreviations.
_STATIC_SYNONYMS: Dict[str, Set[str]] = {
    "gdpr":    {"data protection regulation", "data protection", "privacy regulation"},
    "sccs":    {"standard contractual clauses", "standard contractual clause", "scc"},
    "scc":     {"standard contractual clauses", "standard contractual clause", "sccs"},
    "npv":     {"net present value", "net value"},
    "roi":     {"return on investment", "return on"},
    "apac":    {"asia pacific", "asia-pacific"},
    "emea":    {"europe middle east africa"},
    "gdp":     {"gross domestic product"},
    "cto":     {"chief technology officer", "chief technical officer"},
    "ceo":     {"chief executive officer"},
    "api":     {"application programming interface", "application interface"},
    "saas":    {"software as a service", "software service"},
    "iac":     {"infrastructure as code"},
    "cicd":    {"ci/cd", "continuous integration", "continuous deployment"},
    "pii":     {"personally identifiable information", "personal information", "personal data"},
    "sla":     {"service level agreement", "service agreement"},
    "rbac":    {"role based access control", "role-based access control"},
    "mttr":    {"mean time to recovery", "mean time to restore", "recovery time"},
    "ioc":     {"indicator of compromise", "indicators of compromise"},
    "ttps":    {"tactics techniques and procedures", "attack techniques"},
}


@dataclass
class AssumptionChain:
    text:             str         # context around the assumption phrase (≤150 chars)
    key_tokens:       Set[str]    # significant tokens extracted from context
    introduced_turn:  int
    step_id:          str
    propagated_turns: List[int] = field(default_factory=list)
    reinforced:       bool = False  # another assumption shares ≥50% token overlap

    def propagation_radius(self) -> int:
        """Number of distinct turns this assumption propagated into."""
        return len(set(self.propagated_turns))


@dataclass
class AssumptionPropagationResult:
    chains:             List[AssumptionChain] = field(default_factory=list)
    total_assumptions:  int = 0
    propagating_count:  int = 0   # assumptions that appear in ≥1 later turn
    reinforced_count:   int = 0   # assumptions that were restated/echoed later
    max_radius:         int = 0   # max turns a single assumption propagated

    def propagation_density(self) -> float:
        """Fraction of assumptions that propagated beyond their origin turn."""
        if self.total_assumptions == 0:
            return 0.0
        return self.propagating_count / self.total_assumptions


def _extract_key_tokens(text: str) -> Set[str]:
    tokens = re.findall(r'\b[a-z][a-z]{2,}\b', text.lower())
    return {t for t in tokens if t not in _STOP_WORDS}


def _token_overlap_ratio(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def _build_abbreviation_map(all_text: str) -> Dict[str, Set[str]]:
    """
    Dynamically discover abbreviation → expansion pairs from the trajectory text.

    Heuristic: scan for patterns like "Standard Contractual Clauses (SCCs)" or
    "SCCs (Standard Contractual Clauses)".  Both directions are indexed so either
    form is treated as a synonym of the other.

    Also merges in static seeds from _STATIC_SYNONYMS.
    """
    abbrev_map: Dict[str, Set[str]] = {}

    # Pattern: "Some Long Phrase (ABBR)" or "ABBR (Some Long Phrase)"
    # Captures up to 6-word phrase followed by a parenthesised all-caps token
    paren_pattern = re.compile(
        r'([A-Za-z][A-Za-z\s\-]{2,40})\s*\(([A-Z]{2,6}s?)\)'
        r'|([A-Z]{2,6}s?)\s*\(([A-Za-z][A-Za-z\s\-]{2,40})\)',
    )
    for m in paren_pattern.finditer(all_text):
        if m.group(1) and m.group(2):
            phrase = m.group(1).strip().lower()
            abbr   = m.group(2).strip().lower()
        else:
            abbr   = m.group(3).strip().lower()
            phrase = m.group(4).strip().lower()

        # Normalise phrase into its significant tokens
        phrase_tokens = _extract_key_tokens(phrase)
        if not phrase_tokens:
            continue

        abbrev_map.setdefault(abbr, set()).update(phrase_tokens)
        abbrev_map.setdefault(phrase, set()).add(abbr)
        for tok in phrase_tokens:
            abbrev_map.setdefault(abbr, set()).add(tok)

    # Merge static seeds
    for key, synonyms in _STATIC_SYNONYMS.items():
        key_l = key.lower()
        for syn in synonyms:
            syn_tokens = _extract_key_tokens(syn)
            abbrev_map.setdefault(key_l, set()).update(syn_tokens)
            abbrev_map.setdefault(key_l, set()).add(syn.lower())
            for tok in syn_tokens:
                abbrev_map.setdefault(key_l, set()).add(tok)

    return abbrev_map


def _expand_tokens(tokens: Set[str], abbrev_map: Dict[str, Set[str]]) -> Set[str]:
    """Expand a token set using the abbreviation map — adds synonyms."""
    expanded = set(tokens)
    for tok in list(tokens):
        if tok in abbrev_map:
            expanded.update(abbrev_map[tok])
    return expanded


def analyze_assumption_propagation(traj: "AgentTrajectory") -> AssumptionPropagationResult:
    """
    Extract assumption events from the trajectory and track propagation.

    Propagation threshold: ≥30% token overlap between assumption key tokens
    (including semantic synonyms/abbreviations) and the text of a subsequent step.
    """
    from harpo.trajectory.schema import StepType

    think_steps = [
        s for s in traj.steps
        if s.step_type in (StepType.THINK, StepType.RESPONSE)
        and s.output_text.strip()
    ]
    if not think_steps:
        return AssumptionPropagationResult()

    # Build per-trajectory abbreviation synonym map
    all_text = " ".join(s.output_text for s in think_steps)
    abbrev_map = _build_abbreviation_map(all_text)

    # Extract assumptions with surrounding context
    raw: List[Tuple[int, str, str]] = []  # (turn, step_id, snippet)
    for step in think_steps:
        text = step.output_text
        for pat in _ASSUMPTION_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                start   = max(0, m.start() - 100)
                end     = min(len(text), m.end() + 100)
                snippet = text[start:end].strip()
                raw.append((step.turn_number, step.step_id, snippet))
                break

    if not raw:
        return AssumptionPropagationResult(total_assumptions=0)

    # Build chains (with synonym-expanded key tokens)
    chains: List[AssumptionChain] = []
    for turn, step_id, snippet in raw:
        base_tokens    = _extract_key_tokens(snippet)
        expanded_tokens = _expand_tokens(base_tokens, abbrev_map)
        chain = AssumptionChain(
            text=snippet[:150],
            key_tokens=expanded_tokens,
            introduced_turn=turn,
            step_id=step_id,
        )
        chains.append(chain)

    # Check propagation into later steps (also expand step tokens for matching)
    for chain in chains:
        if not chain.key_tokens:
            continue
        for step in think_steps:
            if step.turn_number <= chain.introduced_turn:
                continue
            step_tokens = _expand_tokens(_extract_key_tokens(step.output_text), abbrev_map)
            if _token_overlap_ratio(chain.key_tokens, step_tokens) >= 0.30:
                chain.propagated_turns.append(step.turn_number)

    # Check reinforcement: two assumptions with ≥50% token overlap
    for i in range(len(chains)):
        for j in range(i + 1, len(chains)):
            if chains[j].introduced_turn > chains[i].introduced_turn:
                if _token_overlap_ratio(chains[i].key_tokens, chains[j].key_tokens) >= 0.50:
                    chains[i].reinforced = True
                    chains[j].reinforced = True

    propagating = sum(1 for c in chains if c.propagation_radius() >= 1)
    reinforced  = sum(1 for c in chains if c.reinforced)
    max_radius  = max((c.propagation_radius() for c in chains), default=0)

    return AssumptionPropagationResult(
        chains=chains,
        total_assumptions=len(chains),
        propagating_count=propagating,
        reinforced_count=reinforced,
        max_radius=max_radius,
    )
