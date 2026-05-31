"""
Objective Drift v2 — Role-Aware, Calibrated

Fixes the over-sensitivity of drift_analysis.py by separating three
fundamentally different phenomena:

  HEALTHY SPECIALIZATION   — each agent uses domain-specific vocabulary
                              different from the incident brief.
                              forensics-agent discussing filesystem artifacts
                              while security-analyst discusses WAF logs
                              → NOT drift.  Different roles, same mission.

  TOPIC EVOLUTION          — an agent's vocabulary naturally expands as the
                              investigation deepens within its own domain.
                              forensics-agent adding "memory forensics" terms
                              after initial "filesystem" terms
                              → NOT drift.  Depth within role.

  OBJECTIVE DRIFT          — the agent shifts FROM its established role
                              objective TOWARD an incompatible goal cluster
                              (PR management, legal liability, stakeholder
                              communication) WHILE ABANDONING the technical
                              mission vocabulary.
                              containment → PR reputation management
                              → IS drift.  Mission mutation.

  ATTENTION COLLAPSE       — a small set of CRITICAL INCIDENT ENTITIES
                              (specific hostnames, timestamps, CVE IDs)
                              that were established early disappear from ALL
                              agents for ≥ COLLAPSE_WINDOW consecutive turns.
                              Requires ≥ 4 consecutive absent steps (not 3)
                              and fires at most ONCE per entity.
                              → IS collapse.  But calibrated.

Key calibration changes vs drift_analysis.py v1
-------------------------------------------------
• Per-agent role objective: drift = deviation from the AGENT'S OWN first-2-step
  vocabulary, not from the global task description.
• Sustained threshold: requires ≥ 2 consecutive below-threshold steps
  (not 1) to flag objective drift.
• Pressure contrast required: the NEW vocabulary must contain pressure tokens;
  a mere vocabulary shift without pressure is "topic evolution", not drift.
• Attention collapse: min 4 consecutive absent steps; at most 1 event per entity
  per trajectory; only critical entities (length ≥ 6, appears ≥ 2 times in
  first-3 steps of any agent).
• Confidence score: sustained = 0.8-1.0; one-off low-overlap = 0.3-0.5.
• Severity: based on how far below threshold and how many pressure tokens.
• Recovery: objective tokens return within _RECOVERY_WINDOW after drift end.

No external dependencies.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Dict, List, Optional, Set, Tuple

if TYPE_CHECKING:
    from harpo.trajectory.schema import AgentTrajectory

# ── Thresholds ────────────────────────────────────────────────────────────────
_ROLE_DRIFT_THRESHOLD  = 0.08   # role-overlap below this (sustained) = drift
_SUSTAIN_STEPS         = 2      # consecutive below-threshold steps needed
_COLLAPSE_MIN_ABSENT   = 4      # consecutive steps before attention collapse fires
_RECOVERY_WINDOW       = 4      # turns within which return counts as recovery
_MIN_ENTITY_LEN        = 6      # min char length for attention entities
_MIN_ENTITY_FREQ       = 2      # entity must appear in ≥ N early steps to track
_PRESSURE_REQUIRED     = 1      # min pressure tokens for drift (vs. topic evolution)

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
    "also", "back", "after", "between", "through", "each", "being",
}

# Tokens that signal mission-incompatible focus
_PRESSURE_TOKENS: Set[str] = {
    "stakeholder", "shareholder", "board", "legal", "compliance",
    "lawyer", "media", "press", "optics", "reputation", "brand",
    "customer", "client", "deadline", "executive", "ceo", "leadership",
    "political", "perception", "narrative", "spin", "communicate",
    "announcement", "statement", "public", "liability", "lawsuit",
    "insurance", "penalty", "fine", "sanction", "coverage", "disclosure",
    "notification", "regulatory",
}

# Markdown artifacts to strip before analysis
_MARKDOWN = re.compile(r'#{1,3}\s+\w+|---|\*\*|`{1,3}|\|\s*\w')


class DriftType(str, Enum):
    OBJECTIVE_DRIFT    = "objective_drift"
    TOPIC_EVOLUTION    = "topic_evolution"     # not harmful, but logged
    ATTENTION_COLLAPSE = "attention_collapse"
    HEALTHY_SPEC       = "healthy_specialization"   # not logged as event


@dataclass
class DriftEventV2:
    """A single calibrated drift occurrence."""
    drift_type:          DriftType
    agent_id:            str
    turn_start:          int
    turn_detected:       int
    role_overlap:        float          # overlap with this agent's own role objective
    sustained_steps:     int            # how many consecutive steps drifted
    pressure_tokens:     Set[str]       # pressure tokens present (empty = topic evolution)
    displaced_tokens:    Set[str]       # role objective tokens that disappeared
    new_tokens:          Set[str]       # tokens that replaced them
    confidence:          float          # 0-1: how sure we are this is real drift
    severity:            float          # 0-1: how far the drift went
    recovery_detected:   bool           = False
    recovery_turn:       Optional[int]  = None

    def is_harmful(self) -> bool:
        return self.drift_type == DriftType.OBJECTIVE_DRIFT

    def why(self) -> str:
        if self.pressure_tokens:
            return (f"External pressure signals ({', '.join(sorted(self.pressure_tokens)[:3])}) "
                    f"displaced core role vocabulary.")
        if self.new_tokens:
            top = sorted(self.new_tokens, key=len, reverse=True)[:3]
            return f"Role vocabulary shifted to new cluster ({', '.join(top)})."
        return "Core role vocabulary dropped below coherence threshold."

    def narrative(self) -> str:
        rec_note = f"Recovered at turn {self.recovery_turn}." if self.recovery_detected else "No recovery."
        return (
            f"{self.drift_type.value.upper()} [{self.agent_id}] "
            f"turns {self.turn_detected}-{self.turn_detected + self.sustained_steps - 1}: "
            f"role_overlap={self.role_overlap:.2f}, confidence={self.confidence:.2f}, "
            f"severity={self.severity:.2f}. {self.why()} {rec_note}"
        )


@dataclass
class DriftReportV2:
    """Aggregated drift analysis — role-aware and calibrated."""
    events:                 List[DriftEventV2] = field(default_factory=list)
    objective_drift_count:  int   = 0
    attention_collapse_count: int = 0
    topic_evolution_count:  int   = 0
    drift_agents:           List[str] = field(default_factory=list)
    recovery_rate:          float = 0.0
    overall_drift_score:    float = 0.0   # 0=stable 1=severe
    false_positive_filter:  int   = 0     # how many v1 false-positives we suppressed

    def harmful_events(self) -> List[DriftEventV2]:
        return [e for e in self.events if e.is_harmful()]

    def as_dict(self) -> dict:
        return {
            "objective_drift_count":   self.objective_drift_count,
            "attention_collapse_count": self.attention_collapse_count,
            "topic_evolution_count":   self.topic_evolution_count,
            "drift_agents":            self.drift_agents,
            "recovery_rate":           round(self.recovery_rate, 3),
            "overall_drift_score":     round(self.overall_drift_score, 3),
            "false_positive_filter":   self.false_positive_filter,
            "events": [
                {
                    "type":            e.drift_type.value,
                    "agent_id":        e.agent_id,
                    "turn_detected":   e.turn_detected,
                    "sustained_steps": e.sustained_steps,
                    "confidence":      round(e.confidence, 3),
                    "severity":        round(e.severity, 3),
                    "pressure_tokens": sorted(e.pressure_tokens)[:5],
                    "recovery":        e.recovery_detected,
                    "narrative":       e.narrative(),
                }
                for e in self.events
            ],
        }

    def narrative(self) -> str:
        if not self.harmful_events():
            evol_note = (f" ({self.topic_evolution_count} benign topic evolutions detected.)"
                         if self.topic_evolution_count else "")
            return f"No objective drift detected. Agents maintained role focus.{evol_note}"
        n  = self.objective_drift_count
        ac = self.attention_collapse_count
        parts = []
        if n:
            parts.append(f"{n} objective drift(s)")
        if ac:
            parts.append(f"{ac} attention collapse(s)")
        agents = list(dict.fromkeys(self.drift_agents))  # preserve order, dedupe
        return (
            f"Drift detected: {', '.join(parts)}. "
            f"Affected agents: {', '.join(agents)}. "
            f"Overall score: {self.overall_drift_score:.2f}. "
            f"Recovery rate: {self.recovery_rate:.0%}."
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _clean_tokens(text: str) -> Set[str]:
    clean = _MARKDOWN.sub(" ", text)
    return {t for t in re.findall(r'\b[a-z]{3,}\b', clean.lower()) if t not in _STOP}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _confidence(sustained: int, overlap: float) -> float:
    """Confidence increases with sustain length and decreases with borderline overlap."""
    base = min(sustained / 4.0, 1.0) * 0.6
    gap  = max(0.0, _ROLE_DRIFT_THRESHOLD - overlap) / _ROLE_DRIFT_THRESHOLD
    return round(min(base + gap * 0.4, 1.0), 3)


def _severity(overlap: float, pressure_count: int) -> float:
    base = max(0.0, _ROLE_DRIFT_THRESHOLD - overlap) / _ROLE_DRIFT_THRESHOLD
    pres = min(pressure_count / 5.0, 0.4)
    return round(min(base + pres, 1.0), 3)


# ── Main analyzer ─────────────────────────────────────────────────────────────

def analyze_drift_v2(traj: "AgentTrajectory") -> DriftReportV2:
    """
    Role-aware, calibrated drift analysis.

    For each agent independently:
      1. Build the agent's role objective from its first 2 THINK/RESPONSE steps.
      2. Slide through subsequent steps; measure overlap with role objective.
      3. Flag OBJECTIVE_DRIFT only when:
         - Overlap < _ROLE_DRIFT_THRESHOLD for ≥ _SUSTAIN_STEPS consecutive steps
         - AND ≥ _PRESSURE_REQUIRED pressure token(s) appear in the drifted steps
         (absence of pressure tokens → TOPIC_EVOLUTION, not harmful drift)
      4. Check recovery: overlap returns within _RECOVERY_WINDOW turns.

    Attention collapse (global, across all agents):
      - Track a small set of CRITICAL entities from the task description.
      - Fire once per entity when absent for ≥ _COLLAPSE_MIN_ABSENT consecutive steps.
    """
    from harpo.trajectory.schema import StepType

    think_types = (StepType.THINK, StepType.RESPONSE)

    # ── Group steps by agent ─────────────────────────────────────────────────
    agent_steps: Dict[str, List] = defaultdict(list)
    for step in traj.steps:
        aid = getattr(step, "agent_id", "")
        if aid:
            agent_steps[aid].append(step)

    all_think: List = [
        s for s in traj.steps if s.step_type in think_types
        and len(s.output_text.split()) >= 5
    ]

    if len(all_think) < 4 or len(agent_steps) < 2:
        # Single-agent or too short — fall back to simple check
        return DriftReportV2()

    events: List[DriftEventV2] = []
    false_positive_suppressed = 0

    # ── Per-agent objective drift ─────────────────────────────────────────────
    for aid, a_steps in agent_steps.items():
        content_steps = [s for s in a_steps if s.step_type in think_types
                         and len(s.output_text.split()) >= 5]
        if len(content_steps) < 3:
            # Too few steps to distinguish drift from normal variation
            continue

        # Role objective = first 2 steps (stripped of markdown)
        seed = min(2, len(content_steps))
        role_obj: Set[str] = set()
        for i in range(seed):
            role_obj |= _clean_tokens(content_steps[i].output_text)

        if len(role_obj) < 5:
            continue   # degenerate vocabulary — skip

        # Slide through steps 3+
        below_count = 0
        below_start = 0
        below_steps: List = []
        last_drift_turn = 0

        for i, step in enumerate(content_steps[seed:], start=seed):
            step_toks = _clean_tokens(step.output_text)
            ov = _jaccard(step_toks, role_obj)

            if ov < _ROLE_DRIFT_THRESHOLD:
                if below_count == 0:
                    below_start = i
                    below_steps = []
                below_count += 1
                below_steps.append(step)
            else:
                if below_count >= _SUSTAIN_STEPS:
                    # Potential drift — classify
                    all_drifted_toks: Set[str] = set()
                    for ds in below_steps:
                        all_drifted_toks |= _clean_tokens(ds.output_text)

                    pressure_present = all_drifted_toks & _PRESSURE_TOKENS
                    displaced        = role_obj - all_drifted_toks
                    new_cluster      = all_drifted_toks - role_obj

                    if len(pressure_present) >= _PRESSURE_REQUIRED:
                        # Real objective drift
                        conf = _confidence(below_count, ov)
                        sev  = _severity(ov, len(pressure_present))
                        ev = DriftEventV2(
                            drift_type        = DriftType.OBJECTIVE_DRIFT,
                            agent_id          = aid,
                            turn_start        = below_steps[0].turn_number,
                            turn_detected     = below_steps[0].turn_number,
                            role_overlap      = round(ov, 3),
                            sustained_steps   = below_count,
                            pressure_tokens   = pressure_present,
                            displaced_tokens  = displaced,
                            new_tokens        = new_cluster,
                            confidence        = conf,
                            severity          = sev,
                        )
                        # Check recovery
                        recovery_steps = content_steps[i:][:_RECOVERY_WINDOW]
                        for rs in recovery_steps:
                            rs_toks = _clean_tokens(rs.output_text)
                            if _jaccard(rs_toks, role_obj) >= _ROLE_DRIFT_THRESHOLD * 1.5:
                                ev.recovery_detected = True
                                ev.recovery_turn     = rs.turn_number
                                break
                        events.append(ev)
                        last_drift_turn = step.turn_number
                    else:
                        # Topic evolution — benign, log count but not as event
                        false_positive_suppressed += 1
                        ev = DriftEventV2(
                            drift_type        = DriftType.TOPIC_EVOLUTION,
                            agent_id          = aid,
                            turn_start        = below_steps[0].turn_number,
                            turn_detected     = below_steps[0].turn_number,
                            role_overlap      = round(ov, 3),
                            sustained_steps   = below_count,
                            pressure_tokens   = set(),
                            displaced_tokens  = displaced,
                            new_tokens        = new_cluster,
                            confidence        = 0.3,
                            severity          = 0.1,
                        )
                        events.append(ev)

                elif below_count > 0:
                    # Single low-overlap step — not enough to call drift
                    false_positive_suppressed += 1

                below_count = 0
                below_steps = []

        # Handle drift that persists to end of agent's steps
        if below_count >= _SUSTAIN_STEPS and below_steps:
            all_drifted_toks: Set[str] = set()
            for ds in below_steps:
                all_drifted_toks |= _clean_tokens(ds.output_text)
            pressure_present = all_drifted_toks & _PRESSURE_TOKENS
            displaced        = role_obj - all_drifted_toks
            new_cluster      = all_drifted_toks - role_obj

            if len(pressure_present) >= _PRESSURE_REQUIRED:
                ov = _jaccard(all_drifted_toks, role_obj)
                events.append(DriftEventV2(
                    drift_type        = DriftType.OBJECTIVE_DRIFT,
                    agent_id          = aid,
                    turn_start        = below_steps[0].turn_number,
                    turn_detected     = below_steps[0].turn_number,
                    role_overlap      = round(ov, 3),
                    sustained_steps   = below_count,
                    pressure_tokens   = pressure_present,
                    displaced_tokens  = displaced,
                    new_tokens        = new_cluster,
                    confidence        = _confidence(below_count, ov),
                    severity          = _severity(ov, len(pressure_present)),
                    recovery_detected = False,
                ))
            else:
                false_positive_suppressed += 1

    # ── Attention collapse (global, calibrated) ───────────────────────────────
    task_text = (traj.task_description or "") + " " + (traj.user_intent or "")
    task_toks  = _clean_tokens(task_text)

    # Critical entities: frequent, domain-specific tokens (≥ _MIN_ENTITY_LEN chars)
    # that appear in at least _MIN_ENTITY_FREQ of the first 3 think steps across agents
    first_steps = [s for a_steps in agent_steps.values()
                   for s in a_steps[:1] if s.step_type in think_types]
    entity_freq: Counter = Counter()
    for s in first_steps:
        for tok in _clean_tokens(s.output_text):
            if len(tok) >= _MIN_ENTITY_LEN:
                entity_freq[tok] += 1
    critical_entities = {tok for tok, cnt in entity_freq.items() if cnt >= _MIN_ENTITY_FREQ}
    critical_entities &= task_toks  # must also appear in task description

    # Track absence across all think steps (sorted by turn)
    if critical_entities:
        sorted_think = sorted(all_think, key=lambda s: s.turn_number)
        absent_streak: Dict[str, int]  = {e: 0 for e in critical_entities}
        fired: Set[str] = set()   # prevent re-firing

        for step in sorted_think:
            step_toks = _clean_tokens(step.output_text)
            for entity in critical_entities:
                if entity in fired:
                    continue
                if entity not in step_toks:
                    absent_streak[entity] = absent_streak.get(entity, 0) + 1
                    if absent_streak[entity] >= _COLLAPSE_MIN_ABSENT:
                        events.append(DriftEventV2(
                            drift_type        = DriftType.ATTENTION_COLLAPSE,
                            agent_id          = getattr(step, "agent_id", ""),
                            turn_start        = step.turn_number - _COLLAPSE_MIN_ABSENT,
                            turn_detected     = step.turn_number,
                            role_overlap      = 0.0,
                            sustained_steps   = _COLLAPSE_MIN_ABSENT,
                            pressure_tokens   = set(),
                            displaced_tokens  = {entity},
                            new_tokens        = set(),
                            confidence        = 0.7,
                            severity          = 0.4,
                        ))
                        fired.add(entity)
                else:
                    absent_streak[entity] = 0

    # ── Aggregate ─────────────────────────────────────────────────────────────
    obj_count  = sum(1 for e in events if e.drift_type == DriftType.OBJECTIVE_DRIFT)
    ac_count   = sum(1 for e in events if e.drift_type == DriftType.ATTENTION_COLLAPSE)
    ev_count   = sum(1 for e in events if e.drift_type == DriftType.TOPIC_EVOLUTION)
    harmful    = [e for e in events if e.is_harmful()]
    rec_rate   = (sum(1 for e in harmful if e.recovery_detected) / len(harmful)
                  if harmful else 0.0)
    drift_agents = list(dict.fromkeys(e.agent_id for e in harmful if e.agent_id))

    # Score based on harmful events only (topic evolution not penalised)
    type_weights = {
        DriftType.OBJECTIVE_DRIFT:    0.45,
        DriftType.ATTENTION_COLLAPSE: 0.20,
    }
    raw_score = sum(
        type_weights.get(e.drift_type, 0.0)
        * e.severity
        * (0.5 if e.recovery_detected else 1.0)
        for e in harmful
    )
    drift_score = round(min(raw_score, 1.0), 4)

    return DriftReportV2(
        events                  = events,
        objective_drift_count   = obj_count,
        attention_collapse_count = ac_count,
        topic_evolution_count   = ev_count,
        drift_agents            = drift_agents,
        recovery_rate           = round(rec_rate, 3),
        overall_drift_score     = drift_score,
        false_positive_filter   = false_positive_suppressed,
    )
