"""
Root Cause Ranking

Multi-factor scoring to rank root causes by their actual significance.
Not all contradictions matter equally.

Ranking factors
---------------
  propagation_radius  — how many turns the assumption spread (0-1)
  n_affected_agents   — how many unique agents were contaminated (0-1)
  damage_score        — from causal propagation (0-1)
  correction_difficulty — unresolved > partial > corrected (0-1)
  contradiction_severity — how many contradiction events link to this domain (0-1)
  confidence          — how certain the attribution is (0-1)

Output
------
  RankedRootCause with rank (1-based), composite_score, and why_ranked string.
  Sorted descending by composite_score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from harpo.forensics.root_cause_engine import RootCause

_AGENT_DISPLAY = {
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
class RankedRootCause:
    rank:            int
    root_cause:      "RootCause"
    composite_score: float
    damage_label:    str   # "CRITICAL" | "HIGH" | "MEDIUM" | "LOW"
    why_ranked:      str   # one sentence explaining the rank

    def render(self) -> str:
        rc  = self.root_cause
        cor = f"Corrected by {_dn(rc.corrected_by)}" if rc.corrected_by else "NOT corrected"
        agents = ", ".join(_dn(a) for a in rc.affected_agents[:4])
        more   = f" (+{len(rc.affected_agents)-4})" if len(rc.affected_agents) > 4 else ""
        return (
            f"  #{self.rank}  {self.damage_label}: {rc.title}\n"
            f"      Origin:   {_dn(rc.origin_agent)}\n"
            f"      Affected: {agents}{more}\n"
            f"      Status:   {rc.resolution} — {cor}\n"
            f"      Score:    {self.composite_score:.2f} | {self.why_ranked}"
        )


def _damage_label(score: float) -> str:
    if score >= 0.55:
        return "CRITICAL"
    if score >= 0.40:
        return "HIGH"
    if score >= 0.25:
        return "MEDIUM"
    return "LOW"


def _correction_difficulty(resolution: str) -> float:
    return {"UNRESOLVED": 1.0, "PARTIAL": 0.6, "CORRECTED": 0.2}.get(resolution, 0.5)


def _why_sentence(rc: "RootCause", score: float) -> str:
    parts = []
    if rc.damage_score >= 0.4:
        parts.append(f"high damage score ({rc.damage_score:.2f})")
    if len(rc.affected_agents) >= 3:
        parts.append(f"spread to {len(rc.affected_agents)} agents")
    if rc.resolution == "UNRESOLVED":
        parts.append("unresolved at trajectory end")
    if rc.confidence >= 0.85:
        parts.append(f"high-confidence attribution ({rc.confidence:.2f})")
    if not parts:
        parts.append(f"composite score {score:.2f}")
    return "Ranked here because: " + ", ".join(parts) + "."


def rank_root_causes(root_causes: List["RootCause"]) -> List[RankedRootCause]:
    """
    Rank root causes by composite score.

    Weights:
      damage_score           0.35
      correction_difficulty  0.25
      n_affected_agents      0.20
      confidence             0.20
    """
    if not root_causes:
        return []

    max_affected = max(len(rc.affected_agents) for rc in root_causes) or 1

    scored = []
    for rc in root_causes:
        affected_norm    = len(rc.affected_agents) / max_affected
        corr_difficulty  = _correction_difficulty(rc.resolution)
        composite = (
            rc.damage_score    * 0.35
            + corr_difficulty  * 0.25
            + affected_norm    * 0.20
            + rc.confidence    * 0.20
        )
        scored.append((composite, rc))

    scored.sort(key=lambda x: x[0], reverse=True)

    ranked = []
    for i, (score, rc) in enumerate(scored, start=1):
        label = _damage_label(score)
        ranked.append(RankedRootCause(
            rank            = i,
            root_cause      = rc,
            composite_score = round(score, 3),
            damage_label    = label,
            why_ranked      = _why_sentence(rc, score),
        ))
    return ranked
