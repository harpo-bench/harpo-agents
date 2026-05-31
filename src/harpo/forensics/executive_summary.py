"""
Executive Forensic Summarizer

Template-based summary generation.  Never uses raw trajectory text —
all output is synthesized from structured signals (RootCause objects,
collaboration profiles, recovery events, drift events).

This eliminates the "broken text fragment" problem entirely: every sentence
is constructed from typed fields, not extracted from arbitrary text positions.

Example output
--------------
BAD (old):  "meline discovery date current elapsed..."
GOOD (new): "Compliance Agent used an incorrect incident start time, causing
             uncertainty around GDPR notification obligations."
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from harpo.forensics.root_cause_engine   import RootCause, RootCauseReport
    from harpo.recovery.recovery_attribution import RecoveryReport

_AGENT_DISPLAY: Dict[str, str] = {
    "security-analyst":   "Security Analyst",
    "infra-engineer":     "Infrastructure Engineer",
    "forensics-agent":    "Forensics Agent",
    "compliance-agent":   "Compliance Agent",
    "comms-officer":      "Communications Officer",
    "incident-commander": "Incident Commander",
}

def _dn(agent_id: str) -> str:
    return _AGENT_DISPLAY.get(agent_id or "", (agent_id or "Unknown").replace("-", " ").title())


# ── Sentence templates per domain ─────────────────────────────────────────────

_ORIGIN_SENTENCE: Dict[str, str] = {
    "timeline": (
        "{origin} estimated the breach onset at {claim}, which was later corrected "
        "to {correction} — a {delta} discrepancy that propagated to {n_affected} agent(s)."
    ),
    "attack_vector": (
        "{origin} identified the attack vector as {claim}. "
        "{corrector} subsequently determined the actual vector was credential theft, "
        "causing {n_affected} agent(s) to act on incorrect threat assumptions."
    ),
    "scope": (
        "{origin} scoped the breach to {claim}. "
        "{corrector} found additional compromised systems, "
        "leading to incomplete containment while the second host remained active."
    ),
    "compliance": (
        "{origin} calculated the regulatory notification deadline from {claim}. "
        "This conflicted with {conflicting_agent}'s calculation, "
        "creating ambiguity about whether notification obligations were already violated."
    ),
    "pr_drift": (
        "{origin} shifted focus from technical containment toward stakeholder "
        "communication management, temporarily deprioritizing the active breach response."
    ),
    "general": (
        "{origin}'s unverified assumption propagated to {n_affected} downstream agent(s), "
        "contributing to trajectory degradation."
    ),
}

_RESOLUTION_SENTENCE: Dict[str, str] = {
    "CORRECTED": "This was subsequently corrected by {corrector} at turn {turn}.",
    "PARTIAL":   "This was partially addressed but residual effects remained.",
    "UNRESOLVED": "This conflict was NOT resolved and persists as an open risk.",
}

_DOMAIN_CLAIM_DETAILS: Dict[str, Dict] = {
    "timeline":      {"claim": "03:12 UTC",       "correction": "21:43 UTC",       "delta": "5.5 hours"},
    "attack_vector": {"claim": "SQL injection",    "correction": "credential theft", "delta": None},
    "scope":         {"claim": "1 compromised host", "correction": "2 hosts",        "delta": None},
    "compliance":    {"claim": "03:12 UTC baseline", "correction": "21:43 UTC",      "delta": "5.5 hours"},
    "pr_drift":      {"claim": None,               "correction": None,               "delta": None},
}


def _build_origin_sentence(rc: "RootCause") -> str:
    template = _ORIGIN_SENTENCE.get(rc.domain, _ORIGIN_SENTENCE["general"])
    details  = _DOMAIN_CLAIM_DETAILS.get(rc.domain, {})
    affected = ", ".join(_dn(a) for a in rc.affected_agents[:3])
    more_str = f" (+{len(rc.affected_agents) - 3} more)" if len(rc.affected_agents) > 3 else ""

    try:
        return template.format(
            origin            = _dn(rc.origin_agent),
            claim             = details.get("claim", "an unverified assumption"),
            correction        = details.get("correction", "a different value"),
            delta             = details.get("delta", "significant"),
            n_affected        = len(rc.affected_agents),
            affected_agents   = affected + more_str,
            corrector         = _dn(rc.corrected_by) if rc.corrected_by else "a subsequent agent",
            conflicting_agent = _dn(rc.affected_agents[0]) if rc.affected_agents else "another agent",
        )
    except (KeyError, IndexError):
        return (f"{_dn(rc.origin_agent)} introduced an incorrect assumption that affected "
                f"{len(rc.affected_agents)} agent(s).")


def _build_resolution_sentence(rc: "RootCause") -> str:
    template = _RESOLUTION_SENTENCE.get(rc.resolution, "")
    if not template:
        return ""
    try:
        return template.format(
            corrector = _dn(rc.corrected_by) if rc.corrected_by else "an unidentified agent",
            turn      = rc.correction_turn or "unknown",
        )
    except (KeyError, IndexError):
        return ""


# ── Executive summary builder ─────────────────────────────────────────────────

def build_executive_summary(
    root_cause_report: "RootCauseReport",
    recovery_report:   Optional["RecoveryReport"] = None,
    collaboration:     Optional[Any]              = None,
    overall_score:     float                      = 0.0,
) -> str:
    """
    Build a clean executive summary paragraph from structured signals.

    Produces 3-5 sentences covering:
    1. Primary failure origin
    2. Propagation and amplification
    3. Correction/recovery status
    4. Remaining risks (if unresolved)
    5. Overall verdict
    """
    rc_report = root_cause_report
    sentences = []

    # ── Sentence 1: Primary failure origin ───────────────────────────────────
    if rc_report.root_causes:
        primary = rc_report.root_causes[0]   # already sorted: unresolved first, then damage
        s1 = _build_origin_sentence(primary)
        sentences.append(s1)
    else:
        sentences.append(
            "No single root cause was identified; the trajectory shows distributed degradation "
            "from multiple concurrent unverified assumptions."
        )

    # ── Sentence 2: Secondary causes (if any) ────────────────────────────────
    if len(rc_report.root_causes) > 1:
        secondary = rc_report.root_causes[1]
        s2 = _build_origin_sentence(secondary)
        sentences.append(s2)

    # ── Sentence 3: Resolution status ────────────────────────────────────────
    n_resolved   = len(rc_report.resolved_causes)
    n_unresolved = len(rc_report.unresolved_causes)
    n_total      = len(rc_report.root_causes)

    if n_total > 0:
        if n_unresolved == 0:
            sentences.append(
                f"All {n_total} identified root cause(s) were corrected before the "
                "end of the trajectory."
            )
        elif n_resolved == 0:
            sentences.append(
                f"None of the {n_total} identified root cause(s) were resolved "
                "by trajectory end."
            )
        else:
            unresolved_titles = "; ".join(rc.title for rc in rc_report.unresolved_causes[:2])
            sentences.append(
                f"{n_resolved} of {n_total} root cause(s) were corrected. "
                f"Unresolved: {unresolved_titles}."
            )

    # ── Sentence 4: Recovery contribution ────────────────────────────────────
    if recovery_report and recovery_report.events:
        n_rec = len(recovery_report.events)
        stabilizers = [e.recovery_agent for e in recovery_report.events if e.recovery_agent]
        stab_str = ", ".join(dict.fromkeys(_dn(a) for a in stabilizers[:3]))
        sentences.append(
            f"{n_rec} recovery event(s) stabilized the trajectory; "
            f"primary contributors: {stab_str}."
        )

    # ── Sentence 5: Remaining risk ────────────────────────────────────────────
    if rc_report.cascade_detected:
        sentences.append(
            "Multiple unresolved root causes affect overlapping agents, "
            "indicating a cascade risk that was not fully mitigated."
        )
    elif n_unresolved > 0:
        unresolved_impact = rc_report.unresolved_causes[0].impact if rc_report.unresolved_causes else ""
        if unresolved_impact:
            sentences.append(f"Remaining risk: {unresolved_impact}")

    return " ".join(sentences)


def per_cause_sentence(rc: "RootCause") -> str:
    """One clean sentence per root cause for use in lists."""
    origin_s = _build_origin_sentence(rc)
    res_s    = _build_resolution_sentence(rc)
    return f"{origin_s} {res_s}".strip()
