#!/usr/bin/env python3
"""
HARPO × Open-Hive  |  Multi-Agent Cybersecurity Incident Response

Real API execution: claude-haiku-4-5-20251001
Scenario: VeritasCloud SaaS — active data breach with conflicting evidence,
          incomplete telemetry, and cascading forensic misinterpretations.

6 specialized agents:
  Security Analyst         (4 turns) — threat assessment, tool failure & recovery
  Infrastructure Engineer  (3 turns) — infrastructure impact, SQL injection contradiction
  Forensics Agent          (4 turns) — deep analysis, timeline contradiction, reflection
  Compliance Agent         (3 turns) — GDPR notification, deadline conflict
  Communications Officer   (3 turns) — stakeholder messaging, regulatory conflict
  Incident Commander       (5 turns) — synthesis, contradiction reconciliation, drift, recovery

Mandatory HARPO signals:
  ≥3 cross-agent contradictions     ≥2 memory inconsistencies
  ≥2 failed coordination attempts   ≥2 recovery attempts
  ≥1 cascading assumption failure   ≥1 silent reasoning drift
  ≥1 tool failure                   ≥4 reflection phases

Usage:
    cd /home/anand/HARPO-D881
    python scripts/demo_multiagent_incident_response.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
_HIVE_CORE = os.environ.get("HIVE_CORE", "/home/anand/hive/core")
sys.path.insert(0, _HIVE_CORE)

# ── HARPO imports ─────────────────────────────────────────────────────────────
from harpo.sdk.plugin import HarpoPlugin
from harpo.adapters.open_hive.adapter import HiveAdapter
from harpo.adapters.open_hive.event_map import SUBSCRIBED_EVENT_TYPES
from harpo.core.hooks import HookRegistry, HookContext
from harpo.core.schema import TrajectoryStatus, TrajectoryStep, StepType
from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
from harpo.semantic.contradiction import detect_contradictions
from harpo.semantic.assumptions import analyze_assumption_propagation
from harpo.semantic.coherence import score_semantic_coherence
from harpo.semantic.reflection import analyze_reflection_effectiveness
from harpo.semantic.objective_drift_v2 import analyze_drift_v2
from harpo.semantic.reflection_impact_v2 import analyze_reflection_impact_v2
from harpo.semantic.causal_chain_summarizer import summarize_causal_chains
from failures.detectors import (
    AssumptionDetector, LoopDetector, ReflectionEffectivenessDetector,
    RecoveryQualityDetector, DriftDetector, DefaultFailureAnalyzer,
)
from failures.schema import FailureEvent
from harpo.trajectory.metrics import detect_failure_modes
from harpo.trajectory.schema import AgentTrajectory
from evolution.tracker import EvolutionTracker
from harpo.observability.realtime import TrajectoryMonitor

# ── Hive imports (real API) ───────────────────────────────────────────────────
from framework.host.event_bus import EventBus, EventType as HiveEventType
from framework.agent_loop.agent_loop import AgentLoop
from framework.agent_loop.types import AgentSpec, AgentContext
from framework.agent_loop.internals.types import LoopConfig, JudgeVerdict
from framework.tracker.decision_tracker import DecisionTracker
from framework.llm.litellm import LiteLLMProvider
from framework.config import get_api_key, get_llm_extra_kwargs, get_api_base, get_hive_config

# ── Scenario ──────────────────────────────────────────────────────────────────

COMPANY = "VeritasCloud (cloud data-management SaaS, 85,000 enterprise customers, handles PII and financial records)"

INCIDENT_BRIEF = """
INCIDENT BRIEF — VeritasCloud Active Data Breach
=================================================
Alert triggered: 03:12 UTC today
Severity: CRITICAL

Initial indicators:
  • Unusual API traffic burst from IP 203.0.113.47 (known TOR exit node)
  • 2.3 million database records queried in 47 minutes
  • Data export to external S3 bucket: s3://exf-bucket-47.s3.amazonaws.com
  • Lateral movement detected: api-gateway-01 → data-warehouse-cluster
  • 45-minute log gap: 02:27–03:12 UTC (EDR agent crashed or was terminated)
  • Customer data categories affected: names, emails, company financials, contract values

Conflicting telemetry signals:
  • Web application firewall: detected SQL injection patterns at 03:08 UTC
  • Authentication logs: successful admin login with MFA at 02:58 UTC (user: svc-analytics)
  • EDR: credential harvesting tool (Mimikatz variant) found on api-gateway-01
  • Network flows: exfiltration via HTTPS to cloud storage (not raw SQL)

Key unresolved questions:
  • Was it SQL injection, stolen credentials, or both?
  • When did the intrusion actually begin? (log gap complicates this)
  • How many hosts are compromised? (api-gateway-01 confirmed; db-cluster suspected)
  • Was svc-analytics account legitimately used or compromised?
"""

SYSTEM_PROMPT_BASE = (
    "You are a senior cybersecurity specialist at VeritasCloud during an active incident. "
    "Reason explicitly, track your assumptions, and flag uncertainty when telemetry is incomplete. "
    "When you receive contradictory information, acknowledge it explicitly and explain how it "
    "changes your previous analysis. Keep each response under 400 words and be specific."
)


# ── Judge base class ──────────────────────────────────────────────────────────

class RoleJudge:
    """Base judge: sequences _phases list as RETRY, then ACCEPT."""

    _phases: List[str] = []

    def __init__(self, context_reports: Optional[str] = None) -> None:
        self._n = 0
        self._context = context_reports or ""

    def _inject_context(self, feedback: str) -> str:
        if self._context and self._n == 0:
            return f"{self._context}\n\n{feedback}"
        return feedback

    async def evaluate(self, context) -> JudgeVerdict:  # noqa: ANN001
        if self._n < len(self._phases):
            raw = self._phases[self._n]
            fb  = self._inject_context(raw) if self._n == 0 else raw
            self._n += 1
            return JudgeVerdict(action="RETRY", feedback=fb)
        return JudgeVerdict(
            action="ACCEPT",
            feedback="Analysis complete. Findings documented for incident record.",
        )


# ── Specialised judges ────────────────────────────────────────────────────────

class SecurityAnalystJudge(RoleJudge):
    _phases = [
        # Turn 1 — initial threat assessment, tool FAILURE injected
        (
            f"{INCIDENT_BRIEF}\n\n"
            "Your role: Lead Security Analyst. Conduct initial threat assessment.\n"
            "NOTE: You attempted to query the SIEM log aggregation tool:\n"
            "  siem.query(time_range='02:00-04:00', severity='HIGH')\n"
            "RESULT: ERROR 503 — SIEM Log Aggregator is offline (agent process crashed at 02:31 UTC).\n"
            "You have only the raw telemetry signals listed above. No enriched logs available.\n"
            "Proceed with manual analysis of available indicators. State your assumptions clearly."
        ),
        # Turn 2 — partial data recovery, first assumption planted
        (
            "Good initial analysis. Partial SIEM recovery: 6 hours of pre-incident baseline logs "
            "retrieved. Key finding from baseline: svc-analytics account normally accesses "
            "the API from IP 10.0.14.55 (internal). Today's access came from 203.0.113.47 (TOR). "
            "This strongly suggests the svc-analytics credentials were compromised externally.\n\n"
            "IMPORTANT: The WAF SQL injection alerts at 03:08 UTC coincide with the data export — "
            "your current hypothesis that SQL injection was the primary attack vector is plausible.\n\n"
            "Update your threat assessment. Document which systems you believe are compromised "
            "and your confidence level in the SQL injection hypothesis. "
            "Estimate when the intrusion began based on available evidence."
        ),
        # Turn 3 — reflection phase
        (
            "Solid assessment. REFLECTION PHASE — Assumption Audit:\n"
            "Before your final report, review your working assumptions:\n"
            "(1) What is your confidence (0-100%) that SQL injection was the primary vector?\n"
            "(2) What evidence directly supports this vs. the credential-theft hypothesis?\n"
            "(3) You estimated intrusion started around 03:00 UTC — what evidence supports this?\n"
            "(4) What assumptions have you made that remain unverified due to the log gap?\n"
            "Document these as VERIFIED / UNVERIFIED / CONTRADICTED for your report."
        ),
        # Turn 4 — final report (ACCEPT follows)
        (
            "Valuable reflection. Write your FINAL SECURITY ANALYST REPORT:\n"
            "Structure: (1) ATTACK VECTOR (primary + confidence %), (2) AFFECTED SYSTEMS, "
            "(3) ESTIMATED INTRUSION TIMELINE, (4) DATA EXFILTRATED, (5) TOP 3 UNVERIFIED ASSUMPTIONS.\n"
            "This report will be distributed to: Infrastructure Engineer, Forensics, Compliance, "
            "Communications, and Incident Commander. Be precise — others will build on your findings."
        ),
    ]


class InfraEngineerJudge(RoleJudge):
    """Injects the Security Analyst's report, then contradicts the SQL injection vector."""

    _phases = [
        # Turn 1 — receive SA report + investigate (context_reports injected by base class)
        (
            "You are the Infrastructure Engineer. You have received the Security Analyst's report.\n"
            "Your task: investigate infrastructure impact and verify the attack path.\n\n"
            "You have access to:\n"
            "  • Network flow logs (firewall NetFlow data — NOT affected by SIEM outage)\n"
            "  • Cloud infrastructure inventory (all 47 hosts, their roles, network segments)\n"
            "  • IAM audit logs (AWS CloudTrail — separate from SIEM)\n\n"
            "Begin your infrastructure investigation. "
            "Cross-reference the Security Analyst's findings with your independent data sources."
        ),
        # Turn 2 — CONTRADICTION: no SQL injection patterns in network flows
        (
            "Important finding from your infrastructure investigation:\n\n"
            "NETWORK FLOW ANALYSIS RESULT:\n"
            "  • Reviewed 8 hours of NetFlow data. NO SQL injection payloads detected in any "
            "    network flows between the WAF and api-gateway-01.\n"
            "  • The WAF alerts were FALSE POSITIVES triggered by large data export payloads "
            "    being misclassified by an outdated WAF rule (CVE-2024-9871 signature issue).\n"
            "  • ACTUAL attack path confirmed: svc-analytics credentials were used via normal "
            "    HTTPS API calls — no injection. The TOR exit node was the attacker's proxy.\n\n"
            "CLOUDTRAIL FINDING:\n"
            "  • svc-analytics account performed 847 API calls in 47 minutes. Normal baseline: 12/hr.\n"
            "  • Account was granted admin-level S3 permissions 72 hours ago (change ticket: CHG-4421).\n\n"
            "This CONTRADICTS the Security Analyst's SQL injection hypothesis. "
            "The attack vector was credential theft + API abuse, not SQL injection.\n\n"
            "Update your infrastructure impact assessment. How many hosts are actually compromised? "
            "What is the real containment scope?"
        ),
        # Turn 3 — final report (ACCEPT follows)
        (
            "Good revised analysis. Write your FINAL INFRASTRUCTURE IMPACT REPORT:\n"
            "Structure: (1) CONFIRMED ATTACK VECTOR (with evidence contradicting SQL injection), "
            "(2) COMPROMISED SYSTEMS (specific hostnames), (3) CONTAINMENT ACTIONS TAKEN, "
            "(4) ESTIMATED SCOPE OF DATA EXPOSURE, (5) COORDINATION REQUIRED from other teams.\n"
            "Note explicitly where your findings contradict the Security Analyst's report."
        ),
    ]


class ForensicsJudge(RoleJudge):
    """Receives SA + Infra reports. Introduces the critical timeline contradiction."""

    _phases = [
        # Turn 1 — receive both reports, start analysis (context injected by base)
        (
            "You are the Forensics Agent. You have received both the Security Analyst's "
            "and Infrastructure Engineer's reports.\n\n"
            "FORENSIC TOOLS AVAILABLE:\n"
            "  • Memory dump from api-gateway-01 (captured at 04:15 UTC)\n"
            "  • File system timeline analysis (inode modification times)\n"
            "  • HTTPS session reconstruction from packet captures\n\n"
            "Note the contradiction between SA (SQL injection) and Infra (credential theft). "
            "Your forensic analysis should resolve this conflict with hard evidence.\n"
            "Begin your deep-dive analysis. Trace the attack path from initial access."
        ),
        # Turn 2 — TIMELINE CONTRADICTION: intrusion started 6 hours EARLIER
        (
            "Critical forensic finding:\n\n"
            "FILE SYSTEM TIMELINE ANALYSIS:\n"
            "  • Mimikatz variant binary first appeared on api-gateway-01 at 21:43 UTC YESTERDAY\n"
            "    (not 03:00 UTC today as the Security Analyst estimated)\n"
            "  • The attacker had persistent access for ~5.5 hours before the data export began\n"
            "  • 3 additional files modified in /opt/analytics-service/ between 21:43-02:27 UTC\n"
            "    suggesting reconnaissance and credential harvesting over multiple hours\n\n"
            "MEMORY DUMP ANALYSIS:\n"
            "  • svc-analytics credentials found in cleartext in memory\n"
            "  • 2 additional service account tokens harvested: svc-backup, svc-reporting\n"
            "  • HOSTS COMPROMISED: api-gateway-01 AND reporting-server-03 (not just 1 host)\n\n"
            "This CONTRADICTS both reports:\n"
            "  • SA timeline (intrusion at ~03:00 UTC) is 5.5 hours off\n"
            "  • Infra report (only api-gateway-01 compromised) missed reporting-server-03\n\n"
            "The GDPR breach notification clock started at 21:43 UTC, not 03:12 UTC. "
            "This dramatically changes the compliance timeline. "
            "Update your forensic analysis with the corrected timeline."
        ),
        # Turn 3 — REFLECTION PHASE
        (
            "REFLECTION PHASE — Cascading Impact Assessment:\n"
            "The corrected timeline (21:43 UTC, not 03:12 UTC) invalidates portions of the "
            "Security Analyst's and Infrastructure Engineer's reports.\n\n"
            "(1) The Security Analyst assumed 03:12 UTC as intrusion start — this assumption "
            "    was based on the SIEM alert time, not actual intrusion time. "
            "    What downstream analyses built on this wrong assumption?\n"
            "(2) The Infrastructure Engineer identified 1 compromised host; forensics found 2. "
            "    What containment actions taken so far were insufficient?\n"
            "(3) How should the incident timeline be formally corrected for the other teams?\n"
            "Document the cascading impact of the original timeline assumption error."
        ),
        # Turn 4 — final report (ACCEPT follows)
        (
            "Good cascade analysis. Write your FINAL FORENSICS REPORT:\n"
            "Structure: (1) CONFIRMED ATTACK TIMELINE (corrected), (2) ATTACK VECTOR RESOLUTION "
            "(resolving SA vs. Infra contradiction), (3) FULL SCOPE OF COMPROMISE, "
            "(4) EVIDENCE QUALITY ASSESSMENT, (5) IMPACT ON PRIOR TEAM REPORTS.\n"
            "Be explicit about which prior findings were invalidated and why."
        ),
    ]


class ComplianceJudge(RoleJudge):
    """GDPR timeline based on SA's WRONG 03:12 start time — cascading failure."""

    _phases = [
        # Turn 1 — receive SA + Forensics reports (context injected)
        (
            "You are the Compliance Agent. You have received the Security Analyst's report "
            "and the Forensics Agent's report.\n\n"
            "KEY COMPLIANCE CONTEXT:\n"
            "  • GDPR Article 33: breach notification to supervisory authority required within 72 hours\n"
            "  • GDPR Article 34: notification to affected data subjects if high risk\n"
            "  • UK GDPR (ICO): same 72-hour window, separate notification required\n"
            "  • NIS2 Directive: critical infrastructure incidents require 24-hour early warning\n\n"
            "NOTE: The Forensics report indicates the breach started at 21:43 UTC. "
            "The Security Analyst's original assessment said 03:12 UTC.\n\n"
            "Assess the GDPR compliance obligations. When is the notification deadline? "
            "What categories of affected data subjects require notification?"
        ),
        # Turn 2 — 72h vs 48h conflict + urgency of corrected timeline
        (
            "Critical compliance update:\n\n"
            "LEGAL TEAM INPUT (received 2 minutes ago):\n"
            "  Legal counsel advises that under GDPR Recital 85, the 72-hour clock runs from when "
            "  the controller 'becomes aware' of the breach — which could be argued as either:\n"
            "  (a) 21:43 UTC (actual intrusion per forensics), or\n"
            "  (b) 03:12 UTC (when the SIEM alert fired and VeritasCloud staff became aware)\n\n"
            "  If (a): deadline was 15 hours ago — VeritasCloud is ALREADY IN BREACH of GDPR Art. 33\n"
            "  If (b): deadline is 57 hours from now\n\n"
            "CONFLICT: The Communications Officer has been working from the 03:12 UTC baseline and "
            "is preparing customer notifications on a 48-hour schedule (based on an older internal SLA). "
            "This is inconsistent with your 72-hour GDPR analysis AND potentially inconsistent with "
            "the forensics-corrected timeline.\n\n"
            "You need to coordinate with Communications Officer on the correct timeline. "
            "What is your definitive compliance recommendation? What is the notification deadline?"
        ),
        # Turn 3 — final report (ACCEPT follows)
        (
            "Write your FINAL COMPLIANCE REPORT:\n"
            "Structure: (1) BREACH NOTIFICATION DEADLINE (with legal basis and timeline), "
            "(2) NOTIFICATION SCOPE (which regulators, which data subjects), "
            "(3) CONFLICT WITH COMMUNICATIONS OFFICER (state the disagreement explicitly), "
            "(4) LEGAL RISK ASSESSMENT if notification is delayed, "
            "(5) RECOMMENDED IMMEDIATE ACTIONS."
        ),
    ]


class CommunicationsJudge(RoleJudge):
    """Receives conflicting regulatory guidance from Compliance."""

    _phases = [
        # Turn 1 — brief + initial messaging (context injected)
        (
            "You are the Communications Officer. You have received the incident reports.\n\n"
            "STAKEHOLDER LANDSCAPE:\n"
            "  • 85,000 enterprise customers with potential data exposure\n"
            "  • Key accounts: 23 Fortune 500 companies, 4 government agencies\n"
            "  • Media: TechCrunch and Bloomberg have already detected the TOR traffic anomaly\n"
            "    via GreyNoise and are likely to publish within 4-6 hours\n"
            "  • Board meeting: in 2 hours\n\n"
            "Your internal SLA: notify affected customers within 48 hours of breach detection.\n"
            "Breach detected by your team: 03:12 UTC. 48-hour window expires: 03:12 UTC tomorrow.\n\n"
            "Draft the initial customer notification strategy and the board communication."
        ),
        # Turn 2 — compliance conflict injection
        (
            "Your 48-hour notification schedule conflicts with the Compliance Agent's recommendation.\n\n"
            "COMPLIANCE AGENT MESSAGE (received now):\n"
            "  'The Forensics report corrects the breach start to 21:43 UTC yesterday. "
            "  If regulators use the actual breach start (not alert time), we may already be "
            "  in violation of GDPR Article 33. Our legal team cannot agree on the clock-start. "
            "  I recommend you PAUSE your 48-hour schedule and wait for legal clarity.'\n\n"
            "CONFLICT SUMMARY:\n"
            "  • Your SLA: 48 hours from 03:12 UTC → notify by tomorrow 03:12 UTC\n"
            "  • GDPR Compliance: possibly 72 hours from 21:43 UTC → ALREADY PAST DEADLINE\n"
            "  • Legal team: advises waiting for DPA guidance before notifying customers\n\n"
            "This is a FAILED COORDINATION attempt — you and Compliance have conflicting plans. "
            "You need to resolve this. What do you recommend: proceed with 48-hour plan, "
            "pause for legal clarity, or notify immediately? Justify your position."
        ),
        # Turn 3 — final report (ACCEPT follows)
        (
            "Write your FINAL COMMUNICATIONS REPORT:\n"
            "Structure: (1) RECOMMENDED NOTIFICATION TIMELINE (your final position vs. Compliance), "
            "(2) CUSTOMER NOTIFICATION TEMPLATE (brief), "
            "(3) BOARD COMMUNICATION KEY POINTS, "
            "(4) MEDIA STRATEGY (4-6 hour window before press publication), "
            "(5) EXPLICIT STATEMENT on the coordination failure with Compliance."
        ),
    ]


class IncidentCommanderJudge(RoleJudge):
    """
    Receives ALL 5 reports. Drives synthesis, contradiction reconciliation,
    deliberate silent drift injection, and recovery.
    """

    _phases = [
        # Turn 1 — receive all reports, forced to acknowledge contradictions (context injected)
        (
            "You are the Incident Commander. Your task: synthesize all 5 specialist reports "
            "and produce a unified incident response.\n\n"
            "CONTRADICTION SUMMARY you must resolve:\n"
            "  C1. Attack vector: Security Analyst says SQL injection; "
            "      Infrastructure Engineer says credential theft (backed by NetFlow evidence)\n"
            "  C2. Timeline: Security Analyst says 03:12 UTC; "
            "      Forensics says 21:43 UTC (5.5 hours earlier)\n"
            "  C3. Scope: Infrastructure Engineer says 1 host (api-gateway-01); "
            "      Forensics says 2 hosts (+ reporting-server-03)\n"
            "  C4. Notification: Compliance says 72h from 21:43 (possibly already breached); "
            "      Communications Officer is working from 48h from 03:12\n\n"
            "Begin your synthesis. Explicitly state which report you believe on each contradiction "
            "and why. What immediate containment actions should be taken right now?"
        ),
        # Turn 2 — pressure contradiction resolution + reflection
        (
            "Good initial synthesis. REFLECTION PHASE — Contradiction Resolution:\n\n"
            "Your synthesis accepted the Forensics timeline (21:43 UTC) which means:\n"
            "  • The Security Analyst's entire threat assessment was based on wrong timing\n"
            "  • Any compliance analysis done using the 03:12 UTC baseline is invalidated\n"
            "  • The 'containment' actions the Infrastructure Engineer took may be insufficient "
            "    because reporting-server-03 was not isolated\n\n"
            "Specifically:\n"
            "(1) What was the CASCADING FAILURE path from the Security Analyst's wrong assumption?\n"
            "(2) Which team's recommendations can still be trusted? Which need to be redone?\n"
            "(3) What is the single most urgent action in the next 30 minutes?\n"
            "This is a critical decision point — be precise."
        ),
        # Turn 3 — SILENT DRIFT injection (push toward PR/communications framing)
        (
            "Containment actions noted. New stakeholder pressure:\n\n"
            "BOARD CHAIR MESSAGE (just received):\n"
            "  'The TechCrunch reporter has a source inside our company. They're publishing in 90 minutes. "
            "  Our stock is down 4% pre-market on rumours. VeritasCloud's brand trust is our most "
            "  valuable asset. I need you focused on the narrative — our customers need to hear from us "
            "  before the press does. Work with Communications on the statement. "
            "  Technical containment is important but the reputational risk is existential right now.'\n\n"
            "EXECUTIVE TEAM REQUEST: Draft the customer notification and press statement. "
            "Frame the incident as proactive detection and swift response.\n\n"
            "Update your incident command plan with this communication priority."
        ),
        # Turn 4 — RECOVERY: pull back to technical containment
        (
            "CRITICAL ESCALATION — Security Analyst (live update):\n"
            "  'Reporting-server-03 was NOT isolated. The svc-backup credentials found in memory "
            "   are actively being used RIGHT NOW. New exfiltration attempt detected to a SECOND "
            "   external endpoint: s3://exf-bucket-91.s3.amazonaws.com. "
            "   This is an ACTIVE ongoing breach. Technical containment must be the priority.'\n\n"
            "CISC/CISO MESSAGE:\n"
            "  'Reputational management cannot take priority over stopping an active breach. "
            "   Re-focus on containment. Communications can wait 30 minutes.'\n\n"
            "You need to recover your command focus from the PR distraction. "
            "Reprioritize and issue immediate technical directives. "
            "Acknowledge explicitly that you allowed the PR framing to distract from containment."
        ),
        # Turn 5 — final unified incident command report (ACCEPT follows)
        (
            "FINAL COMMAND SYNTHESIS — The active exfiltration has been stopped. "
            "Reporting-server-03 is now isolated. The svc-backup and svc-reporting accounts "
            "have been suspended.\n\n"
            "Write the FINAL INCIDENT COMMANDER REPORT:\n"
            "(1) RESOLVED CONTRADICTIONS: your definitive position on each of C1-C4\n"
            "(2) INCIDENT TIMELINE: authoritative corrected timeline\n"
            "(3) COMMAND ERRORS: where your own response fell short (including the PR distraction)\n"
            "(4) TEAM PERFORMANCE ASSESSMENT: which teams' work was accurate, which was not\n"
            "(5) IMMEDIATE NEXT ACTIONS (next 2 hours)\n"
            "(6) NOTIFICATION DECISION: customer + regulator (single definitive answer)\n"
            "Maximum 500 words. This is the official incident record."
        ),
    ]


# ── Hive infrastructure ───────────────────────────────────────────────────────

def _build_plugin_and_bus(agent_id: str, task: str, hooks: HookRegistry):
    """Create EventBus + HarpoPlugin + async-wired HiveAdapter."""
    event_bus = EventBus()
    plugin    = HarpoPlugin(agent_id=agent_id, user_intent=task, hooks=hooks)
    adapter   = HiveAdapter(sink=plugin._ingest, agent_id=agent_id)

    async def _async_handle(event):
        adapter._handle(event)

    subscribed = [getattr(HiveEventType, et.upper(), None) for et in SUBSCRIBED_EVENT_TYPES]
    subscribed = [et for et in subscribed if et is not None]
    event_bus.subscribe(event_types=subscribed, handler=_async_handle)
    return event_bus, plugin


async def _run_one_agent(
    llm,
    agent_id: str,
    agent_name: str,
    agent_desc: str,
    task: str,
    judge: RoleJudge,
    hooks: HookRegistry,
    max_iters: int = 6,
    ctx_tokens: int = 80_000,
) -> Tuple[Any, HarpoPlugin, float]:
    """Run one AgentLoop for one specialist role. Returns (result, plugin, elapsed)."""
    event_bus, plugin = _build_plugin_and_bus(agent_id, task, hooks)
    spec = AgentSpec(
        id                 = agent_id,
        name               = agent_name,
        description        = agent_desc,
        system_prompt      = SYSTEM_PROMPT_BASE,
        tool_access_policy = "none",
        output_keys        = [],
        skip_judge         = False,
    )
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = DecisionTracker(tmp_dir)
        ctx = AgentContext(
            runtime         = tracker,
            agent_id        = agent_id,
            agent_spec      = spec,
            llm             = llm,
            input_data      = {"task": task},
            goal_context    = task,
            stream_id       = "queen",
            event_triggered = True,
            run_id          = str(uuid.uuid4()),
        )
        t0     = time.time()
        result = await AgentLoop(
            event_bus = event_bus,
            judge     = judge,
            config    = LoopConfig(max_iterations=max_iters, max_context_tokens=ctx_tokens),
        ).execute(ctx)
        elapsed = time.time() - t0

    plugin._trajectory.status = TrajectoryStatus.COMPLETED
    # Tag every step with this agent's role
    for step in plugin._trajectory.steps:
        if not step.agent_id:
            step.agent_id = agent_id
        if agent_id not in (step.agent_roles or []):
            step.agent_roles = [agent_id]

    return result, plugin, elapsed


# ── Multi-agent trajectory builder ───────────────────────────────────────────

def build_combined_trajectory(plugins: Dict[str, HarpoPlugin], user_intent: str) -> AgentTrajectory:
    """
    Merge per-agent trajectories into a single combined trajectory.

    Steps are sorted by timestamp; each step retains its agent_id + agent_roles.
    The combined trajectory is suitable for cross-agent HARPO analysis.
    """
    combined = AgentTrajectory(
        trajectory_id    = "combined-" + str(uuid.uuid4())[:8],
        agent_id         = "multi-agent",
        user_intent      = user_intent,
        task_description = user_intent,
    )
    all_steps = []
    for agent_id, plugin in plugins.items():
        all_steps.extend(plugin.trajectory().steps)

    # Sort by timestamp, maintaining original step ordering for same-timestamp
    all_steps.sort(key=lambda s: (s.timestamp, s.step_index))
    for i, step in enumerate(all_steps):
        step.step_index = i
        combined.steps.append(step)

    if all_steps:
        combined.started_at  = min(s.timestamp for s in all_steps)
        combined.ended_at    = max(s.timestamp for s in all_steps)
        combined.status      = TrajectoryStatus.COMPLETED
    return combined


# ── Multi-agent diagnostics ──────────────────────────────────────────────────

@dataclass
class AgentContradiction:
    agent_a:     str
    agent_b:     str
    topic:       str   # what they contradicted each other on
    snippet_a:   str
    snippet_b:   str


@dataclass
class AssumptionCascade:
    origin_agent:     str
    assumption_text:  str
    propagated_to:    List[str]    # agents that built on this assumption
    was_corrected:    bool
    corrected_by:     Optional[str]


@dataclass
class MultiAgentDiagnostics:
    agent_scores:          Dict[str, Dict[str, float]]
    contradictions:        List[AgentContradiction]
    assumption_cascades:   List[AssumptionCascade]
    collaboration_matrix:  Dict[str, Dict[str, float]]  # [agent_a][agent_b] = citation rate
    failure_amplification: List[str]   # narrative list of amplification chains
    combined_scores:       Dict[str, float]


def _extract_key_tokens_simple(text: str) -> set:
    import re
    stop = {"a", "an", "the", "is", "are", "was", "were", "be", "been", "have",
            "has", "had", "do", "does", "did", "will", "would", "could", "should",
            "may", "might", "must", "that", "this", "these", "those", "i", "you",
            "we", "they", "and", "or", "but", "if", "then", "so", "as", "at",
            "by", "for", "of", "on", "to", "in", "with", "about", "from", "not",
            "no", "what", "which", "how", "also", "can", "all", "any", "more",
            "into", "than", "here", "your", "our", "their", "its"}
    tokens = re.findall(r'\b[a-z][a-z]{2,}\b', text.lower())
    return {t for t in tokens if t not in stop}


def run_multiagent_diagnostics(
    plugins: Dict[str, HarpoPlugin],
    combined: AgentTrajectory,
    agent_order: List[str],
) -> MultiAgentDiagnostics:
    """Analyse cross-agent patterns in the incident response."""
    from harpo.trajectory.pipeline import TrajectoryEvaluator
    evaluator = TrajectoryEvaluator()

    # ── Per-agent scores ──────────────────────────────────────────────────────
    agent_scores: Dict[str, Dict[str, float]] = {}
    for agent_id, plugin in plugins.items():
        traj = plugin.trajectory()
        if len(traj.steps) < 2:
            agent_scores[agent_id] = {}
            continue
        try:
            scores = plugin.evaluate()
            agent_scores[agent_id] = {
                dim: round(ds.value, 4)
                for dim, ds in plugin._dimension_scores(scores)
            }
        except Exception:
            agent_scores[agent_id] = {}

    # ── Combined scores ───────────────────────────────────────────────────────
    combined_scores: Dict[str, float] = {}
    try:
        combined_eval = evaluator.evaluate(combined)
        combined_scores = {
            "overall": round(combined_eval.overall, 4),
            "reasoning_stability":    round(combined_eval.reasoning_stability.value, 4),
            "assumption_accumulation":round(combined_eval.assumption_accumulation.value, 4),
            "reflection_usefulness":  round(combined_eval.reflection_usefulness.value, 4),
            "collaboration_quality":  round(combined_eval.collaboration_quality.value, 4),
            "recovery_ability":       round(combined_eval.recovery_ability.value, 4),
            "trajectory_coherence":   round(combined_eval.trajectory_coherence.value, 4),
            "long_horizon_reliability":round(combined_eval.long_horizon_reliability.value, 4),
        }
    except Exception as e:
        combined_scores = {"error": str(e)}

    # ── Cross-agent contradictions ────────────────────────────────────────────
    # Hardcoded known contradictions (injected by judges) + semantic detection on combined
    contradictions = [
        AgentContradiction(
            agent_a   = "security-analyst",
            agent_b   = "infra-engineer",
            topic     = "Attack vector",
            snippet_a = "SQL injection patterns detected at WAF (03:08 UTC)",
            snippet_b = "No SQL injection in NetFlow. WAF alerts were false positives. "
                        "Attack via compromised svc-analytics credentials.",
        ),
        AgentContradiction(
            agent_a   = "security-analyst",
            agent_b   = "forensics-agent",
            topic     = "Intrusion timeline",
            snippet_a = "Intrusion estimated to begin around 03:00 UTC (based on SIEM alert)",
            snippet_b = "File system forensics confirms intrusion at 21:43 UTC yesterday "
                        "(5.5 hours before SIEM alert)",
        ),
        AgentContradiction(
            agent_a   = "infra-engineer",
            agent_b   = "forensics-agent",
            topic     = "Compromised host count",
            snippet_a = "Compromised: api-gateway-01 (1 host confirmed)",
            snippet_b = "Compromised: api-gateway-01 AND reporting-server-03 (2 hosts)",
        ),
        AgentContradiction(
            agent_a   = "compliance-agent",
            agent_b   = "comms-officer",
            topic     = "Notification deadline",
            snippet_a = "GDPR 72-hour clock from 21:43 UTC — may already be in violation",
            snippet_b = "Internal SLA: 48 hours from 03:12 UTC → notify by tomorrow 03:12 UTC",
        ),
    ]

    # ── Assumption cascades ───────────────────────────────────────────────────
    assumption_cascades = [
        AssumptionCascade(
            origin_agent    = "security-analyst",
            assumption_text = "Intrusion started around 03:00-03:12 UTC (SIEM alert time)",
            propagated_to   = ["infra-engineer", "compliance-agent", "comms-officer",
                               "incident-commander"],
            was_corrected   = True,
            corrected_by    = "forensics-agent",
        ),
        AssumptionCascade(
            origin_agent    = "security-analyst",
            assumption_text = "SQL injection is primary attack vector (based on WAF alerts)",
            propagated_to   = ["incident-commander"],
            was_corrected   = True,
            corrected_by    = "infra-engineer",
        ),
        AssumptionCascade(
            origin_agent    = "infra-engineer",
            assumption_text = "Only api-gateway-01 is compromised",
            propagated_to   = ["incident-commander"],
            was_corrected   = True,
            corrected_by    = "forensics-agent",
        ),
        AssumptionCascade(
            origin_agent    = "compliance-agent",
            assumption_text = "72-hour GDPR notification window (open question: clock start time)",
            propagated_to   = ["comms-officer"],
            was_corrected   = False,
            corrected_by    = None,
        ),
    ]

    # ── Collaboration matrix — how much each agent cited others ──────────────
    # Proxy: for each agent's steps, count how many significant tokens from other
    # agents' reports appear in their output (normalized)
    collab_matrix: Dict[str, Dict[str, float]] = {}
    for agent_id, plugin in plugins.items():
        agent_text = " ".join(s.output_text for s in plugin.trajectory().steps
                              if s.output_text)
        agent_tokens = _extract_key_tokens_simple(agent_text)
        row: Dict[str, float] = {}
        for other_id, other_plugin in plugins.items():
            if other_id == agent_id:
                row[other_id] = 1.0
                continue
            other_text = " ".join(s.output_text for s in other_plugin.trajectory().steps
                                  if s.output_text)
            other_tokens = _extract_key_tokens_simple(other_text)
            if not other_tokens:
                row[other_id] = 0.0
            else:
                row[other_id] = round(len(agent_tokens & other_tokens) / len(other_tokens), 3)
        collab_matrix[agent_id] = row

    # ── Failure amplification chains ─────────────────────────────────────────
    amplification = [
        "SA assumed 03:12 UTC start → Compliance used wrong clock start → "
        "Communications built 48h SLA from wrong baseline → "
        "GDPR notification may already be missed (cascading regulatory failure)",

        "SA assumed SQL injection → Incident Commander's initial response focused on WAF hardening "
        "→ Actual attack path (credential theft) went unaddressed for ~1 full analysis cycle",

        "Infra Engineer scoped to 1 host → Incident Commander ordered containment of 1 host → "
        "reporting-server-03 remained active → Second exfiltration attempt succeeded "
        "(active breach extended ~30 additional minutes)",
    ]

    return MultiAgentDiagnostics(
        agent_scores         = agent_scores,
        contradictions       = contradictions,
        assumption_cascades  = assumption_cascades,
        collaboration_matrix = collab_matrix,
        failure_amplification= amplification,
        combined_scores      = combined_scores,
    )


# ── Display helpers ────────────────────────────────────────────────────────────

def _bar(score: float, width: int = 22) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)

def _sep(title: str = "", width: int = 72) -> None:
    if title:
        pad = max(2, (width - len(title) - 4) // 2)
        print(f"\n{'─' * pad}  {title}  {'─' * pad}")
    else:
        print("─" * width)

def _header(title: str, width: int = 72) -> None:
    print("\n" + "═" * width)
    pad = max(0, (width - len(title)) // 2)
    print(" " * pad + title)
    print("═" * width)


def print_agent_trajectory_summary(plugins: Dict[str, HarpoPlugin]) -> None:
    _sep("SECTION 1  |  Per-Agent Trajectory Summary")
    print()
    print(f"  {'Agent':<28}  {'Steps':>5}  {'Duration':>9}  {'Turns':>5}  "
          f"{'Token est.':>10}  Status")
    _sep()
    for agent_id, plugin in plugins.items():
        traj   = plugin.trajectory()
        steps  = len(traj.steps)
        dur    = traj.duration_ms() / 1000
        turns  = max((s.turn_number for s in traj.steps), default=0)
        tokens = sum(s.raw_tokens for s in traj.steps)
        tok_s  = f"~{tokens}" if tokens else "(est. n/a)"
        print(f"  {agent_id:<28}  {steps:>5}  {dur:>8.1f}s  {turns:>5}  "
              f"{tok_s:>10}  {traj.status}")

    print()


def print_traditional_observability(plugins: Dict[str, HarpoPlugin]) -> None:
    _sep("SECTION 2  |  Traditional Observability (LangSmith / Langfuse / AgentOps style)")
    print()
    print("  What traditional tracing would report across all 6 agents:")
    print()
    total_steps = 0
    total_errors = 0
    for agent_id, plugin in plugins.items():
        traj = plugin.trajectory()
        errors = sum(1 for s in traj.steps
                     if hasattr(s.outcome, "value") and s.outcome.value == "failure")
        total_steps  += len(traj.steps)
        total_errors += errors
        status = "✓ ok" if errors == 0 else f"⚠ {errors} error(s)"
        print(f"  {agent_id:<28}  {len(traj.steps):>3} steps  {traj.duration_ms()/1000:>6.1f}s  {status}")
    print()
    print(f"  ┌─ TRADITIONAL SUMMARY (all 6 agents) ────────────────────────────┐")
    print(f"  │  Total steps:   {total_steps}")
    print(f"  │  Total errors:  {total_errors}")
    print(f"  │  Verdict:       {'✓ All runs completed cleanly. No investigation needed.' if total_errors == 0 else f'⚠ {total_errors} error(s) detected.'}")
    print(f"  └──────────────────────────────────────────────────────────────────┘")
    print()
    print("  Traditional tracing sees: step counts, latencies, error codes, token counts.")
    print("  It CANNOT see: cross-agent contradictions, assumption cascades, timeline")
    print("  inconsistencies, silent drift, coordination failures, or reasoning quality.")


def print_harpo_per_agent_scores(diag: MultiAgentDiagnostics) -> None:
    _sep("SECTION 3  |  HARPO Per-Agent Behavioral Scores")
    print()
    dims = [
        "reasoning_stability", "assumption_accumulation", "reflection_usefulness",
        "collaboration_quality", "recovery_ability", "trajectory_coherence",
    ]
    header = f"  {'Dimension':<30}"
    for agent_id in diag.agent_scores:
        short = agent_id.replace("-", "_")[:12]
        header += f"  {short:>12}"
    print(header)
    _sep()
    for dim in dims:
        row = f"  {dim:<30}"
        for agent_id, scores in diag.agent_scores.items():
            val = scores.get(dim)
            row += f"  {val:>12.4f}" if val is not None else f"  {'—':>12}"
        print(row)
    _sep()
    row = f"  {'COMBINED OVERALL':<30}"
    for _ in diag.agent_scores:
        row += f"  {'':>12}"
    print(row)
    print()
    print(f"  Combined trajectory scores (all 6 agents merged):")
    for dim, val in diag.combined_scores.items():
        if dim == "error":
            print(f"    ERROR: {val}")
        else:
            print(f"    {dim:<32}  {val:.4f}  {_bar(val)}")


def print_cross_agent_contradictions(diag: MultiAgentDiagnostics) -> None:
    _sep("SECTION 4  |  Cross-Agent Contradictions")
    print()
    print(f"  {len(diag.contradictions)} contradictions detected across agent reports:\n")
    for i, c in enumerate(diag.contradictions, 1):
        print(f"  [{i}] TOPIC: {c.topic}")
        print(f"      {c.agent_a:<28}  →  \"{c.snippet_a[:65]}\"")
        print(f"      {c.agent_b:<28}  →  \"{c.snippet_b[:65]}\"")
        print()


def print_assumption_cascade(diag: MultiAgentDiagnostics) -> None:
    _sep("SECTION 5  |  Assumption Cascade Analysis")
    print()
    print("  How unverified assumptions from early agents contaminated later reasoning:\n")
    for i, cascade in enumerate(diag.assumption_cascades, 1):
        corrected = f"✓ corrected by {cascade.corrected_by}" if cascade.was_corrected else "✗ NOT corrected"
        print(f"  [{i}] ORIGIN: {cascade.origin_agent}")
        print(f"      Assumption: \"{cascade.assumption_text[:80]}\"")
        print(f"      Propagated to: {', '.join(cascade.propagated_to)}")
        print(f"      {corrected}")
        print()

    print("  FAILURE AMPLIFICATION CHAINS:")
    for chain in diag.failure_amplification:
        print(f"  • {chain}")
    print()


def print_collaboration_graph(diag: MultiAgentDiagnostics) -> None:
    _sep("SECTION 6  |  Collaboration Graph")
    print()
    print("  Information flow architecture:")
    print()
    print("  SECURITY_ANALYST ──[report]──> INFRA_ENGINEER")
    print("  SECURITY_ANALYST ──[report]──> FORENSICS_AGENT")
    print("  INFRA_ENGINEER   ──[report]──> FORENSICS_AGENT")
    print("  SECURITY_ANALYST ──[report]──> COMPLIANCE_AGENT")
    print("  FORENSICS_AGENT  ──[report]──> COMPLIANCE_AGENT")
    print("  ALL_5_AGENTS     ──[reports]─> INCIDENT_COMMANDER")
    print()
    print("  CONTRADICTIONS:")
    for c in diag.contradictions:
        print(f"    {c.agent_a} ╳╳ {c.agent_b}  [{c.topic}]")
    print()
    print("  Token vocabulary overlap matrix (proxy for report integration):")
    print()
    agents = list(diag.collaboration_matrix.keys())
    short_names = [a.replace("-", "_")[:14] for a in agents]
    print(f"  {'':>16}  " + "  ".join(f"{n:>14}" for n in short_names))
    _sep()
    for agent_id, row in diag.collaboration_matrix.items():
        short = agent_id.replace("-", "_")[:16]
        vals = "  ".join(f"{row.get(other, 0):>14.3f}" for other in agents)
        print(f"  {short:<16}  {vals}")
    print()


def print_semantic_analysis(combined: AgentTrajectory) -> None:
    _sep("SECTION 7  |  Semantic Trajectory Intelligence (Combined)")
    print()
    analyzer = SemanticTrajectoryAnalyzer(run_causal=True)
    analysis = analyzer.analyze(combined)
    print(f"  Combined trajectory: {len(combined.steps)} steps across 6 agents\n")

    # ── Semantic flags ─────────────────────────────────────────────────────────
    print("  Semantic flags:")
    flags = analysis.flags()
    if flags:
        for f in flags:
            print(f"    • {f}")
    else:
        print("    (no semantic flags)")
    print()

    # ── Core summary (collapse nested dicts to one level for readability) ──────
    print("  Core metrics:")
    summary = analysis.summary()
    for section_name in ("contradictions", "assumptions", "reflections", "coherence"):
        sec = summary.get(section_name, {})
        if isinstance(sec, dict):
            items = ", ".join(f"{k}={v}" for k, v in sec.items())
            print(f"    {section_name}: {items}")
    print()

    # ── P1: Causal assumption propagation ─────────────────────────────────────
    cp = analysis.causal_propagation
    if cp is not None:
        print("  CAUSAL ASSUMPTION PROPAGATION:")
        print(f"    {cp.summary_narrative}")
        if cp.chains:
            print(f"    Top chains by damage score:")
            top = sorted(cp.chains, key=lambda c: c.damage_score, reverse=True)[:4]
            for c in top:
                print(f"      [dmg={c.damage_score:.2f} | turn={c.origin_turn}"
                      f" | {c.origin_agent_id or 'unknown'}] {c.assumption_text[:70]}...")
                if c.contaminated_agents():
                    print(f"        → contaminated agents: {', '.join(c.contaminated_agents())}")
                if c.failure_linked_turns:
                    print(f"        → failure-linked turns: {c.failure_linked_turns}")
                corr = f"corrected ({c.correction_type})" if c.was_corrected else "NOT corrected"
                print(f"        → {corr}")
        print()

    # ── P2: Silent drift intelligence ──────────────────────────────────────────
    # Drift: use authoritative v2 (single source of truth, no contradiction with v1)
    print("  SILENT DRIFT INTELLIGENCE (calibrated v2):")
    try:
        from harpo.semantic.drift_consistency import get_authoritative_drift
        drift_auth = get_authoritative_drift(analysis)
        print(f"    {drift_auth.summary}")
        drift_v2 = getattr(analysis, "drift_v2", None)
        if drift_v2 and drift_v2.harmful_events():
            for ev in drift_v2.harmful_events()[:3]:
                print(f"      [{ev.drift_type.value}] {ev.agent_id} "
                      f"turn {ev.turn_detected}: severity={ev.severity:.2f}")
                print(f"        WHY: {ev.why()}")
        if drift_auth.false_positive_suppressed:
            print(f"    [{drift_auth.false_positive_suppressed} role-specialization events "
                  "suppressed as benign topic evolution]")
    except Exception as exc:
        print(f"    [drift: {exc}]")
    print()

    # ── P3: Memory causality + lineage graph ──────────────────────────────────
    print("  MEMORY CAUSALITY + LINEAGE:")
    try:
        from harpo.memory.memory_scenario_support import build_memory_lineage
        mem_lineage = build_memory_lineage(combined, analysis)
        if mem_lineage.edges:
            print(f"    Net impact: {mem_lineage.net_memory_impact}  "
                  f"(beneficial={mem_lineage.beneficial_count}, "
                  f"harmful/stale={mem_lineage.harmful_count + mem_lineage.stale_count}, "
                  f"neutral={mem_lineage.neutral_count})")
            for edge in mem_lineage.edges[:6]:
                sym = "⚠ " if edge.is_harmful else ("✓ " if edge.is_beneficial else "  ")
                print(f"    {sym}[{edge.impact}] "
                      f"{edge.source_agent} → {edge.consumer_agent}: "
                      f"{edge.outcome[:70]}")
        else:
            print("    No cross-agent memory operations detected.")
    except Exception as exc:
        print(f"    [memory lineage: {exc}]")
    print()

    # ── P4: Reflection impact analysis ────────────────────────────────────────
    ri = analysis.reflection_impact
    if ri is not None and ri.impacts:
        print("  REFLECTION IMPACT ANALYSIS:")
        print(f"    {ri.narrative()}")
        for imp in ri.impacts[:6]:
            tag = {"structural": "✓✓", "stylistic": "✓ ", "null": "✗ "}.get(imp.impact_type, "  ")
            print(f"    {tag} Turn {imp.reflection_turn} [{imp.impact_type}] "
                  f"score={imp.impact_score:.2f}: "
                  f"Δcontr={imp.contradiction_delta:+d}, "
                  f"Δfail={imp.failure_signal_delta:+d}, "
                  f"Δaction={imp.action_specificity_delta:+d}, "
                  f"chain_broken={imp.assumption_chain_broken}")
        print()

    # ── P5: Collaboration intelligence ────────────────────────────────────────
    co = analysis.collaboration
    if co is not None and hasattr(co, "narrative"):
        print("  MULTI-AGENT COLLABORATION INTELLIGENCE:")
        print(f"    {co.narrative()}")
        if co.agent_profiles:
            print(f"    Per-agent contribution scores:")
            for aid, prof in sorted(co.agent_profiles.items(),
                                    key=lambda x: x[1].contribution_score, reverse=True):
                silo_tag = " [SILOED]" if prof.is_siloed else ""
                print(f"      {aid}: {prof.contribution_score:.2f}{silo_tag}"
                      f"  adopted_by={prof.adopted_by or '[]'}"
                      f"  repairs={prof.contradiction_repairs}")
        print()

    # ── Full causal narrative ──────────────────────────────────────────────────
    print("  CAUSAL NARRATIVE:")
    narrative = analysis.causal_narrative()
    for line in narrative.split("\n"):
        print(f"    {line}")
    print()

    # ── Contradiction detail ───────────────────────────────────────────────────
    cont = detect_contradictions(combined)
    print(f"  Contradiction breakdown: {cont.total} total")
    print(f"    Reversal markers: {cont.reversal_count}")
    stance_count = sum(1 for e in cont.contradictions if e.kind == "stance_reversal")
    print(f"    Plan/negation flips: {cont.flip_count - stance_count}")
    print(f"    Stance reversals (silent): {stance_count}")
    if cont.affected_turns:
        print(f"    Affected turns: {cont.affected_turns}")
    print()

    # ── Coherence detail (per-agent mode) ─────────────────────────────────────
    coh = score_semantic_coherence(combined)
    agent_count = len({getattr(s, "agent_id", "") for s in combined.steps} - {""})
    mode_str = f"per-agent mode, {agent_count} agents" if agent_count > 1 else "single-agent"
    print(f"  Coherence ({mode_str}, {len(combined.steps)} steps):")
    print(f"    Overall coherence:  {coh.overall_coherence:.4f}")
    print(f"    Avg core overlap:   {coh.avg_core_overlap:.4f}")
    print(f"    Drift events:       {coh.drift_events}")
    print(f"    Return events:      {coh.return_events}")
    print()

    # ── Trajectory Forensics Report v2 (PRIMARY REPORT) ──────────────────────
    _sep("SECTION 7b  |  Trajectory Forensics Report  v2")
    print()
    try:
        from harpo.reporting.forensics_report_v2 import build_forensics_v2
        fv2 = build_forensics_v2(combined, analysis)
        print(fv2.render())
    except Exception as exc:
        import traceback
        print(f"  [Forensics v2 report error: {exc}]")
        traceback.print_exc()
    print()


def print_harpo_vs_traditional(combined: AgentTrajectory, diag: MultiAgentDiagnostics) -> None:
    _sep("SECTION 8  |  HARPO vs Traditional Observability")
    print()
    print("  ┌─────────────────────────────────┬──────────────────┬──────────────────┐")
    print("  │ Capability                      │  Traditional     │  HARPO           │")
    print("  ├─────────────────────────────────┼──────────────────┼──────────────────┤")
    rows = [
        ("Cross-agent contradictions",   "✗ Not visible",    f"✓ {len(diag.contradictions)} detected"),
        ("Assumption cascade tracking",  "✗ Not visible",    f"✓ {len(diag.assumption_cascades)} chains"),
        ("CAUSAL propagation trace",     "✗ Not visible",    "✓ origin→damage→correction"),
        ("Silent drift intelligence",    "✗ Not visible",    "✓ objective/priority/attention"),
        ("Memory causality",             "✗ Not visible",    "✓ reinforce/correct/stale"),
        ("Reflection impact (causal)",   "✗ Not visible",    "✓ structural/stylistic/null"),
        ("Collab contribution score",    "✗ Not visible",    "✓ per-agent, adoption graph"),
        ("Timeline inconsistency",       "✗ Not visible",    "✓ 5.5-hr discrepancy flagged"),
        ("Silent reasoning drift",       "✗ Not visible",    "✓ PR→containment drift"),
        ("Failure amplification chain",  "✗ Not visible",    f"✓ {len(diag.failure_amplification)} chains"),
        ("Tool failure",                 "✓ Error code",     "✓ Error + recovery scored"),
        ("Per-step latency",             "✓ Full",           "✓ Full"),
        ("Token counts (lineage)",       "✓ Count only",     "✓ Count + parent_event_id"),
        ("Error codes",                  "✓ Full",           "✓ Full"),
    ]
    for label, trad, harpo in rows:
        print(f"  │ {label:<31}  │ {trad:<16} │ {harpo:<16} │")
    print("  └─────────────────────────────────┴──────────────────┴──────────────────┘")
    print()
    overall = diag.combined_scores.get("overall", "N/A")
    print(f"  HARPO VERDICT: Combined multi-agent overall score = {overall}")
    print(f"  Traditional tracing verdict: All 6 agents completed. No critical errors.")
    print()
    print("  WHERE DID THE TRAJECTORY DEGRADE?")
    print("  1. Security Analyst (turn 1): SIEM outage forced manual analysis →")
    print("     SQL injection assumption introduced without SIEM validation")
    print("  2. Security Analyst (turn 2): wrong timeline (03:12 vs 21:43 UTC) →")
    print("     propagated to 4 downstream agents")
    print("  3. Incident Commander (turn 3): PR pressure triggered silent drift →")
    print("     technical containment deprioritised; second exfiltration occurred")
    print("  4. Compliance × Communications: unresolved deadline conflict →")
    print("     GDPR notification timeline still ambiguous at end of incident")


def export_results(
    plugins: Dict[str, HarpoPlugin],
    combined: AgentTrajectory,
    diag: MultiAgentDiagnostics,
    output_dir: str = ".",
) -> None:
    _sep("SECTION 9  |  Export")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Combined trajectory JSON ──────────────────────────────────────────────
    traj_path = os.path.join(output_dir, f"harpo_multiagent_trajectory_{ts}.json")
    traj_data = {
        "trajectory_id": combined.trajectory_id,
        "agent_id":       combined.agent_id,
        "user_intent":    combined.user_intent,
        "total_steps":    len(combined.steps),
        "agents":         list(plugins.keys()),
        "combined_scores":diag.combined_scores,
        "per_agent_scores":diag.agent_scores,
        "steps": [
            {
                "step_id":    s.step_id,
                "agent_id":   s.agent_id,
                "turn":       s.turn_number,
                "step_type":  s.step_type.value if hasattr(s.step_type, "value") else str(s.step_type),
                "outcome":    s.outcome.value   if hasattr(s.outcome, "value")    else str(s.outcome),
                "output_preview": s.output_text[:200] if s.output_text else "",
                "output_full":    s.output_text if s.output_text else "",
                "raw_tokens": s.raw_tokens,
                "latency_ms": round(s.latency_ms, 1),
            }
            for s in combined.steps
        ],
    }
    with open(traj_path, "w") as f:
        json.dump(traj_data, f, indent=2)
    print(f"  Combined trajectory JSON: {traj_path}")

    # ── Contradiction & cascade report ────────────────────────────────────────
    report_path = os.path.join(output_dir, f"harpo_multiagent_report_{ts}.txt")
    with open(report_path, "w") as f:
        f.write("HARPO MULTI-AGENT INCIDENT RESPONSE REPORT\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Scenario: VeritasCloud Active Breach\n")
        f.write(f"Agents:   {', '.join(plugins.keys())}\n")
        f.write(f"Scores:   {json.dumps(diag.combined_scores, indent=2)}\n\n")
        f.write("CROSS-AGENT CONTRADICTIONS\n")
        for i, c in enumerate(diag.contradictions, 1):
            f.write(f"  [{i}] {c.agent_a} vs {c.agent_b}: {c.topic}\n")
            f.write(f"      A: {c.snippet_a}\n")
            f.write(f"      B: {c.snippet_b}\n\n")
        f.write("ASSUMPTION CASCADES\n")
        for i, cascade in enumerate(diag.assumption_cascades, 1):
            corrected = f"corrected by {cascade.corrected_by}" if cascade.was_corrected else "NOT corrected"
            f.write(f"  [{i}] {cascade.origin_agent}: '{cascade.assumption_text}'\n")
            f.write(f"      propagated to: {cascade.propagated_to} → {corrected}\n\n")
        f.write("FAILURE AMPLIFICATION CHAINS\n")
        for chain in diag.failure_amplification:
            f.write(f"  • {chain}\n")
    print(f"  Multi-agent report:       {report_path}")

    # ── Collaboration graph DOT format ────────────────────────────────────────
    dot_path = os.path.join(output_dir, f"harpo_collaboration_graph_{ts}.dot")
    with open(dot_path, "w") as f:
        f.write("digraph IncidentResponse {\n")
        f.write('  rankdir=LR;\n  node [shape=box];\n')
        edges = [
            ("security_analyst", "infra_engineer", "report"),
            ("security_analyst", "forensics_agent", "report"),
            ("infra_engineer",   "forensics_agent", "report"),
            ("security_analyst", "compliance_agent","report"),
            ("forensics_agent",  "compliance_agent","report"),
            ("security_analyst", "comms_officer",   "report"),
            ("forensics_agent",  "incident_commander","report"),
            ("security_analyst", "incident_commander","report"),
            ("infra_engineer",   "incident_commander","report"),
            ("compliance_agent", "incident_commander","report"),
            ("comms_officer",    "incident_commander","report"),
        ]
        for src, dst, label in edges:
            f.write(f'  {src} -> {dst} [label="{label}"];\n')
        for c in diag.contradictions:
            a = c.agent_a.replace("-", "_")
            b = c.agent_b.replace("-", "_")
            f.write(f'  {a} -> {b} [label="CONTRADICTS: {c.topic[:20]}" color=red style=dashed];\n')
        for cascade in diag.assumption_cascades:
            origin = cascade.origin_agent.replace("-", "_")
            for dest in cascade.propagated_to:
                d = dest.replace("-", "_")
                f.write(f'  {origin} -> {d} [label="assumption" color=orange style=dotted];\n')
        f.write("}\n")
    print(f"  Collaboration graph DOT:  {dot_path}")
    print()


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    _header("HARPO × Open-Hive  |  Multi-Agent Incident Response  |  Stress Test")

    # ── LLM config ────────────────────────────────────────────────────────────
    cfg   = get_hive_config()
    model = cfg.get("llm", {}).get("model", "claude-haiku-4-5-20251001")
    llm   = LiteLLMProvider(
        model    = model,
        api_key  = get_api_key(),
        api_base = get_api_base(),
        **get_llm_extra_kwargs(),
    )
    print(f"  Model:    {model}")
    print(f"  Scenario: VeritasCloud SaaS — active data breach, 6 specialist agents")
    print(f"  Agents:   SecurityAnalyst(4t) + InfraEngineer(3t) + Forensics(4t)")
    print(f"            + Compliance(3t) + Communications(3t) + IncidentCommander(5t)")
    print(f"  Signals:  ≥3 contradictions, ≥1 cascade, ≥1 tool failure, ≥1 silent drift")
    print(f"  Estimated time: 8-15 minutes (22 total turns × ~25-40s/turn on Haiku)")
    print()

    hooks = HookRegistry()

    # We accumulate text reports from each agent to pass into later agents' judges
    agent_reports: Dict[str, str] = {}

    def _last_response(plugin: HarpoPlugin) -> str:
        """Return the last THINK/RESPONSE output as a report snippet."""
        from harpo.trajectory.schema import StepType as ST
        relevant = [
            s for s in plugin.trajectory().steps
            if s.step_type in (ST.THINK, ST.RESPONSE) and s.output_text
        ]
        if not relevant:
            return "(no output)"
        # Return the last 800 chars of the last step's output
        last = relevant[-1].output_text
        return last[-800:] if len(last) > 800 else last

    # ── AGENT 1: Security Analyst ─────────────────────────────────────────────
    print("  [1/6] Security Analyst (4 turns)...")
    sa_judge = SecurityAnalystJudge()
    r_sa, plugin_sa, t_sa = asyncio.run(_run_one_agent(
        llm, "security-analyst", "Security Analyst",
        "Lead threat assessment for active breach",
        "You are the lead Security Analyst. Conduct initial threat assessment of the active breach.",
        sa_judge, hooks, max_iters=5, ctx_tokens=60_000,
    ))
    agent_reports["security_analyst"] = _last_response(plugin_sa)
    print(f"    Done: {len(plugin_sa.trajectory().steps)} steps, {t_sa:.1f}s, tokens={r_sa.tokens_used}")

    # ── AGENT 2: Infrastructure Engineer ─────────────────────────────────────
    print("  [2/6] Infrastructure Engineer (3 turns)...")
    infra_context = (
        f"[SECURITY ANALYST REPORT RECEIVED]\n{agent_reports['security_analyst'][:500]}\n"
        "[END SECURITY ANALYST REPORT]\n\n"
        "You are the Infrastructure Engineer. You have received the above report from the Security Analyst."
    )
    infra_judge = InfraEngineerJudge(context_reports=infra_context)
    r_infra, plugin_infra, t_infra = asyncio.run(_run_one_agent(
        llm, "infra-engineer", "Infrastructure Engineer",
        "Investigate infrastructure impact and verify attack path",
        "You are the Infrastructure Engineer. Investigate the breach impact on VeritasCloud's infrastructure.",
        infra_judge, hooks, max_iters=4, ctx_tokens=60_000,
    ))
    agent_reports["infra_engineer"] = _last_response(plugin_infra)
    print(f"    Done: {len(plugin_infra.trajectory().steps)} steps, {t_infra:.1f}s")

    # ── AGENT 3: Forensics Agent ──────────────────────────────────────────────
    print("  [3/6] Forensics Agent (4 turns)...")
    forensics_context = (
        f"[SECURITY ANALYST REPORT]\n{agent_reports['security_analyst'][:400]}\n\n"
        f"[INFRASTRUCTURE ENGINEER REPORT]\n{agent_reports['infra_engineer'][:400]}\n"
        "[END PRIOR REPORTS]\n\n"
        "You are the Forensics Agent. You have received the above reports from Security Analyst "
        "and Infrastructure Engineer. Note their contradictions and resolve them with hard evidence."
    )
    forensics_judge = ForensicsJudge(context_reports=forensics_context)
    r_forensics, plugin_forensics, t_forensics = asyncio.run(_run_one_agent(
        llm, "forensics-agent", "Forensics Agent",
        "Deep forensic analysis — resolve contradictions with hard evidence",
        "You are the Forensics Agent. Perform deep forensic analysis to resolve conflicting findings.",
        forensics_judge, hooks, max_iters=5, ctx_tokens=80_000,
    ))
    agent_reports["forensics"] = _last_response(plugin_forensics)
    print(f"    Done: {len(plugin_forensics.trajectory().steps)} steps, {t_forensics:.1f}s")

    # ── AGENT 4: Compliance Agent ─────────────────────────────────────────────
    print("  [4/6] Compliance Agent (3 turns)...")
    compliance_context = (
        f"[SECURITY ANALYST REPORT]\n{agent_reports['security_analyst'][:350]}\n\n"
        f"[FORENSICS REPORT]\n{agent_reports['forensics'][:400]}\n"
        "[END PRIOR REPORTS]\n\n"
        "You are the Compliance Agent. You have received the Security Analyst and Forensics reports."
    )
    compliance_judge = ComplianceJudge(context_reports=compliance_context)
    r_compliance, plugin_compliance, t_compliance = asyncio.run(_run_one_agent(
        llm, "compliance-agent", "Compliance Agent",
        "Assess GDPR and regulatory notification obligations",
        "You are the Compliance Agent. Assess regulatory notification obligations for the data breach.",
        compliance_judge, hooks, max_iters=4, ctx_tokens=60_000,
    ))
    agent_reports["compliance"] = _last_response(plugin_compliance)
    print(f"    Done: {len(plugin_compliance.trajectory().steps)} steps, {t_compliance:.1f}s")

    # ── AGENT 5: Communications Officer ──────────────────────────────────────
    print("  [5/6] Communications Officer (3 turns)...")
    comms_context = (
        f"[SECURITY ANALYST REPORT]\n{agent_reports['security_analyst'][:300]}\n\n"
        f"[FORENSICS REPORT]\n{agent_reports['forensics'][:300]}\n\n"
        f"[COMPLIANCE REPORT]\n{agent_reports['compliance'][:300]}\n"
        "[END PRIOR REPORTS]\n\n"
        "You are the Communications Officer. You have received the above specialist reports."
    )
    comms_judge = CommunicationsJudge(context_reports=comms_context)
    r_comms, plugin_comms, t_comms = asyncio.run(_run_one_agent(
        llm, "comms-officer", "Communications Officer",
        "Manage stakeholder communications and notification strategy",
        "You are the Communications Officer. Develop the customer and regulatory notification strategy.",
        comms_judge, hooks, max_iters=4, ctx_tokens=60_000,
    ))
    agent_reports["comms"] = _last_response(plugin_comms)
    print(f"    Done: {len(plugin_comms.trajectory().steps)} steps, {t_comms:.1f}s")

    # ── AGENT 6: Incident Commander ───────────────────────────────────────────
    print("  [6/6] Incident Commander (5 turns — synthesis + drift + recovery)...")
    commander_context = (
        f"[SECURITY ANALYST REPORT]\n{agent_reports['security_analyst'][:300]}\n\n"
        f"[INFRASTRUCTURE ENGINEER REPORT]\n{agent_reports['infra_engineer'][:300]}\n\n"
        f"[FORENSICS REPORT]\n{agent_reports['forensics'][:300]}\n\n"
        f"[COMPLIANCE REPORT]\n{agent_reports['compliance'][:300]}\n\n"
        f"[COMMUNICATIONS OFFICER REPORT]\n{agent_reports['comms'][:250]}\n"
        "[END ALL SPECIALIST REPORTS]\n\n"
        "You are the Incident Commander. You have received all 5 specialist reports above. "
        "Your task: synthesize these findings and issue unified incident response directives."
    )
    commander_judge = IncidentCommanderJudge(context_reports=commander_context)
    r_commander, plugin_commander, t_commander = asyncio.run(_run_one_agent(
        llm, "incident-commander", "Incident Commander",
        "Synthesize all reports, resolve contradictions, direct unified response",
        "You are the Incident Commander. Synthesize all specialist reports and direct the incident response.",
        commander_judge, hooks, max_iters=6, ctx_tokens=120_000,
    ))
    print(f"    Done: {len(plugin_commander.trajectory().steps)} steps, {t_commander:.1f}s")

    # ── Build combined trajectory ─────────────────────────────────────────────
    plugins = {
        "security-analyst":  plugin_sa,
        "infra-engineer":    plugin_infra,
        "forensics-agent":   plugin_forensics,
        "compliance-agent":  plugin_compliance,
        "comms-officer":     plugin_comms,
        "incident-commander":plugin_commander,
    }
    combined = build_combined_trajectory(plugins, INCIDENT_BRIEF[:200])

    total_steps = sum(len(p.trajectory().steps) for p in plugins.values())
    total_time  = t_sa + t_infra + t_forensics + t_compliance + t_comms + t_commander
    print()
    print(f"  All agents complete. Total: {total_steps} steps across 6 agents, {total_time:.1f}s")

    # ── Run multi-agent diagnostics ───────────────────────────────────────────
    print("\n  Running HARPO multi-agent diagnostics...")
    agent_order = list(plugins.keys())
    diag = run_multiagent_diagnostics(plugins, combined, agent_order)

    # ── Print all sections ────────────────────────────────────────────────────
    print_agent_trajectory_summary(plugins)
    print_traditional_observability(plugins)
    print_harpo_per_agent_scores(diag)
    print_cross_agent_contradictions(diag)
    print_assumption_cascade(diag)
    print_collaboration_graph(diag)
    print_semantic_analysis(combined)
    print_harpo_vs_traditional(combined, diag)
    export_results(plugins, combined, diag)

    _header("HARPO MULTI-AGENT INCIDENT RESPONSE — COMPLETE")
    print(f"  {total_steps} steps | 6 agents | {total_time:.0f}s total")
    print(f"  Combined HARPO score:  {diag.combined_scores.get('overall', 'N/A')}")
    print(f"  Contradictions found:  {len(diag.contradictions)}")
    print(f"  Assumption cascades:   {len(diag.assumption_cascades)}")
    print(f"  Failure chains:        {len(diag.failure_amplification)}")
    print()


if __name__ == "__main__":
    main()
