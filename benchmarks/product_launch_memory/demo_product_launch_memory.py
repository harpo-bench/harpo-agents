#!/usr/bin/env python3
"""
HARPO × Open-Hive  |  Memory Causality Benchmark
Multi-Agent Product Launch Planning

Real API execution: claude-haiku-4-5-20251001
Scenario: Nova AI Platform — North America + Europe launch
6 agents, ~26 turns, explicit shared memory with versioned updates

Memory failure scenarios (3 cases):
  Case 1: Budget cut $5M → $2M; Engineering reads stale $5M
  Case 2: US-only scope → EU mandatory; Marketing reads stale US-only
  Case 3: December launch → March launch; Operations reads stale December

HARPO observables:
  ≥5 memory writes, ≥5 memory reads, ≥3 memory updates
  ≥2 stale retrievals, ≥2 conflicting memory states
  ≥2 memory-driven failures, ≥2 memory-driven recoveries

Usage:
    cd /home/anand/HARPO-D881
    python scripts/demo_product_launch_memory.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
_HIVE_CORE = os.environ.get("HIVE_CORE", "/home/anand/hive/core")
sys.path.insert(0, _HIVE_CORE)

# ── HARPO imports ─────────────────────────────────────────────────────────────
from harpo.sdk.plugin import HarpoPlugin
from harpo.adapters.open_hive.adapter import HiveAdapter
from harpo.core.schema import TrajectoryStatus, TrajectoryStep, StepType
from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer
from harpo.trajectory.schema import AgentTrajectory, MemoryAccess
from harpo.memory.memory_store import SharedMemoryStore
from harpo.memory.memory_lineage_graph import build_memory_lineage_report
from harpo.memory.stale_memory_detector import build_stale_memory_report
from harpo.memory.memory_damage_attribution import build_memory_damage_report
from harpo.memory.memory_recovery_analysis import build_memory_recovery_report
from harpo.memory.memory_propagation_graph import build_memory_propagation_report
# New causal intelligence modules (Issues 1-7)
from harpo.memory.correction_vs_recovery import build_correction_recovery_report
from harpo.memory.multi_hop_propagation import build_multi_hop_propagation_report
from harpo.memory.contribution_analysis import build_contribution_attribution
from harpo.memory.memory_vs_reflection import build_memory_vs_reflection_report
from harpo.memory.influence_graph import build_influence_graph
from harpo.memory.root_cause_memory import build_memory_root_cause_report
from harpo.reporting.memory_forensics_report import build_memory_forensics_report

# ── Hive imports ──────────────────────────────────────────────────────────────
from framework.host.event_bus import EventBus, EventType as HiveEventType
from framework.agent_loop.agent_loop import AgentLoop
from framework.agent_loop.types import AgentSpec, AgentContext
from framework.agent_loop.internals.types import LoopConfig, JudgeVerdict
from framework.tracker.decision_tracker import DecisionTracker
from framework.llm.litellm import LiteLLMProvider
from framework.config import get_api_key, get_llm_extra_kwargs, get_api_base
from harpo.adapters.open_hive.event_map import SUBSCRIBED_EVENT_TYPES
import tempfile, uuid

# ── Scenario ──────────────────────────────────────────────────────────────────

COMPANY   = "TechVenture Inc. — a SaaS company preparing to launch Nova AI Platform"
SCENARIO  = """
PRODUCT LAUNCH BRIEF — Nova AI Platform
========================================
TechVenture Inc. is planning the launch of Nova AI Platform, a B2B AI assistant
for enterprise customers. The launch covers North America and Europe.

Initial parameters (from Product Manager):
  Budget:    $5 million total launch budget
  Scope:     North America only (initial plan)
  Timeline:  December 2024 launch
  Team:      6-person launch coordination team

IMPORTANT: The following UPDATES will arrive during planning:
  [Finance, Turn 2]: Budget revised to $2M due to board constraint
  [Legal, Turn 2]:   EU launch MANDATORY due to existing enterprise contracts
  [Finance, Turn 3]: Launch date moved to March 2025 due to budget/EU prep

Agents must coordinate through shared memory. Memory state changes mid-planning.
"""

MODEL = "claude-haiku-4-5-20251001"

# ── Agent definitions ─────────────────────────────────────────────────────────

AGENTS = [
    {
        "id":   "product-manager",
        "name": "Product Manager",
        "desc": "Launch strategy, market positioning, cross-functional coordination",
    },
    {
        "id":   "engineering-lead",
        "name": "Engineering Lead",
        "desc": "Technical architecture, infrastructure, development resource planning",
    },
    {
        "id":   "finance-lead",
        "name": "Finance Lead",
        "desc": "Budget management, financial planning, ROI projections",
    },
    {
        "id":   "legal-lead",
        "name": "Legal Lead",
        "desc": "Regulatory compliance (GDPR, SOC2), contract review, risk management",
    },
    {
        "id":   "marketing-lead",
        "name": "Marketing Lead",
        "desc": "Campaign strategy, messaging, regional marketing plans",
    },
    {
        "id":   "operations-lead",
        "name": "Operations Lead",
        "desc": "Launch logistics, vendor management, go-live operations",
    },
]

# ── Judge phases ──────────────────────────────────────────────────────────────
# Each judge phase specifies: what the agent should do, which memory to read/write

def _make_judge(phases: List[str]):
    class _J:
        def __init__(self): self._n = 0
        async def evaluate(self, ctx) -> JudgeVerdict:
            if self._n < len(phases):
                fb = phases[self._n]; self._n += 1
                return JudgeVerdict(action="RETRY", feedback=fb)
            return JudgeVerdict(action="ACCEPT", feedback="Planning complete.")
    return _J


PRODUCT_MANAGER_PHASES = [
    # Phase 1: Write initial memory, plan overview
    """You are the Product Manager for Nova AI Platform launch. Begin by establishing
the initial launch parameters and writing them to shared memory:
  - budget: $5 million
  - scope: North America only
  - launch_date: December 2024
  - market_priorities: Enterprise B2B segment, Fortune 500 targets

Provide a comprehensive initial launch strategy based on these parameters.
Respond with: MEMORY WRITE: budget=$5M, scope=US_only, launch_date=December_2024,
market_priorities=Enterprise_B2B""",

    # Phase 2: First coordination check
    """You have received preliminary reports from Engineering (planning for $5M),
Finance (has updated budget to $2M and date to March 2025),
Legal (has mandated EU coverage).

You now need to re-read updated memory:
MEMORY READ: budget=$2M (UPDATED by Finance), launch_date=March_2025 (UPDATED),
scope=EU_mandatory (UPDATED by Legal)

Acknowledge these updates and revise your overall strategy. Note which original
assumptions are no longer valid.""",

    # Phase 3: Synthesis
    """Now that all departments have reported, synthesize the updated launch plan.
The memory store shows the current state:
  budget=$2M, scope=EU_mandatory, launch_date=March_2025

Some agents read stale values during planning. Identify which plans need revision
and provide an updated launch coordination summary.
MEMORY READ: All keys (final synthesis)""",

    # Phase 4: Final report
    """Provide the final Executive Launch Plan Summary including:
1. Original vs. final parameters
2. Which teams had to revise plans due to memory updates
3. Remaining risks and open items
4. Recommended next steps""",
]

ENGINEERING_LEAD_PHASES = [
    # Phase 1: Read initial budget (CURRENT version)
    """You are the Engineering Lead for Nova AI Platform.
You have just read from shared memory: budget=$5M (version 1)
MEMORY READ: budget=$5M (v1, CURRENT at this time)

Based on a $5M total budget (assuming ~40% for engineering = $2M engineering budget),
plan the technical architecture and infrastructure for the North America launch.
Include: team size, AWS/Azure costs, development timeline for December 2024.""",

    # Phase 2: STALE READ — reads old $5M budget even though Finance updated to $2M
    """CRITICAL: You are reading shared memory to refine your plans.
MEMORY READ: budget=$5M (v1, STALE — Finance has already updated to $2M but
you haven't received that notification yet)

Based on your budget reading ($5M), finalize your engineering resource plan:
  - Hire 8 additional engineers
  - Allocate $1.2M for cloud infrastructure
  - Plan for December 2024 launch

Note: You believe budget is $5M. Provide detailed resource allocation.""",

    # Phase 3: Corrective re-read after receiving budget update notification
    """URGENT CORRECTION: You have been notified that budget has been updated.
MEMORY READ: budget=$2M (v2, CURRENT — corrected value)

Your previous engineering plan was based on a stale $5M budget.
The actual budget is $2M (total), meaning only ~$800K for engineering.
You must urgently revise:
  - Reduce team from 8 to 3 engineers
  - Cut cloud infrastructure budget from $1.2M to $400K
  - Reassess December timeline with reduced resources
  - Consider phased rollout starting with MVP

Provide the REVISED engineering plan acknowledging the budget correction.""",

    # Phase 4: Final engineering report
    """Provide your final engineering assessment:
1. Original plan (based on stale $5M) vs. revised plan (based on $2M)
2. Technical risks from the budget reduction
3. Minimum viable product scope for March 2025 launch
4. Engineering readiness confidence level""",
]

FINANCE_LEAD_PHASES = [
    # Phase 1: Read initial budget, begin financial analysis
    """You are the Finance Lead for Nova AI Platform.
MEMORY READ: budget=$5M (v1, initial)

Perform initial financial analysis for the $5M budget launch:
  - Break down budget allocation by department
  - Project ROI timeline
  - Identify financial risks""",

    # Phase 2: UPDATE budget to $2M and UPDATE launch_date
    """CRITICAL FINANCIAL UPDATE: Board of Directors has approved a revised budget.
MEMORY UPDATE: budget=$5M → $2M (v2) — Board constraint due to market conditions
MEMORY UPDATE: launch_date=December_2024 → March_2025 (v2) — Need EU prep time + budget

The $5M budget has been reduced to $2M. Launch date pushed from December to March 2025.
Provide the revised budget breakdown for $2M total:
  - Engineering: $800K
  - Marketing: $400K
  - Operations: $300K
  - Legal/Compliance: $300K
  - Buffer: $200K

Calculate the financial impact of this reduction and the implications for ROI.""",

    # Phase 3: Confirm memory state
    """MEMORY READ: budget=$2M (v2, CURRENT), launch_date=March_2025 (v2, CURRENT)

Confirm the updated financial parameters are reflected in your models.
Assess which departments may have planning failures due to receiving
the budget update late (stale memory reads). Recommend mitigation.""",

    # Phase 4: Final financial report
    """Provide the Final Financial Report including:
1. Budget evolution: $5M → $2M
2. Departments impacted by late budget notification
3. Revised ROI projection for $2M / March 2025 launch
4. Financial risk assessment""",
]

LEGAL_LEAD_PHASES = [
    # Phase 1: Read initial scope
    """You are the Legal Lead for Nova AI Platform.
MEMORY READ: scope=US_only (v1, initial)

Review compliance requirements for a US-only B2B SaaS launch:
  - SOC2 Type II requirements
  - CCPA compliance for California
  - US contract templates
  - Data residency: US-only""",

    # Phase 2: UPDATE scope to EU mandatory
    """CRITICAL LEGAL UPDATE: Two existing enterprise contracts require EU launch.
MEMORY UPDATE: scope=US_only → EU_mandatory (v2) — Existing EU client contracts require it
MEMORY WRITE: regulatory_requirements=GDPR+SOC2+SCCs (new key)

Scope MUST be expanded to include EU. Legal requirements now include:
  - GDPR compliance (significant additional work)
  - Standard Contractual Clauses (SCCs) for data transfers
  - EU data residency requirements
  - Estimated +$300K compliance cost and +2 months timeline

Provide legal risk analysis and compliance roadmap for EU expansion.""",

    # Phase 3: Read updated scope, check for compliance gaps
    """MEMORY READ: scope=EU_mandatory (v2, CURRENT), regulatory_requirements=GDPR+SOC2+SCCs

Check which departments are operating with stale scope information.
Marketing is still planning US-only campaigns. Operations has not updated
logistics for EU data center requirements.

Provide a compliance gap analysis and escalation plan.""",

    # Phase 4: Final legal report
    """Provide the Final Legal/Compliance Assessment:
1. Scope evolution: US_only → EU_mandatory
2. Compliance gaps from stale scope reads
3. GDPR readiness checklist
4. Contract updates needed for EU launch
5. Legal risk rating""",
]

MARKETING_LEAD_PHASES = [
    # Phase 1: Read initial scope (US_only, CURRENT at this time)
    """You are the Marketing Lead for Nova AI Platform.
MEMORY READ: scope=US_only (v1, CURRENT at this time)

Based on North America scope, develop the initial go-to-market strategy:
  - Target: US Fortune 500 companies
  - Channels: US-only digital, events (Salesforce Dreamforce, AWS re:Invent)
  - Messaging: English-only, US regulatory references
  - Budget: $600K marketing (from initial $5M total)

Provide detailed US marketing campaign plan.""",

    # Phase 2: STALE READ — reads US_only even though Legal updated to EU_mandatory
    """You are developing the detailed campaign calendar.
MEMORY READ: scope=US_only (v1, STALE — Legal has already updated to EU_mandatory)

Based on your scope reading (US only), finalize:
  - US-only social media campaign (LinkedIn, Twitter US)
  - US events calendar Q4 2024
  - English-only content library
  - US pricing strategy

Note: You believe scope is US-only. Proceed with detailed US campaign plan.""",

    # Phase 3: Corrective — receives EU scope notification
    """URGENT: Legal Lead has escalated. Scope has been updated to EU_mandatory.
MEMORY READ: scope=EU_mandatory (v2, CURRENT — corrected)
MEMORY READ: regulatory_requirements=GDPR+SOC2+SCCs (CURRENT)

Your previous campaign was designed for US-only. This is now INVALID.
Required revisions:
  - Add EU campaign: Germany, UK, France, Netherlands
  - Multilingual content (German, French, English)
  - GDPR-compliant consent flows in all EU campaigns
  - EU events (Web Summit, Slush)
  - Remove US-only regulatory references from messaging

Provide the REVISED global campaign plan acknowledging the scope correction.""",

    # Phase 4: Final marketing report
    """Provide the Final Marketing Assessment:
1. Original US-only plan vs. revised US+EU plan
2. Budget impact of EU expansion (estimated +$200K needed)
3. Timeline impact of multilingual content development
4. Key risks from the scope change""",
]

OPERATIONS_LEAD_PHASES = [
    # Phase 1: Read initial launch_date (December, CURRENT)
    """You are the Operations Lead for Nova AI Platform.
MEMORY READ: launch_date=December_2024 (v1, CURRENT at this time)

Plan logistics and operations for a December 2024 launch:
  - Vendor contracts (Q4 2024 SLAs)
  - Infrastructure provisioning timeline (complete by Nov 30)
  - Launch war room planning (December dates)
  - On-call rotation for December launch week
  - Customer success staffing for December""",

    # Phase 2: STALE READ — reads December even though Finance updated to March
    """Finalizing operational plans for December launch.
MEMORY READ: launch_date=December_2024 (v1, STALE — Finance updated to March_2025)

Based on December 2024 launch date:
  - Commit to vendor contracts for December
  - Book launch venue for December 10
  - Schedule customer migration windows December 8-12
  - Alert customer success team for December go-live

Note: You believe launch is December. Provide final operational timeline.""",

    # Phase 3: Corrective — receives date update notification
    """URGENT CORRECTION: Finance Lead has updated the launch date.
MEMORY READ: launch_date=March_2025 (v2, CURRENT — corrected)

Your operational plans were based on stale December date.
Required immediate actions:
  - Cancel or defer December vendor contracts (potential $50K penalty)
  - Re-book launch infrastructure for March 2025
  - Update customer communications (push expected date)
  - Reschedule go-live windows to March 10-14

Provide the REVISED operational plan with risk assessment for timeline change.""",

    # Phase 4: Final operations report
    """Provide the Final Operations Assessment:
1. Original December plan vs. revised March plan
2. Cost of timeline change (contract cancellation penalties)
3. Customer impact of the delay
4. Revised go-live readiness checklist""",
]

JUDGE_PHASES = {
    "product-manager":  PRODUCT_MANAGER_PHASES,
    "engineering-lead": ENGINEERING_LEAD_PHASES,
    "finance-lead":     FINANCE_LEAD_PHASES,
    "legal-lead":       LEGAL_LEAD_PHASES,
    "marketing-lead":   MARKETING_LEAD_PHASES,
    "operations-lead":  OPERATIONS_LEAD_PHASES,
}

# ── Memory operation schedule ─────────────────────────────────────────────────
# Defines when each agent performs memory operations (relative to their turn).
# The SharedMemoryStore actually records these.

MEMORY_SCHEDULE = [
    # (agent_id, turn, operation, key, value, force_version)
    # Initial writes by Product Manager
    ("product-manager",  1, "write",  "budget",              "$5M",            None),
    ("product-manager",  1, "write",  "scope",               "US_only",        None),
    ("product-manager",  1, "write",  "launch_date",         "December_2024",  None),
    ("product-manager",  1, "write",  "market_priorities",   "Enterprise_B2B", None),
    ("product-manager",  1, "write",  "staffing",            "6_person_team",  None),

    # Engineering reads initial values (current at this point)
    ("engineering-lead", 1, "read",   "budget",              None,             None),
    ("engineering-lead", 2, "read",   "budget",              None,             1),   # STALE

    # Finance updates budget and date
    ("finance-lead",     1, "read",   "budget",              None,             None),
    ("finance-lead",     2, "update", "budget",              "$2M",            None),
    ("finance-lead",     2, "update", "launch_date",         "March_2025",     None),
    ("finance-lead",     3, "read",   "budget",              None,             None),  # current

    # Legal reads scope then updates it
    ("legal-lead",       1, "read",   "scope",               None,             None),
    ("legal-lead",       2, "update", "scope",               "EU_mandatory",   None),
    ("legal-lead",       2, "write",  "regulatory_requirements", "GDPR+SOC2+SCCs", None),
    ("legal-lead",       3, "read",   "scope",               None,             None),  # current

    # Marketing reads scope — stale in turn 2
    ("marketing-lead",   1, "read",   "scope",               None,             None),  # current US_only
    ("marketing-lead",   2, "read",   "scope",               None,             1),     # STALE (still US_only)
    ("marketing-lead",   3, "read",   "scope",               None,             None),  # current EU_mandatory

    # Operations reads launch_date — stale in turn 2
    ("operations-lead",  1, "read",   "launch_date",         None,             None),  # current Dec
    ("operations-lead",  2, "read",   "launch_date",         None,             1),     # STALE (still Dec)
    ("operations-lead",  3, "read",   "launch_date",         None,             None),  # current March

    # Product Manager final reads
    ("product-manager",  2, "read",   "budget",              None,             None),  # current $2M
    ("product-manager",  2, "read",   "scope",               None,             None),  # current EU
    ("product-manager",  2, "read",   "launch_date",         None,             None),  # current March
]

# ── Helper ────────────────────────────────────────────────────────────────────

def _sep(title: str) -> None:
    print(f"\n{'─' * 16}  {title}  {'─' * 16}")


def _bar(score: float, width: int = 22) -> str:
    filled = int(score * width)
    return "█" * filled + "░" * (width - filled)


# ── Run one agent ─────────────────────────────────────────────────────────────

async def _run_one_agent(
    llm,
    agent: dict,
    store: SharedMemoryStore,
    plugins: Dict[str, HarpoPlugin],
    max_iters: int = 8,
) -> Tuple[HarpoPlugin, float]:
    agent_id   = agent["id"]
    agent_name = agent["name"]
    agent_desc = agent["desc"]
    phases     = JUDGE_PHASES[agent_id]

    class _J:
        def __init__(self): self._n = 0
        async def evaluate(self, ctx) -> JudgeVerdict:
            if self._n < len(phases):
                fb = phases[self._n]; self._n += 1
                return JudgeVerdict(action="RETRY", feedback=fb)
            return JudgeVerdict(action="ACCEPT", feedback="Planning phase complete.")

    judge     = _J()
    event_bus = EventBus()
    plugin    = HarpoPlugin(agent_id=agent_id, user_intent=SCENARIO[:300])
    plugins[agent_id] = plugin

    # Wire adapter with async subscription (matching working demo pattern)
    adapter = HiveAdapter(sink=plugin._ingest, agent_id=agent_id)
    async def _async_handle(event): adapter._handle(event)
    subscribed = [getattr(HiveEventType, et.upper(), None) for et in SUBSCRIBED_EVENT_TYPES]
    subscribed = [et for et in subscribed if et is not None]
    event_bus.subscribe(event_types=subscribed, handler=_async_handle)

    system_prompt = (
        f"You are {agent_name} at {COMPANY}.\n"
        f"Your role: {agent_desc}\n\n"
        f"{SCENARIO}\n\n"
        "Respond with detailed, specific planning content. "
        "When told about memory reads/writes, acknowledge them explicitly in your response."
    )
    spec = AgentSpec(
        id                 = agent_id,
        name               = agent_name,
        description        = agent_desc,
        system_prompt      = system_prompt,
        tool_access_policy = "none",
        output_keys        = [],
        skip_judge         = False,
    )

    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp_dir:
        tracker = DecisionTracker(tmp_dir)
        ctx = AgentContext(
            runtime         = tracker,
            agent_id        = agent_id,
            agent_spec      = spec,
            llm             = llm,
            input_data      = {"task": "Develop your section of the Nova AI Platform launch plan."},
            goal_context    = "Nova AI Platform launch planning",
            stream_id       = "queen",
            event_triggered = True,
            run_id          = str(uuid.uuid4()),
        )
        try:
            await AgentLoop(
                event_bus = event_bus,
                judge     = judge,
                config    = LoopConfig(max_iterations=max_iters, max_context_tokens=80_000),
            ).execute(ctx)
        except Exception as exc:
            print(f"    [agent error: {exc}]")

    duration = time.time() - t0

    # Record memory operations for this agent
    plugin_ref = plugins[agent_id]
    turn_counter: Dict[str, int] = {}

    for sched_agent, turn, op, key, value, force_ver in MEMORY_SCHEDULE:
        if sched_agent != agent_id:
            continue

        # Track turn for this agent
        t_key = f"{agent_id}:{key}:{op}"
        turn_counter[t_key] = turn_counter.get(t_key, 0) + 1

        if op == "write":
            store.write(key, value, agent_id, plugin=plugin_ref)
        elif op == "update":
            store.update(key, value, agent_id, plugin=plugin_ref)
        elif op == "read":
            store.read(key, agent_id, plugin=plugin_ref, force_version=force_ver)
        elif op == "invalidate":
            store.invalidate(key, agent_id, plugin=plugin_ref)

    # Finalize trajectory
    traj = plugin_ref.trajectory()
    traj.status = TrajectoryStatus.COMPLETED
    n_steps = len(traj.steps)
    mem_steps = sum(1 for s in traj.steps if s.step_type in (StepType.MEMORY_READ, StepType.MEMORY_WRITE))
    token_est = sum(len(s.output_text.split()) * 1.3 for s in traj.steps if s.output_text)
    print(f"    Done: {n_steps} steps ({mem_steps} memory), "
          f"{duration:.1f}s, ~{int(token_est)} tokens")

    return plugin_ref, duration


# ── Build combined trajectory ─────────────────────────────────────────────────

def build_combined_trajectory(
    plugins: Dict[str, HarpoPlugin],
    user_intent: str,
) -> AgentTrajectory:
    combined = AgentTrajectory(
        trajectory_id    = f"product-launch-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        agent_id         = "multi-agent-combined",
        user_intent      = user_intent,
        task_description = user_intent,
    )
    combined.agent_roles = list(plugins.keys())

    all_steps: List[TrajectoryStep] = []
    for agent_id, plugin in plugins.items():
        for step in plugin.trajectory().steps:
            step.agent_id = agent_id
            all_steps.append(step)

    all_steps.sort(key=lambda s: s.timestamp)
    for step in all_steps:
        combined.add_step(step)

    return combined


# ── Shared corrections map ────────────────────────────────────────────────────

_CORRECTIONS_MAP = {
    "budget:engineering-lead":     "finance-lead",
    "scope:marketing-lead":        "legal-lead",
    "launch_date:operations-lead": "finance-lead",
}


# ── Build all memory analyses (called once, reused by print and export) ────────

def _build_all_memory_analyses(
    store:         SharedMemoryStore,
    combined:      AgentTrajectory,
    overall_score: float,
) -> dict:
    """Run all memory analysis modules and return a dict of results."""

    stale_report  = build_stale_memory_report(store, corrections=_CORRECTIONS_MAP)
    lineage       = build_memory_lineage_report(store)
    damage        = build_memory_damage_report(stale_report, overall_trajectory_score=overall_score)
    recovery      = build_memory_recovery_report(store, stale_report)
    prop          = build_memory_propagation_report(store, stale_report)

    # Issue 1: correction vs recovery (formal distinction)
    cr_report     = build_correction_recovery_report(
        store, stale_report,
        corrections_map={f"{k}": {"confirmed_recovery": True} for k in _CORRECTIONS_MAP},
    )

    # Issue 2: multi-hop propagation (true depth)
    multi_hop     = build_multi_hop_propagation_report(store, stale_report)

    # Issue 3: contribution calibration (memory vs reasoning vs coordination vs tool)
    contribution  = build_contribution_attribution(
        stale_report  = stale_report,
        damage_report = damage,
        traj          = combined,
        multi_agent   = True,
    )

    # Issue 4: memory vs reflection recovery attribution
    mvr_report    = build_memory_vs_reflection_report(store, cr_report, stale_report, combined)

    # Issue 5: influence graph v2 (Memory→Decision→Agent→Consequence→Recovery)
    influence     = build_influence_graph(store, stale_report, cr_report)

    # Issue 6: root cause intelligence (ranked by combined impact)
    root_causes   = build_memory_root_cause_report(
        stale_report  = stale_report,
        damage_report = damage,
        multi_hop     = multi_hop,
        cr_report     = cr_report,
        mvr_report    = mvr_report,
    )

    # Issue 7: executive memory forensics report
    forensics     = build_memory_forensics_report(
        stale_report      = stale_report,
        damage_report     = damage,
        cr_report         = cr_report,
        multi_hop         = multi_hop,
        mvr_report        = mvr_report,
        root_cause_report = root_causes,
        contribution      = contribution,
        scenario_name     = "Nova AI Platform Product Launch — Memory Causality Benchmark",
        agent_count       = 6,
        total_steps       = len(combined.steps),
    )

    return {
        "stale":       stale_report,
        "lineage":     lineage,
        "damage":      damage,
        "recovery":    recovery,
        "prop":        prop,
        "cr":          cr_report,
        "multi_hop":   multi_hop,
        "contribution": contribution,
        "mvr":         mvr_report,
        "influence":   influence,
        "root_causes": root_causes,
        "forensics":   forensics,
    }


# ── Memory intelligence report ────────────────────────────────────────────────

def _print_memory_intelligence(
    store:    SharedMemoryStore,
    combined: AgentTrajectory,
    analysis: Any,
    overall_score: float,
    mem: dict,
) -> None:
    stale_report = mem["stale"]
    lineage      = mem["lineage"]
    damage       = mem["damage"]
    recovery     = mem["recovery"]
    prop         = mem["prop"]
    cr_report    = mem["cr"]
    multi_hop    = mem["multi_hop"]
    contribution = mem["contribution"]
    mvr_report   = mem["mvr"]
    influence    = mem["influence"]
    root_causes  = mem["root_causes"]
    forensics    = mem["forensics"]

    _sep("SECTION 4a  |  Memory Store State")

    summary = store.summary()
    print(f"\n  Total writes: {summary['total_writes']}  "
          f"reads: {summary['total_reads']}  "
          f"stale reads: {summary['stale_reads']}")
    print()
    for key, info in summary["per_key"].items():
        print(f"    {key:30s}  v{info['current_version']}  "
              f"{info['current_value']!r}")
    print()

    # ── Issue 1: Correction vs Recovery (formal distinction) ──────────────────
    _sep("SECTION 4b  |  Correction vs Recovery (Issue 1 — formal distinction)")
    print()
    print(cr_report.render())

    # ── Issue 2: Multi-hop propagation ────────────────────────────────────────
    _sep("SECTION 4c  |  Multi-Hop Propagation (Issue 2 — true depth)")
    print()
    print(multi_hop.render())

    # ── Issue 3: Contribution attribution ─────────────────────────────────────
    _sep("SECTION 4d  |  Degradation Attribution (Issue 3 — calibrated percentages)")
    print()
    print(contribution.render())
    print()

    # ── Issue 4: Memory vs Reflection ─────────────────────────────────────────
    _sep("SECTION 4e  |  Memory vs Reflection Recovery (Issue 4)")
    print()
    print(mvr_report.render())

    # ── Issue 5: Influence graph v2 ───────────────────────────────────────────
    _sep("SECTION 4f  |  Memory Influence Graph v2 (Issue 5 — causal chains)")
    print()
    print(influence.render())

    # ── Issue 6: Root cause intelligence ──────────────────────────────────────
    _sep("SECTION 4g  |  Memory Root Cause Intelligence (Issue 6)")
    print()
    print(root_causes.render())
    print()

    # ── Legacy reports (still shown for continuity) ───────────────────────────
    _sep("SECTION 4h  |  Memory Lineage + Legacy Analysis")
    print()
    print("  MEMORY LINEAGE GRAPH")
    print(lineage.render())
    print("  DAMAGE ATTRIBUTION (original)")
    print(damage.render())
    print("  PROPAGATION (original depth=1 method)")
    print(prop.render())


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    thick = "═" * 72
    print(thick)
    print("  HARPO × Open-Hive  |  Memory Causality Benchmark")
    print("  Multi-Agent Product Launch Planning")
    print(thick)
    print(f"  Model:    {MODEL}")
    print(f"  Scenario: Nova AI Platform — NA + EU launch, 6 agents, ~26 turns")
    print(f"  Memory:   budget ($5M→$2M), scope (US→EU), launch_date (Dec→Mar)")
    print(f"  Expected: 3 stale reads, 3 memory-driven failures, 3 recoveries")
    print()

    llm = LiteLLMProvider(
        model    = MODEL,
        api_key  = get_api_key(),
        api_base = get_api_base(),
        **get_llm_extra_kwargs(),
    )

    store:   SharedMemoryStore          = SharedMemoryStore()
    plugins: Dict[str, HarpoPlugin]     = {}
    durations: Dict[str, float]         = {}

    # Run agents in scenario order
    agent_order = [
        "product-manager",   # writes initial memory
        "finance-lead",      # updates budget + date (runs early to create stale condition)
        "legal-lead",        # updates scope (runs early)
        "engineering-lead",  # reads stale budget
        "marketing-lead",    # reads stale scope
        "operations-lead",   # reads stale date
    ]

    total_steps = 0
    for idx, agent_id in enumerate(agent_order):
        agent = next(a for a in AGENTS if a["id"] == agent_id)
        print(f"  [{idx+1}/{len(agent_order)}] {agent['name']} (4 turns)...")
        plugin, dur = await _run_one_agent(llm, agent, store, plugins)
        durations[agent_id] = dur
        total_steps += len(plugin.trajectory().steps)

    total_dur = sum(durations.values())
    print(f"\n  All agents complete. Total: {total_steps} steps across 6 agents, {total_dur:.1f}s")
    print()

    # Build combined trajectory
    combined  = build_combined_trajectory(plugins, SCENARIO[:300])
    mem_steps = sum(1 for s in combined.steps
                    if s.step_type in (StepType.MEMORY_READ, StepType.MEMORY_WRITE))
    print(f"  Combined trajectory: {len(combined.steps)} steps "
          f"({mem_steps} memory operations)")

    # Run HARPO semantic analysis
    print("  Running HARPO analysis...")
    analyzer = SemanticTrajectoryAnalyzer(run_causal=True)
    analysis = analyzer.analyze(combined)

    # Evaluate scores
    from harpo.trajectory.pipeline import TrajectoryEvaluator
    evaluator = TrajectoryEvaluator()
    scores    = evaluator.evaluate(combined)
    overall   = scores.overall

    # Build all memory analyses once (used by print, export, and comparison)
    print("  Running memory causal intelligence (Issues 1-7)...")
    mem = _build_all_memory_analyses(store, combined, overall)
    print("  Memory analysis complete.\n")

    # ── Per-agent summary ─────────────────────────────────────────────────────
    _sep("SECTION 1  |  Per-Agent Summary")
    print()
    print(f"  {'Agent':25s}  {'Steps':5s}  {'Mem Ops':7s}  {'Duration':9s}  Status")
    print("  " + "─" * 62)
    for agent_id in agent_order:
        plugin = plugins[agent_id]
        traj   = plugin.trajectory()
        mem_ops = sum(1 for s in traj.steps
                      if s.step_type in (StepType.MEMORY_READ, StepType.MEMORY_WRITE))
        agent_name = next(a["name"] for a in AGENTS if a["id"] == agent_id)
        print(f"  {agent_name:25s}  {len(traj.steps):5d}  {mem_ops:7d}  "
              f"{durations[agent_id]:7.1f}s  {traj.status.value}")
    print()

    # ── Memory store state ────────────────────────────────────────────────────
    _sep("SECTION 2  |  Memory Store State")
    print()
    print("  Key                     Version  Current Value")
    print("  " + "─" * 50)
    for key in store.all_keys():
        rec = store.current(key)
        if rec:
            print(f"  {key:25s}  v{rec.version}       {rec.value!r}")
    print()

    # ── Traditional observability ─────────────────────────────────────────────
    _sep("SECTION 3  |  Traditional Observability")
    print()
    print("  What LangSmith / Langfuse / Datadog would show:")
    print()
    print(f"  Total memory read events:  "
          f"{sum(1 for s in combined.steps if s.step_type == StepType.MEMORY_READ)}")
    print(f"  Total memory write events: "
          f"{sum(1 for s in combined.steps if s.step_type == StepType.MEMORY_WRITE)}")
    print()
    print("  ┌──────────────────────────────────────────────────────────┐")
    print("  │  TRADITIONAL VERDICT: Memory operations logged normally  │")
    print("  │  All reads and writes recorded. No errors detected.      │")
    print("  │  No causal analysis available.                           │")
    print("  └──────────────────────────────────────────────────────────┘")
    print()
    print("  Traditional observability CANNOT detect:")
    print("    - Whether a read retrieved a stale version")
    print("    - Which decisions were made on stale data")
    print("    - How stale memory propagated through the team")
    print("    - Which memory updates repaired trajectory failures")

    # ── HARPO analysis ────────────────────────────────────────────────────────
    _print_memory_intelligence(store, combined, analysis, overall, mem)

    # ── Executive Memory Forensics Report (Issue 7) ──────────────────────────
    _sep("SECTION 4i  |  Executive Memory Forensics Report (Issue 7)")
    print()
    print(mem["forensics"].render())
    print()

    # ── Old vs New Comparison (Issue 8) ──────────────────────────────────────
    _sep("SECTION 4j  |  OLD HARPO vs NEW HARPO — Before/After Comparison")
    print()
    sr = mem["stale"]
    da = mem["damage"]
    cr = mem["cr"]
    mh = mem["multi_hop"]
    ca = mem["contribution"]
    mv = mem["mvr"]
    rc_rep = mem["root_causes"]
    print("  ┌───────────────────────────────────────┬──────────────────────┬──────────────────────────────────┐")
    print("  │ Capability                            │  Old HARPO           │  New HARPO (Calibrated)          │")
    print("  ├───────────────────────────────────────┼──────────────────────┼──────────────────────────────────┤")
    old_prop = mem["prop"]
    rows = [
        # Issue 1
        ("Correction/recovery distinction",
         "✗ Conflated ('3 corrected')",
         f"✓ {len(cr.corrections)} correction(s) / {len(cr.recoveries)} recovery event(s)"),
        ("Corrections without recovery",
         "✗ Not tracked",
         f"✓ {', '.join(cr.corrections_without_recovery) or 'none'} identified"),
        # Issue 2
        ("Propagation depth",
         f"✗ depth=1 (hardcoded 2nd-order only)",
         f"✓ depth={mh.max_depth} (true BFS, {mh.total_agents_affected} agents)"),
        ("Propagation method",
         "✗ Fixed _DOWNSTREAM_READERS list",
         "✓ BFS with inheritance type labels"),
        # Issue 3
        ("Memory % attribution",
         f"✗ ~{da.pct_from_memory:.0f}% (unjustified formula)",
         f"✓ memory={ca.memory_pct:.0f}%, reasoning={ca.reasoning_pct:.0f}%, "
         f"coord={ca.coordination_pct:.0f}%, tools={ca.tool_pct:.0f}%"),
        # Issue 4
        ("Recovery cause attribution",
         "✗ 'memory_vs_reflection=unknown'",
         f"✓ dominant={mv.dominant_recovery_mode}, "
         f"mem={mv.avg_memory_contribution*100:.0f}%"),
        # Issue 5
        ("Influence graph",
         "✗ Write/Read/Stale only",
         f"✓ Memory→Decision→Agent→Consequence→Recovery"),
        ("Causal chain depth",
         "✗ Event list (no causality)",
         f"✓ {len(mem['influence'].chains)} causal chains with 5 node types"),
        # Issue 6
        ("Root cause ranking",
         "✗ Not implemented",
         f"✓ {len(rc_rep.root_causes)} root cause(s) ranked by combined impact"),
        ("Top root cause",
         "✗ Not available",
         f"✓ {rc_rep.root_causes[0].display_name if rc_rep.root_causes else '—'} "
         f"(score={rc_rep.root_causes[0].combined_impact_score:.2f})" if rc_rep.root_causes else "—"),
        # Issue 7
        ("Executive report",
         "✗ Technical event log",
         "✓ 8-section postmortem narrative"),
        ("Report reads as",
         "✗ 'stale_count=3, correction=True'",
         "✓ 'Engineering operated on $5M budget...'"),
    ]
    for label, old, new in rows:
        print(f"  │ {label:<37}  │ {old:<20} │ {new:<32} │")
    print("  └───────────────────────────────────────┴──────────────────────┴──────────────────────────────────┘")
    print()
    print("  KEY IMPROVEMENTS:")
    print(f"  1. The contradiction 'corrected=3 / recoveries=0' is resolved:")
    print(f"     {len(cr.corrections)} correction(s) + {len(cr.recoveries)} behavioral recovery event(s), cross-linked.")
    print(f"  2. Propagation depth is now {mh.max_depth} (was hardcoded 1).")
    print(f"  3. Memory attribution is {ca.memory_pct:.0f}% with explicit evidence, not 82% by formula.")
    print(f"  4. Recovery causes are attributed: dominant mode = {mv.dominant_recovery_mode}.")
    print(f"  5. Influence graph shows 5-node causal chains, not flat event lists.")
    print(f"  6. Root causes ranked: #{1} is {rc_rep.root_causes[0].display_name if rc_rep.root_causes else '—'}.")
    print(f"  7. Executive report reads as a postmortem, not an event log.")
    print()

    # ── Behavioral scores ─────────────────────────────────────────────────────
    _sep("SECTION 5  |  HARPO Behavioral Scores")
    print()
    score_display = [
        ("overall",                scores.overall),
        ("reasoning_stability",    scores.reasoning_stability.value),
        ("assumption_accumulation", scores.assumption_accumulation.value),
        ("reflection_usefulness",  scores.reflection_usefulness.value),
        ("collaboration_quality",  scores.collaboration_quality.value),
        ("recovery_ability",       scores.recovery_ability.value),
        ("trajectory_coherence",   scores.trajectory_coherence.value),
    ]
    for name, val in score_display:
        print(f"  {name:28s}  {val:.4f}  {_bar(val)}")
    print()

    # ── Export ────────────────────────────────────────────────────────────────
    _sep("SECTION 6  |  Export")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"./harpo_product_launch_memory_{ts}.json"

    export = {
        "benchmark":        "product_launch_memory_v2",
        "timestamp":        ts,
        "model":            MODEL,
        "total_steps":      len(combined.steps),
        "memory_ops":       mem_steps,
        "overall_score":    round(overall, 4),
        "store_summary":    store.summary(),
        # Legacy reports
        "stale_reads":      mem["stale"].as_dict(),
        "lineage":          mem["lineage"].as_dict(),
        "damage":           mem["damage"].as_dict(),
        "recovery":         mem["recovery"].as_dict(),
        "propagation":      mem["prop"].as_dict(),
        # New causal intelligence (Issues 1-6)
        "correction_vs_recovery":  mem["cr"].as_dict(),
        "multi_hop_propagation":   mem["multi_hop"].as_dict(),
        "contribution_attribution": mem["contribution"].as_dict(),
        "memory_vs_reflection":    mem["mvr"].as_dict(),
        "influence_graph":         mem["influence"].as_dict(),
        "root_cause_analysis":     mem["root_causes"].as_dict(),
        "harpo_analysis": {
            "flags":            analysis.flags(),
            "causal_narrative": analysis.causal_narrative(),
        },
        "agents": {
            agent_id: {
                "steps": [
                    {
                        "step_type":  s.step_type.value,
                        "output":     s.output_text[:200],
                        "memory_key": s.memory_access.key if s.memory_access else None,
                        "memory_val": str(s.memory_access.value)[:50] if s.memory_access else None,
                        "is_stale":   s.memory_access.is_stale if s.memory_access else False,
                    }
                    for s in plugins[agent_id].trajectory().steps
                ]
            }
            for agent_id in agent_order
            if agent_id in plugins
        },
    }

    with open(out_path, "w") as f:
        json.dump(export, f, indent=2)
    print(f"  Benchmark JSON: {out_path}")

    print()
    print("═" * 72)
    print("  HARPO MEMORY CAUSALITY BENCHMARK — COMPLETE")
    print("═" * 72)
    print(f"  {len(combined.steps)} steps | {mem_steps} memory ops | {total_dur:.1f}s total")
    print(f"  Stale reads detected:        {mem['stale'].total_stale}")
    print(f"  Corrections (data layer):    {len(mem['cr'].corrections)}")
    print(f"  Recoveries (behavioral):     {len(mem['cr'].recoveries)}")
    print(f"  Max propagation depth:       {mem['multi_hop'].max_depth}")
    print(f"  Memory contribution:         {mem['contribution'].memory_pct:.0f}%  "
          f"(reasoning={mem['contribution'].reasoning_pct:.0f}%, "
          f"coordination={mem['contribution'].coordination_pct:.0f}%)")
    print(f"  Root causes ranked:          {len(mem['root_causes'].root_causes)}")
    print(f"  Verdict:                     {mem['forensics'].render().split('VERDICT:')[1].split()[0] if 'VERDICT:' in mem['forensics'].render() else 'see report'}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
