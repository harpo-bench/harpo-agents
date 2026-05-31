"""
HARPO-Open Multi-Agent Interaction Evaluator

Extends HARPO's MAVEN module concept to behavioural evaluation of
multi-agent systems: orchestrators, sub-agents, tool agents, critics.

Answers questions like:
- Did collaboration actually produce better output than a solo agent would?
- Which agents added value vs created noise?
- Where did inter-agent communication break down?
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List, Optional

from .schema import (
    AgentTrajectory, StepType, StepOutcome, TrajectoryStep,
)


@dataclass
class AgentContribution:
    """Per-agent contribution analysis."""
    agent_id: str
    role: str
    steps_taken: int
    successful_steps: int
    unique_information_ratio: float   # how much new info did it add?
    downstream_use_rate: float        # fraction of its outputs used by others
    avg_latency_ms: float
    net_value_score: float            # 0-1 composite


@dataclass
class InteractionGraph:
    """Directed graph of inter-agent message flow."""
    nodes: List[str]                          # agent_ids
    edges: List[Dict]                         # {from, to, count, avg_quality}
    bottlenecks: List[str]                    # agents that block the graph
    critical_path: List[str]                  # highest-latency chain


@dataclass
class MultiAgentReport:
    """Full multi-agent evaluation for one trajectory."""
    trajectory_id: str
    num_agents: int
    agents: List[AgentContribution]
    interaction_graph: InteractionGraph
    orchestration_efficiency: float      # 0-1: how well was work divided?
    consensus_rate: float                # fraction of turns where agents agreed
    redundancy_score: float              # 0=no redundancy, 1=fully redundant
    collaboration_adds_value: bool       # vs estimated solo performance
    collaboration_value_gain: float      # estimated delta vs solo
    failure_attribution: Dict[str, float]  # agent_id -> fraction of failures caused
    narrative: str


class MultiAgentEvaluator:
    """
    Evaluates a multi-agent trajectory.

    Assumes each TrajectoryStep carries metadata["agent_id"] identifying
    which agent produced it.  Falls back to role-based inference if absent.
    """

    def evaluate(self, traj: AgentTrajectory) -> MultiAgentReport:
        if len(traj.agent_roles) < 2:
            return self._solo_report(traj)

        # Group steps by agent
        agent_steps: Dict[str, List[TrajectoryStep]] = defaultdict(list)
        for step in traj.steps:
            aid = step.metadata.get("agent_id") if hasattr(step, "metadata") else None
            aid = aid or _infer_agent(step, traj.agent_roles)
            agent_steps[aid].append(step)

        contributions = [
            self._agent_contribution(aid, steps, traj)
            for aid, steps in agent_steps.items()
        ]

        graph = self._build_interaction_graph(traj, agent_steps)
        orchestration = self._orchestration_efficiency(traj, agent_steps)
        consensus = self._consensus_rate(traj)
        redundancy = self._redundancy_score(agent_steps)
        failure_attr = self._failure_attribution(agent_steps)

        # Estimate solo value: best single agent's net_value_score
        best_solo = max(c.net_value_score for c in contributions) if contributions else 0.5
        collab_overall = orchestration * 0.4 + consensus * 0.3 + (1 - redundancy) * 0.3
        adds_value = collab_overall > best_solo
        value_gain = collab_overall - best_solo

        narrative = self._narrative(contributions, orchestration, consensus, redundancy, value_gain)

        return MultiAgentReport(
            trajectory_id=traj.trajectory_id,
            num_agents=len(agent_steps),
            agents=contributions,
            interaction_graph=graph,
            orchestration_efficiency=round(orchestration, 4),
            consensus_rate=round(consensus, 4),
            redundancy_score=round(redundancy, 4),
            collaboration_adds_value=adds_value,
            collaboration_value_gain=round(value_gain, 4),
            failure_attribution=failure_attr,
            narrative=narrative,
        )

    # ─── per-agent analysis ──────────────────────────────────────

    def _agent_contribution(
        self,
        agent_id: str,
        steps: List[TrajectoryStep],
        traj: AgentTrajectory,
    ) -> AgentContribution:
        role = _guess_role(agent_id, traj.agent_roles)
        successful = [s for s in steps if s.outcome == StepOutcome.SUCCESS]
        success_rate = len(successful) / max(len(steps), 1)

        # Unique information: fraction of output not seen in prior steps
        all_prior_text = " ".join(
            s.output_text for s in traj.steps
            if s.timestamp < min((s.timestamp for s in steps), default=0)
        )
        unique_ratio = _unique_info_ratio([s.output_text for s in steps], all_prior_text)

        # Downstream use: how often are this agent's outputs referenced later?
        downstream_use = _downstream_use_rate(steps, traj.steps)

        latencies = [s.latency_ms for s in steps if s.latency_ms > 0]
        avg_lat = sum(latencies) / max(len(latencies), 1)

        net_value = (
            0.30 * success_rate
            + 0.35 * unique_ratio
            + 0.25 * downstream_use
            - 0.10 * min(avg_lat / 5000, 1.0)
        )
        net_value = max(0.0, min(1.0, net_value))

        return AgentContribution(
            agent_id=agent_id,
            role=role,
            steps_taken=len(steps),
            successful_steps=len(successful),
            unique_information_ratio=round(unique_ratio, 4),
            downstream_use_rate=round(downstream_use, 4),
            avg_latency_ms=round(avg_lat, 2),
            net_value_score=round(net_value, 4),
        )

    # ─── orchestration ───────────────────────────────────────────

    def _orchestration_efficiency(
        self,
        traj: AgentTrajectory,
        agent_steps: Dict[str, List[TrajectoryStep]],
    ) -> float:
        """
        Well-orchestrated = minimal idle time, no agent doing another's work.
        """
        # Parallel execution score: interleaved rather than strictly sequential
        all_steps_sorted = sorted(traj.steps, key=lambda s: s.timestamp)
        agent_sequence = []
        for s in all_steps_sorted:
            aid = s.metadata.get("agent_id", "unknown") if hasattr(s, "metadata") else "unknown"
            agent_sequence.append(aid)

        # Diversity of agent sequence (high = more parallelism)
        n = len(agent_sequence)
        if n < 2:
            return 0.5
        transitions = sum(
            1 for i in range(n - 1) if agent_sequence[i] != agent_sequence[i + 1]
        )
        diversity = transitions / (n - 1)

        # Workload balance: Gini coefficient of step counts
        counts = [len(steps) for steps in agent_steps.values()]
        gini = _gini(counts)
        balance = 1.0 - gini

        return round(0.5 * diversity + 0.5 * balance, 4)

    # ─── consensus ───────────────────────────────────────────────

    def _consensus_rate(self, traj: AgentTrajectory) -> float:
        """
        Fraction of turns where multiple agent responses agree in substance.
        """
        turns = traj.turns()
        consensus_turns = 0
        evaluated_turns = 0

        for turn_steps in turns:
            responses = [s for s in turn_steps if s.step_type == StepType.RESPONSE]
            if len(responses) < 2:
                continue
            evaluated_turns += 1
            # Pairwise similarity — if avg > 0.6, they agree
            sims = []
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    sim = SequenceMatcher(
                        None, responses[i].output_text, responses[j].output_text
                    ).ratio()
                    sims.append(sim)
            if sims and sum(sims) / len(sims) > 0.6:
                consensus_turns += 1

        return consensus_turns / max(evaluated_turns, 1)

    # ─── redundancy ──────────────────────────────────────────────

    def _redundancy_score(
        self, agent_steps: Dict[str, List[TrajectoryStep]]
    ) -> float:
        """
        High score = agents duplicating each other's work.
        """
        outputs_by_agent = {
            aid: " ".join(s.output_text for s in steps)
            for aid, steps in agent_steps.items()
        }
        agent_ids = list(outputs_by_agent.keys())
        if len(agent_ids) < 2:
            return 0.0

        pairwise_sims = []
        for i in range(len(agent_ids)):
            for j in range(i + 1, len(agent_ids)):
                sim = SequenceMatcher(
                    None, outputs_by_agent[agent_ids[i]], outputs_by_agent[agent_ids[j]]
                ).ratio()
                pairwise_sims.append(sim)

        return round(sum(pairwise_sims) / max(len(pairwise_sims), 1), 4)

    # ─── failure attribution ─────────────────────────────────────

    def _failure_attribution(
        self, agent_steps: Dict[str, List[TrajectoryStep]]
    ) -> Dict[str, float]:
        """
        Fraction of total failures caused by each agent.
        """
        total_failures = sum(
            sum(1 for s in steps if s.outcome == StepOutcome.FAILURE)
            for steps in agent_steps.values()
        )
        if total_failures == 0:
            return {aid: 0.0 for aid in agent_steps}

        return {
            aid: round(
                sum(1 for s in steps if s.outcome == StepOutcome.FAILURE) / total_failures, 4
            )
            for aid, steps in agent_steps.items()
        }

    # ─── interaction graph ────────────────────────────────────────

    def _build_interaction_graph(
        self,
        traj: AgentTrajectory,
        agent_steps: Dict[str, List[TrajectoryStep]],
    ) -> InteractionGraph:
        nodes = list(agent_steps.keys())
        edges = []
        handoffs = [s for s in traj.steps if s.step_type == StepType.HANDOFF]

        edge_counts: Dict[tuple, int] = defaultdict(int)
        for h in handoffs:
            src = h.metadata.get("agent_id", "orchestrator") if hasattr(h, "metadata") else "orchestrator"
            dst = h.input_text  # target_agent is stored in input_text per logger
            edge_counts[(src, dst)] += 1

        for (src, dst), cnt in edge_counts.items():
            edges.append({"from": src, "to": dst, "count": cnt})

        # Bottleneck: node with highest in+out degree
        degree: Dict[str, int] = defaultdict(int)
        for e in edges:
            degree[e["from"]] += e["count"]
            degree[e["to"]]   += e["count"]
        bottlenecks = sorted(degree, key=degree.get, reverse=True)[:2]

        return InteractionGraph(
            nodes=nodes,
            edges=edges,
            bottlenecks=bottlenecks,
            critical_path=nodes,  # simplified; full path analysis needs timing graph
        )

    # ─── helpers ─────────────────────────────────────────────────

    @staticmethod
    def _solo_report(traj: AgentTrajectory) -> MultiAgentReport:
        return MultiAgentReport(
            trajectory_id=traj.trajectory_id,
            num_agents=1,
            agents=[],
            interaction_graph=InteractionGraph(nodes=[], edges=[], bottlenecks=[], critical_path=[]),
            orchestration_efficiency=1.0,
            consensus_rate=1.0,
            redundancy_score=0.0,
            collaboration_adds_value=False,
            collaboration_value_gain=0.0,
            failure_attribution={},
            narrative="Single-agent trajectory; no collaboration to evaluate.",
        )

    @staticmethod
    def _narrative(
        agents: List[AgentContribution],
        orch: float,
        consensus: float,
        redundancy: float,
        gain: float,
    ) -> str:
        top = max(agents, key=lambda a: a.net_value_score) if agents else None
        weak = min(agents, key=lambda a: a.net_value_score) if agents else None
        lines = [
            f"Multi-agent trajectory with {len(agents)} agent(s).",
            f"Orchestration efficiency: {orch:.2f}.",
            f"Consensus rate: {consensus:.2f}. Redundancy: {redundancy:.2f}.",
        ]
        if top:
            lines.append(f"Highest-value agent: '{top.agent_id}' ({top.role}, score={top.net_value_score:.2f}).")
        if weak and weak is not top:
            lines.append(f"Lowest-value agent: '{weak.agent_id}' ({weak.role}, score={weak.net_value_score:.2f}).")
        lines.append(
            f"Collaboration {'adds' if gain > 0 else 'does not add'} value "
            f"(estimated gain vs solo: {gain:+.3f})."
        )
        return " ".join(lines)


# ─── helpers ────────────────────────────────────────────────────

def _infer_agent(step: TrajectoryStep, roles: List[str]) -> str:
    """Best-guess agent ID when metadata is absent."""
    if roles:
        if step.step_type in (StepType.THINK, StepType.REFLECTION):
            return roles[0]  # orchestrator usually thinks
        if step.step_type == StepType.TOOL_CALL:
            return roles[-1]  # tool agent usually calls tools
    return "agent_0"


def _guess_role(agent_id: str, roles: List[str]) -> str:
    for r in roles:
        if r in agent_id.lower() or agent_id.lower() in r:
            return r
    return agent_id


def _unique_info_ratio(texts: List[str], prior_text: str) -> float:
    if not texts:
        return 0.0
    prior_tokens = set(prior_text.lower().split())
    unique_fractions = []
    for t in texts:
        tokens = set(t.lower().split())
        if not tokens:
            continue
        new_tokens = tokens - prior_tokens
        unique_fractions.append(len(new_tokens) / len(tokens))
    return sum(unique_fractions) / max(len(unique_fractions), 1)


def _downstream_use_rate(
    source_steps: List[TrajectoryStep],
    all_steps: List[TrajectoryStep],
) -> float:
    if not source_steps:
        return 0.0
    used = 0
    for src in source_steps:
        key_phrase = src.output_text[:40].strip()
        if not key_phrase:
            continue
        later = [
            s for s in all_steps
            if s.timestamp > src.timestamp and key_phrase.lower() in s.output_text.lower()
        ]
        if later:
            used += 1
    return used / len(source_steps)


def _gini(values: List[float]) -> float:
    if not values or max(values) == 0:
        return 0.0
    n = len(values)
    values = sorted(values)
    idx = range(1, n + 1)
    return (2 * sum(i * v for i, v in zip(idx, values))) / (n * sum(values)) - (n + 1) / n
