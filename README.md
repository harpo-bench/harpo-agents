# HARPO-Agents

**Trajectory Intelligence for Self-Evolving Agent Systems**

> Given an agent execution trace, HARPO answers: *Why did this trajectory succeed, fail, or recover?*

---

## What HARPO-Agents Is

HARPO-Agents is a **trace-based trajectory intelligence framework**. It takes an agent execution trace as input and produces a structured causal report — not "what happened" but **why the trajectory took the path it did**.

```
Input:   Agent execution trace (any framework)
Output:  Trajectory intelligence report
```

This is not an observability dashboard. It is not a memory store. It is a **causal analysis layer** that answers questions existing tools cannot:

- Which assumption contaminated 4 downstream agents before being corrected?
- Which memory read reinforced a faulty belief for 3 turns?
- Which agent was the primary trajectory stabilizer?
- What is still unresolved at trajectory end?

## Differentiation

| Layer | Tool | Question |
|---|---|---|
| Observability | AgentOps, LangSmith, Langfuse | What happened? |
| Memory | Mem0 | What was stored? |
| Evaluation | TRACE, LLM-as-Judge | Was the output correct? |
| **Causality** | **HARPO** | **Why did it succeed, fail, or recover?** |

## Current Capabilities

- **Assumption Cascades** — detect origin, trace propagation, score damage, attribute correction
- **Contradiction Detection** — 4-pass: reversal markers, plan flips, negation flips, stance reversals
- **Memory Causality** — stale read detection, multi-hop propagation, correction vs recovery distinction
- **Recovery Attribution** — credit by agent, classify cause (memory / reflection / external / human)
- **Root Cause Ranking** — combined impact: damage × propagation × repair difficulty
- **Executive Forensics Report** — 8-section postmortem narrative, not event log

## Quick Start

```python
from harpo.sdk.plugin import HarpoPlugin
from harpo.adapters.open_hive import HiveAdapter

# Attach to any Open-Hive AgentLoop (zero changes to agent code)
plugin = HarpoPlugin(agent_id="my-agent", user_intent="research task")
adapter = HiveAdapter(sink=plugin._ingest, agent_id="my-agent")

# ... run your agent ...

# After completion
traj = plugin.trajectory()
from harpo.trajectory.pipeline import TrajectoryEvaluator
from harpo.semantic.analyzer import SemanticTrajectoryAnalyzer

scores = TrajectoryEvaluator().evaluate(traj)
analysis = SemanticTrajectoryAnalyzer().analyze(traj)
print(f"Overall score: {scores.overall:.2f}")
print(analysis.causal_narrative())
```

## Installation

```bash
pip install harpo-agents                    # core (no dependencies)
pip install "harpo-agents[api]"             # + FastAPI server
```

For Open-Hive benchmarks, set:
```bash
export HIVE_CORE=/path/to/hive/core
```

## Supported Frameworks

| Framework | Status |
|---|---|
| Open-Hive | ✓ Complete |
| LangGraph | Stub (Phase 2) |
| CrewAI | Stub (Phase 2) |
| AutoGen | Stub (Phase 2) |
| OpenHands | Stub (Phase 2) |

## Benchmarks

See `benchmarks/` for validated multi-agent evaluation scenarios:

- **Incident Response** — 6 agents, cybersecurity breach, contradictions + assumption cascades
- **Product Launch Memory** — 6 agents, stale memory reads + multi-hop propagation
- **Enterprise Stress Test** — large-scale multi-agent execution

## Research

HARPO-Agents introduces a formal causal graph framework for agent trajectory analysis:

```
Trace → Event Graph → Failure Graph → Recovery Graph → Reliability Graph
                          ↓
              Minimum Failure Path (future)
              Counterfactual Validation (future)
```

See `docs/HARPO_CAUSAL_GRAPH_FOUNDATION.md` for the formal specification.

## License

MIT — see [LICENSE](LICENSE)
