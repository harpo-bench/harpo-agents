# HARPO-Agents: MVP Definition and Research Vision

*Single source of truth describing what HARPO-Agents is, what it does, and where it is going.*

---

## Why HARPO Exists

Long-horizon agent trajectories fail silently.

Traditional observability tools report what happened: step counts, latencies, error codes, token volumes. They cannot report why a trajectory degraded — which assumption contaminated downstream agents, which memory operation reinforced a faulty belief, which reflection changed only vocabulary while leaving the underlying error intact.

HARPO exists to answer one question:

**Why did this trajectory succeed, fail, or recover?**

Not "what happened." Not "which step errored." Not "how many tokens were used."

WHY the trajectory took the path it did — and which specific events caused the outcome.

---

## What HARPO Is

HARPO-Agents is a **trace-based trajectory intelligence framework** for long-horizon and self-evolving agent systems.

```
Input:   Agent execution traces
Output:  Trajectory intelligence reports
```

Given any agent execution trace — from Open-Hive, LangGraph, CrewAI, AutoGen, OpenHands, or any framework with an adapter — HARPO produces a structured causal analysis identifying:

- Which assumptions originated where and contaminated which downstream agents
- Which contradictions occurred and whether they were resolved
- Which memory operations caused planning failures
- Which events contributed to recovery and whether recovery was complete
- Which agent was the primary stabilizer or destabilizer
- What the unresolved risks are at trajectory end

This is not output evaluation. This is **process evaluation** — analysis of the execution trace that produced the output, not the output itself.

---

## What HARPO Is Not

HARPO-Agents is not:

- **An observability dashboard** — It does not replace LangSmith, Langfuse, or Datadog. Those tools trace what happened. HARPO explains why it happened.
- **A memory store** — It does not manage agent memory. It analyzes what memory operations caused trajectory degradation or recovery.
- **An agent runtime** — It does not run agents. It evaluates agent execution traces.
- **An orchestration framework** — It does not coordinate agents. It evaluates how well coordination worked.
- **An LLM evaluation judge** — It does not call an external model to score outputs. All core analysis is deterministic: token overlap, pattern matching, event structure, graph reachability.

HARPO is a **causal analysis layer**. It sits above the runtime and below the developer — consuming traces, producing causal explanations.

---

## The Core MVP

### Input

An agent execution trace in one of three forms:

1. **Live stream**: GenericAgentEvents emitted by a framework adapter (Open-Hive, LangGraph, CrewAI, AutoGen) during execution
2. **JSONL log**: Post-hoc log file ingested via log reader
3. **Python objects**: AgentTrajectory constructed directly in code

### Output: Trajectory Intelligence Report

```
TRAJECTORY INTELLIGENCE REPORT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Verdict:         PARTIALLY RESOLVED
Overall Score:   0.59 / 1.00

Root Causes (ranked by combined impact):
  #1  Incorrect breach timeline assumption [CRITICAL]
      Origin: Security Analyst, Turn 1
      Propagated to: 4 agents across 2 hops
      Status: ✓ Corrected by Forensics Agent at Turn 7

  #2  Conflicting GDPR notification deadline [HIGH]
      Origin: Compliance Agent, Turn 3
      Propagated to: Communications Officer
      Status: ✗ Unresolved at trajectory end

Assumption Cascades:    4 detected  (1 unresolved)
Contradictions:         5 total     (2 cross-agent)
Memory Failures:        3 stale reads, depth=2
Recovery Events:        4 (3 memory-driven, 1 reflection-driven)

Degradation Attribution:
  Memory:       46%   (3 stale reads, 6 downstream agents contaminated)
  Reasoning:    24%   (5 contradictions, 3 uncorrected assumptions)
  Coordination: 30%   (1 unresolved deadline conflict)
  Tools:         0%

Remaining Risk:
  → GDPR notification window conflict persists.
    Compliance and Communications operated on different clock baselines.
    This ambiguity was never resolved.
```

### Current MVP Capabilities

**Assumption Cascades**
- Detects where unverified assumptions originated (12 marker patterns + synonym expansion)
- Tracks which agents inherited them (token overlap ≥ 30%)
- Classifies trigger type: uncertainty / inference / incomplete_data / delegation
- Scores downstream damage from propagation radius and failure-signal co-occurrence
- Identifies whether reflection or contradiction corrected the assumption

**Contradiction Analysis**
- Pass 1: Explicit reversal markers (25 phrases: "actually", "this contradicts", etc.)
- Pass 2: Plan flip detection (binary action-verb polarity shift within ±5 steps)
- Pass 3: Negation flip detection (negation pattern across adjacent turns)
- Pass 4: Stance reversal (entity positive→negative across ≥2 turns)
- Cross-agent contradiction tracking with deduplication

**Memory Causality**
- Stale read detection via versioned SharedMemoryStore (event-grounded)
- Causal taxonomy: REINFORCEMENT / CORRECTION / STALE_REUSE / ASSUMPTION_STORAGE
- Multi-hop propagation via BFS graph traversal (true depth, not hardcoded)
- Correction vs Recovery distinction: data-layer update ≠ behavioral plan revision
- 4-category degradation attribution: memory / reasoning / coordination / tool
- Recovery cause attribution: memory_update / reflection / external / human / mixed
- Memory influence graph: Memory → Decision → Agent → Consequence → Recovery
- Root cause ranking by combined impact (damage × propagation × repair difficulty)

**Recovery Attribution**
- Recovery event detection via RECOVERY StepType + downstream failure density analysis
- Credit assignment to specific agents (strongest recoverer)
- Recovery confidence scoring from evidence signal count
- Propagation termination detection (did this recovery stop further spread?)

**Root Cause Ranking**
- Signal reliability hierarchy: contradictions > assumptions > drift > memory
- Combined impact: damage×0.45 + propagation×0.30 + repair_difficulty×0.25
- 8-section executive forensics report (postmortem format, not event log)
- Verdict: RESOLVED / PARTIALLY RESOLVED / UNRESOLVED

---

## Positioning Reference

| Layer | Tool | Question Answered |
|---|---|---|
| Observability | AgentOps, LangSmith, Langfuse | What happened? |
| Memory | Mem0 | What was stored? |
| Evaluation | TRACE, LLM-as-Judge | Was the output correct? |
| **Causality** | **HARPO** | **Why did it succeed, fail, or recover?** |

HARPO is not a competitor to any of the above. It is a new layer that consumes their traces and produces causal explanations they cannot generate.

**The analogy**:
- AgentOps → Observability
- Mem0 → Memory
- HARPO → Trajectory Causality

---

## Current Capability Maturity

| Capability | Status | Implementation | Confidence |
|---|---|---|---|
| Assumption cascade detection | **Production** | semantic/assumptions.py + causal_propagation.py | High |
| Contradiction detection (4-pass) | **Production** | semantic/contradiction.py | High |
| Memory stale read detection | **Production** | memory/memory_store.py + stale_memory_detector.py | High (event-grounded) |
| Multi-hop propagation (BFS) | **Production** | memory/multi_hop_propagation.py | High |
| Correction vs Recovery distinction | **Production** | memory/correction_vs_recovery.py | High |
| Recovery event detection | **Production** | recovery/recovery_attribution.py | Medium |
| Semantic coherence scoring | **Production** | semantic/coherence.py | Medium |
| Reflection effectiveness | **Production** | semantic/reflection.py + reflection_impact.py | Medium |
| Degradation attribution (4-category) | **Prototype** | memory/contribution_analysis.py | Medium (heuristic weights) |
| Recovery cause attribution | **Prototype** | memory/memory_vs_reflection.py | Medium |
| Causal influence graph | **Prototype** | memory/influence_graph.py | Medium (domain-knowledge dependent) |
| Root cause ranking | **Prototype** | memory/root_cause_memory.py | Medium |
| Silent drift detection | **Prototype** | semantic/drift_analysis.py + objective_drift_v2.py | Medium |
| Evolution regression detection | **Prototype** | evolution/tracker.py | Low |
| Minimum Failure Path | **Not Implemented** | — | — |
| Counterfactual Validation | **Not Implemented** | — | — |
| Reliability Graph | **Not Implemented** | — | — |
| Long-horizon statistical bounds | **Not Implemented** | — | — |

---

## Research Roadmap

### Phase 1 — Current (Heuristic Causal Layer)

The current system identifies failure signals through token overlap, pattern matching,
and event structure. The causal conclusions are heuristic: they are plausible
attributions grounded in evidence, not formally proven causal claims.

Current system produces: "Assumption X propagated to agents Y, Z. Contradiction C
was detected between turns 4 and 7. Recovery by agent A reduced failure density."

### Phase 2 — Event Graph (Formal Causal Structure)

Define a typed directed graph G_E where:
- Nodes are typed events (THINK, TOOL_CALL, MEMORY_READ, ASSUMPTION, CONTRADICTION, RECOVERY)
- Edges are typed causal relations (caused_by, derived_from, read_from, corrected_by, reinforced_by)
- Edge weights encode causal strength (token overlap ratio, version lag, etc.)

Upgrade path: replace heuristic propagation tracking with graph reachability.
Assumption propagation becomes: does a directed path of derived_from edges exist
from AS node to THINK node?

### Phase 3 — Failure Graph + Recovery Graph

Extract from G_E:
- G_F (Failure Graph): subgraph restricted to failure origin nodes and their propagation
- G_R (Recovery Graph): nodes representing repair events with corrected_by edges

Root cause attribution becomes: which node in G_F has the highest weighted in-degree?

### Phase 4 — Minimum Failure Path

**Formal problem**: Given failure node F, find the minimum set of ancestor events
in G_E whose removal disconnects all directed paths from the trajectory root to F.

**Reduction**: Minimum Vertex Cut on a DAG (Menger's Theorem). Polynomial-time
via max-flow on the split graph.

**Interpretation**: "What is the minimal intervention that would have prevented this failure?"

### Phase 5 — Counterfactual Validation

**Formal problem**: Given root cause attribution "event E caused failure F",
validate by removing E from G_E and checking whether F is still reachable.

**Failure Reduction Score**:
```
FRS(e, F) = |failures_reachable(G_E)| - |failures_reachable(G_E \ {e})|
```

This provides the first computable bound on root cause confidence — the difference
between "plausible attribution" and "validated causal claim."

### Phase 6 — Reliability Graph

**Formal problem**: Given N trajectory executions of the same task, characterize
which failures are structural (appear in ≥60% of runs) vs. transient (appear in <60%).

**Failure Persistence Rate** (FPR): |persistent_failures| / |all_failures|
**Assumption Recurrence Rate** (ARR): runs_containing_assumption_a / N

High FPR + high ARR → structural failure mode requiring agent redesign, not monitoring.

### Long-Term Vision

```
Trace
  ↓
Event Graph (typed causal DAG)
  ↓
Failure Graph (failure subgraphs with propagation)
  ↓
Recovery Graph (repair events with attribution)
  ↓
Reliability Graph (cross-run persistence analysis)
  ↓
Minimum Failure Path (minimal intervention computation)
  ↓
Counterfactual Validation (causal claim verification)
  ↓
"Why did this class of trajectories fail — and what would prevent it?"
```

---

## Design Constraints That Must Be Preserved

1. **No external LLM required for core analysis.** All MVP capabilities are deterministic
   (token overlap, pattern matching, event structure). Optional LLM verification is
   a future research direction, never a dependency.

2. **Framework-agnostic architecture.** The analysis layer (trajectory/, semantic/,
   memory/, forensics/) has zero imports from any agent framework. Open-Hive is an
   adapter, not a dependency.

3. **Process over outcome.** HARPO evaluates reasoning quality regardless of final
   correctness. The same output can come from a reliable or unreliable process.

4. **Interpretability over accuracy.** Every score is backed by specific evidence
   (agent name, turn number, token overlap ratio). No black-box scores.

5. **Graceful degradation.** Missing signals default gracefully (not to 0.0, not
   to errors). Analyses fall back to less precise modes rather than crashing.
