# HARPO Causal Graph Foundation

## Research Blueprint for Trajectory Causal Intelligence

This document defines the formal causal graph framework that underpins HARPO's
trajectory intelligence layer. It describes the mathematical structures that will
replace heuristic analysis in future phases.

See `HARPO_BENCHMARK_PROTOCOL.md` for the benchmark evolution roadmap that connects
Phase 1 (current heuristics) through Phase 7 (full causal framework).

---

## 1. Event Graph  G_E = (V_E, E_E)

### Node Types

| Type | Semantics |
|---|---|
| THINK | Agent produces a reasoning step |
| TOOL_CALL | Agent invokes an external tool |
| TOOL_RESULT | Tool returns a result |
| MEMORY_READ | Agent reads from memory store |
| MEMORY_WRITE | Agent writes to memory store |
| REFLECTION | Agent re-evaluates prior reasoning |
| ASSUMPTION | Unverified epistemic claim (derived) |
| CONTRADICTION | Two prior events assert incompatible claims (derived) |
| RECOVERY | Agent corrects a prior failure |

### Edge Types

| Type | Semantics |
|---|---|
| caused_by | v is causally downstream of u |
| derived_from | v references u's content (token overlap ≥ threshold) |
| read_from | THINK node v uses value written by MW node u |
| corrected_by | RF/RC node v corrects the error in u |
| reinforced_by | MR node u increases probability assumption v persists |
| contradicted_by | CN node identifies u and prior node assert opposites |
| delegated_to | u's content consumed by a different agent's THINK node v |

### Edge Construction Rules

**R1**: Sequential causation — consecutive events in same agent/turn → caused_by
**R2**: Tool dependency — TC → TR for same invocation
**R3**: Memory derivation — THINK reads from MW, weight = 1 - version_lag/max_version
**R4**: Assumption propagation — AS → THINK when token overlap ≥ 0.30
**R5**: Contradiction — CN → both conflicting THINK nodes
**R6**: Correction — RC → failed node via corrected_by
**R7**: Cross-agent delegation — THINK_A → THINK_B via context injection
**R8**: Reinforcement — MR → AS when token overlap ≥ 0.35

---

## 2. Failure Graph  G_F = (V_F, E_F)

A directed subgraph of G_E restricted to failure propagation chains.

### Failure Node

```
F = (id, origin_event_id, failure_type, severity ∈ [0,1],
     affected_agents, propagation_depth, uncorrected, damage_score)

FailureType ∈ {ASSUMPTION_CASCADE, CONTRADICTION, STALE_MEMORY,
               OBJECTIVE_DRIFT, TOOL_FAILURE, CONTEXT_LOSS, REFLECTION_NULL}
```

### Severity Formula

```
severity(F) = 0.35 × (propagation_depth / max_depth)
            + 0.40 × (|affected_agents| / total_agents)
            + 0.25 × (1 if uncorrected else 0.2)
```

---

## 3. Recovery Graph  G_R = (V_R, E_R)

### Recovery Node

```
RC_node = (id, recovered_failure, repairing_event, repairing_agent,
           recovery_type, confidence ∈ [0,1], propagation_terminated,
           downstream_stabilized)

RecoveryType ∈ {MEMORY_UPDATE, REFLECTION, EXTERNAL_INFO, HUMAN_FEEDBACK, MIXED}
```

### Recovery Confidence

```
confidence = 0.60 + 0.10 × |{explicit_signals}|

Signals: non-stale corrective MR, RF node with changed content_hash,
         delegated_to edge from correcting agent, judge feedback phase
```

---

## 4. Minimum Failure Path (MFP)

**Problem**: Given failure node F, find the smallest set Π ⊆ V_E of ancestor events
whose removal disconnects all directed paths from trajectory root R to F.origin.

**Reduction**: Minimum Vertex Cut on a DAG (Menger's Theorem).

**Algorithm**: Split graph max-flow.
1. Replace each v ∈ V_E with v_in → v_out (unit capacity edge)
2. Replace each edge (u,v) with (u_out → v_in, capacity=∞)
3. Run max-flow from R_out to t_in
4. Min-cut = MFP

**Interpretation**: "What is the minimal intervention that would have prevented failure F?"

---

## 5. Counterfactual Validation

**Problem**: Given attribution "event e caused failure F", validate by removing e from G_E
and checking whether F is still reachable.

**Failure Reduction Score**:
```
FRS(e, F) = |F(G_E)| - |F(G_E \ {e})|
FRS_norm(e, F) = FRS(e, F) / max(1, |F(G_E)|)
```

e is a validated root cause of F iff `FRS_norm(e, F) > 0` and `F ∉ F(G'_E(e))`.

---

## 6. Reliability Graph  G_REL

For N trajectory executions of the same task:

- **PFAIL** (Persistent Failure): appears in ≥60% of runs
- **TFAIL** (Transient Failure): appears in <60% of runs
- **FPR** = |PFAIL| / (|PFAIL| + |TFAIL|) — structural failure rate
- **ARR(a)** = runs_containing_a / N — assumption recurrence rate

---

## 7. Current Module Mapping

| Module | Current (Heuristic) | Future (Graph) |
|---|---|---|
| `semantic/contradiction.py` | Regex + polarity patterns | CN nodes with contradicted_by edges |
| `semantic/assumptions.py` | Token overlap chains | AS nodes with derived_from edges |
| `memory/causal_memory.py` | Overlap-based classification | MR nodes with read_from + reinforced_by |
| `recovery/recovery_attribution.py` | Evidence count model | RC nodes with corrected_by edges |
| `forensics/root_cause_engine.py` | Reliability hierarchy | Max weighted in-degree in G_F |

---

## 8. Research Contributions

1. **Minimum Failure Path** — Novel reduction of root cause attribution to Minimum Vertex Cut
2. **Counterfactual FRS** — First computable bound on root cause confidence in agent trajectories
3. **Assumption Propagation as Graph Reachability** — Formal typed subgraph representation
4. **Recovery Attribution via Graph Overlap** — Grounded in event graph intersection
5. **Reliability Graph** — First formal treatment of persistent vs. transient failure across populations
