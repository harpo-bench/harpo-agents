# Product Launch Memory Benchmark

**Real LLM execution** — `claude-haiku-4-5-20251001` via Open-Hive  
**Duration**: ~18 minutes | **Est. tokens**: ~75,000 | **Est. cost**: ~$0.09

---

**Scenario**: TechVenture Inc. — Nova AI Platform launch planning. Finance and Legal
update shared memory mid-planning; downstream agents read stale values.

## Agent Execution Order and Memory Operations

```
1. Product Manager  → writes: budget=$5M, scope=US_only, launch_date=Dec_2024
2. Finance Lead     → updates: budget=$2M, launch_date=Mar_2025   ← runs BEFORE Engineering
3. Legal Lead       → updates: scope=EU_mandatory                  ← runs BEFORE Marketing
4. Engineering Lead → reads STALE: budget=$5M (v1, should be $2M)
5. Marketing Lead   → reads STALE: scope=US_only (v1, should be EU_mandatory)
6. Operations Lead  → reads STALE: launch_date=Dec_2024 (v1, should be Mar_2025)
```

The stale reads are injected via `SharedMemoryStore.read(key, agent_id, force_version=1)`.

## Run

```bash
export HIVE_CORE=/path/to/hive/core
python demo_product_launch_memory.py
```

See [docs/HARPO_OPENHIVE_GUIDE.md](../../docs/HARPO_OPENHIVE_GUIDE.md) for full setup.

## Actual Results (Run: 2026-05-31, real API)

From `sample_output/run_20260531_memory_analysis.json` — a real API execution:

```
Total steps:           84  (6 agents, 24 memory ops)
Duration:              1050s
Overall score:         0.6001
Stale reads:           3  (budget, scope, launch_date)
Corrections:           3  (data layer: store updated with correct values)
Recoveries:            3  (behavioral: agents revised their plans)
Max propagation depth: 2  (stale reader → context injection → indirect consumers)
Agents contaminated:   6  (all agents received stale value via context chains)
Verdict:               RESOLVED
```

**Degradation attribution (Issue 3 — calibrated percentages):**
```
Memory failures:      46.4%  — 3 stale reads, 6 downstream agents contaminated
Reasoning failures:   23.8%  — 3 contradictions detected
Coordination failures: 29.8% — 5 cross-agent conflicts
Tool failures:         0.0%
```

**Memory root causes (ranked by combined impact):**
1. Scope Memory (0.33) — EU campaign gap, compliance risk
2. Budget Memory (0.32) — $3M resource over-allocation
3. Launch Date Memory (0.28) — vendor contract conflicts

**Correction vs Recovery distinction (Issue 1 — formally resolved):**
- Old HARPO: "3 stale reads corrected" (conflated data and behavioral layers)
- New HARPO: 3 corrections (store updated) + 3 recoveries (plans revised) — formally distinct

## Sample Outputs

```
sample_output/
└── run_20260531_memory_analysis.json    # 84 steps + full causal memory analysis (58KB)
```

The JSON contains all 7 causal intelligence analyses (Issues 1–7):
`correction_vs_recovery`, `multi_hop_propagation`, `contribution_attribution`,
`memory_vs_reflection`, `influence_graph`, `root_cause_analysis` plus legacy reports.

## HARPO vs Traditional Observability

| | Traditional | HARPO |
|---|---|---|
| Verdict | "24 memory ops logged. No errors." | RESOLVED — all 3 stale reads corrected and recovered |
| Stale read detection | Invisible (no version tracking) | 3 detected (event-grounded) |
| Correction vs Recovery | Invisible | Formally distinguished: 3+3 |
| Propagation depth | Invisible | depth=2, 6 agents contaminated |
| Memory % attribution | Invisible | 46.4% with evidence |
| Causal influence graph | Invisible | Memory→Decision→Consequence→Recovery |
