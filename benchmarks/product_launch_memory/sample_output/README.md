# Product Launch Memory — Sample Output (2026-05-31)

Real API execution: `claude-haiku-4-5-20251001` via Open-Hive  
6 agents, 84 steps, 1050s, ~75,000 tokens

---

## Files

### `run_20260531_memory_analysis.json` (58KB)

Complete memory causality analysis export. Top-level keys:

```
benchmark              — "product_launch_memory_v2"
model                  — "claude-haiku-4-5-20251001"
total_steps            — 84
memory_ops             — 24
overall_score          — 0.6001
store_summary          — per-key version state
stale_reads            — 3 stale reads with consequences
lineage                — read-from lineage graph
damage                 — per-stale-read damage scores
recovery               — memory recovery events
propagation            — legacy depth-1 propagation
correction_vs_recovery — ISSUE 1: formal data vs behavioral distinction
multi_hop_propagation  — ISSUE 2: true BFS depth with inheritance types
contribution_attribution — ISSUE 3: memory/reasoning/coordination/tool %
memory_vs_reflection   — ISSUE 4: recovery cause attribution
influence_graph        — ISSUE 5: Memory→Decision→Agent→Consequence→Recovery
root_cause_analysis    — ISSUE 6: ranked root causes by combined impact
harpo_analysis         — causal narrative from SemanticTrajectoryAnalyzer
agents                 — per-agent step breakdown
```

Key findings from this run:

```json
{
  "stale_reads": { "total_stale": 3, "corrected_count": 3, "uncorrected_count": 0 },
  "correction_vs_recovery": {
    "correction_count": 3,
    "recovery_count": 3,
    "corrections_with_recovery": ["budget", "launch_date", "scope"]
  },
  "multi_hop_propagation": {
    "max_propagation_depth": 2,
    "total_agents_affected": 6
  },
  "contribution_attribution": {
    "memory": { "pct": 46.4 },
    "reasoning": { "pct": 23.8 },
    "coordination": { "pct": 29.8 },
    "tools": { "pct": 0.0 }
  },
  "root_cause_analysis": {
    "most_damaging": "scope",
    "deepest_propagation": "scope"
  }
}
```

---

## Verification: This Is Real LLM Output

The `agents` section in the JSON contains per-agent step breakdowns with the
actual LLM-generated planning text. For example, Engineering Lead's stale budget
response includes specific headcount numbers ("8 additional engineers", "$1.2M cloud
infrastructure") based on the $5M assumption — text that Claude generated in response
to the judge's stale memory injection. Finance Lead's corrective response explicitly
references "$2M" and explains the board constraint. These are not scripted strings.

HARPO's stale detection is event-grounded: it reads the `is_stale` flag from
`SharedMemoryStore.read(force_version=1)` — a Python-level flag, not LLM inference.
The causal analysis (which decisions were impacted, which agents recovered) is then
performed on the LLM text via token overlap heuristics.
