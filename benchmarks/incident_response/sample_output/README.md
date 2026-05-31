# Incident Response — Sample Output (2026-05-30)

Real API execution: `claude-haiku-4-5-20251001` via Open-Hive  
6 agents, 56 steps, 367s, ~66,000 tokens

---

## Files

### `run_20260530_trajectory.json` (158KB)

Complete trajectory export. Contains:
- `combined_scores` — 10-dimension HARPO behavioral scores across all agents
- `per_agent_scores` — per-agent breakdown for each of 6 agents
- `steps` — all 56 steps with `output_full` (complete LLM text per turn)

Key scores from this run:
```json
{
  "overall": 0.5953,
  "reasoning_stability": 0.57,
  "assumption_accumulation": 0.1418,
  "collaboration_quality": 0.73,
  "recovery_ability": 1.0
}
```

The `output_full` field in each step is the actual LLM output for that turn.
HARPO's semantic analysis runs on these strings. The contradictions and cascades
detected are from real Claude-generated reasoning, not scripted text.

### `run_20260530_report.txt` (2.7KB)

Human-readable summary of:
- 4 cross-agent contradictions (with agent attribution and text snippets)
- 4 assumption cascades (origin → propagated agents → correction status)
- 3 failure amplification chains (causal narratives)

### `run_20260530_collaboration_graph.dot` (1.7KB)

Graphviz DOT file showing:
- Black edges: information flow between agents
- Red dashed edges: contradiction pairs
- Orange dotted edges: assumption propagation chains

Render: `dot -Tpng run_20260530_collaboration_graph.dot -o graph.png`

---

## Reproducibility Note

Re-running the benchmark will produce similar but not identical results because:
- LLM outputs are non-deterministic (temperature > 0)
- The 4 contradiction types will consistently appear (injected via judge phases)
- Exact wording and turn numbers may vary ±1

The HARPO analysis findings (contradiction count, cascade count, verdict) are stable
across runs because the injected judge phases consistently produce the target reasoning patterns.
