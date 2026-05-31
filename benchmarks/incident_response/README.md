# Incident Response Benchmark

**Real LLM execution** — `claude-haiku-4-5-20251001` via Open-Hive  
**Duration**: ~6 minutes | **Est. tokens**: ~66,000 | **Est. cost**: ~$0.08

---

**Scenario**: VeritasCloud SaaS — active data breach with conflicting forensic signals across 6 specialist agents.

**Injected failures**: assumption cascades, cross-agent contradictions, silent drift, tool failure, GDPR deadline conflict

**Expected verdict**: PARTIALLY RESOLVED (GDPR deadline conflict unresolved)

## Agents and Injections

| Agent | Turns | Key injection |
|---|---|---|
| Security Analyst | 4 | SIEM outage (tool failure) + 03:12 UTC timeline assumption |
| Infrastructure Engineer | 3 | Contradicts SQL injection; scopes to 1 host |
| Forensics Agent | 4 | Contradicts 03:12 UTC with 21:43 UTC; expands to 2 hosts |
| Compliance Agent | 3 | Introduces 72h GDPR clock from 21:43 UTC baseline |
| Communications Officer | 3 | Builds 48h SLA from 03:12 UTC (wrong baseline) |
| Incident Commander | 5 | Turn 3: PR stakeholder pressure (silent drift injection) |

## Run

```bash
export HIVE_CORE=/path/to/hive/core
python demo_multiagent_incident_response.py
```

See [docs/HARPO_OPENHIVE_GUIDE.md](../../docs/HARPO_OPENHIVE_GUIDE.md) for full setup.

## Actual Results (Run: 2026-05-30, real API)

From `sample_output/run_20260530_trajectory.json` — a real API execution:

```
Total steps:           56  (6 agents)
Duration:              367s
Combined HARPO score:  0.5953
Contradictions:        4 cross-agent
Assumption cascades:   4 (1 unresolved: GDPR deadline)
Failure chains:        3
Verdict:               PARTIALLY RESOLVED
```

**Contradictions detected:**
1. Security Analyst (SQL injection) ↔ Infrastructure Engineer (credential theft)
2. Security Analyst (03:12 UTC) ↔ Forensics Agent (21:43 UTC — 5.5 hours earlier)
3. Infrastructure Engineer (1 host) ↔ Forensics Agent (2 hosts)
4. Compliance Agent (72h from 21:43) ↔ Communications Officer (48h from 03:12)

**Cascades:**
- 03:12 UTC timeline → 4 agents → corrected by Forensics ✓
- SQL injection vector → Incident Commander → corrected by Infra ✓
- 1 host scope → Incident Commander → corrected by Forensics ✓
- GDPR 72h window → Communications → **NOT corrected** ✗

## Sample Outputs

```
sample_output/
├── run_20260530_trajectory.json          # 56 steps, full LLM text (158KB)
├── run_20260530_report.txt              # Contradiction + cascade summary (2.7KB)
└── run_20260530_collaboration_graph.dot  # Graphviz DOT
```

Render the collaboration graph:
```bash
dot -Tpng sample_output/run_20260530_collaboration_graph.dot -o graph.png
```

## HARPO vs Traditional Observability

| | Traditional | HARPO |
|---|---|---|
| Verdict | "All 6 agents completed. No errors." | PARTIALLY RESOLVED — GDPR risk persists |
| Contradictions | Invisible | 4 detected with attribution |
| Assumption cascade | Invisible | 4 chains, 1 unresolved |
| Silent drift | Invisible | Detected in Incident Commander |
| Failure chains | Invisible | 3 documented |
