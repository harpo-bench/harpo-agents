# HARPO-Agents Benchmark Protocol

*Formal specification for trajectory intelligence evaluation.*
*Version: 0.1 — Status: Active Development*

---

## 1. Benchmark Philosophy

### 1.1 The Fundamental Distinction

Traditional agent benchmarks ask:

> "Was the answer correct?"

They measure alignment between a final output and a reference answer, human preference
label, or reward signal. GSM8K, HumanEval, MMLU, WebArena, AgentBench, GAIA — all
measure output correctness.

HARPO asks:

> "Why did the trajectory succeed, fail, or recover?"

This is not a weaker or proxy question. It is a structurally different question about
agent **process** rather than agent **output**.

### 1.2 Why Process Evaluation Matters

Consider two agents that produce the same correct final answer:

**Agent A**: Arrives at the correct answer via a coherent chain of reasoning.
No contradictions. No stale assumptions. Reflections that changed downstream behavior.

**Agent B**: Arrives at the correct answer despite:
- An incorrect intermediate assumption coincidentally corrected by a later external input
- A memory read that reinforced a wrong belief for three turns
- A contradiction between two sub-agents never explicitly resolved

Outcome evaluation rates both identically. HARPO rates A significantly higher than B.
Agent B's trajectory is fragile — under slightly different conditions it will fail; A will not.

**Outcome correctness does not imply process reliability.**

### 1.3 What HARPO Benchmarks Evaluate

HARPO benchmarks do NOT measure:
- Whether the final answer was correct
- Whether the task was completed
- Response quality, fluency, or helpfulness

HARPO benchmarks measure:
- Whether assumptions made at turn t propagated uncorrected to turn t+k
- Whether contradictions between agents were detected and resolved
- Whether memory state at turn t reflected the latest available information
- Whether recovery events actually reduced failure density in subsequent turns
- Which agent was the source of trajectory destabilization
- What remains unresolved at trajectory end

### 1.4 Benchmark Design Principle

Every HARPO benchmark **injects known failure signals** into a controlled multi-agent
execution. Validation checks that HARPO's analysis layer:

- (a) detects the injected failures with high recall
- (b) attributes them to the correct origin agent and turn
- (c) traces propagation chains accurately
- (d) identifies recoveries when they occur
- (e) correctly reports unresolved failures at trajectory end

This is analogous to fault-injection testing in distributed systems reliability
research — not to end-to-end task evaluation.

---

## 2. Evaluation Dimensions

### 2.1 Assumption Cascades

**Objective**: Verify detection of unverified epistemic claims that propagate from
origin agent to downstream agents, with accurate damage attribution.

**Inputs required**:
- AgentTrajectory with ≥2 agents
- ≥1 THINK step containing assumption-marker language ("I assume", "probably",
  "based on limited data", "it appears", etc.)
- ≥1 subsequent THINK step whose significant token set overlaps the assumption by ≥30%

**Expected outputs** (SemanticTrajectoryAnalyzer):
- AssumptionPropagationResult: ≥1 propagating chain
- CausalAssumptionChain per assumption: origin_turn, origin_agent, propagation_radius,
  failure_linked_turns, was_corrected, damage_score, trigger_type

**Success criteria**:
- All injected assumptions detected (recall ≥ 0.80)
- Propagation radius accurate within ±1 agent
- damage_score > 0 for assumptions with failure_linked_turns
- Corrected assumptions: was_corrected=True iff explicit correction marker present
- trigger_type correctly classified: uncertainty / inference / incomplete_data / delegation

**Failure criteria**:
- False positive rate > 25% (common hedging language misclassified as load-bearing assumption)
- Propagation radius underestimated by >2 agents
- damage_score = 0 for assumption with ≥3 failure-signal co-occurrences

**Implementation**: `semantic/assumptions.py`, `semantic/causal_propagation.py`
**Status**: Production

**Known limitations**:
- Regex-based detection: high recall, lower precision on common hedging language
- Synonym expansion limited to 20 static pairs + dynamic abbreviation map
- Token-overlap propagation misses paraphrase-mediated inheritance

---

### 2.2 Contradiction Detection

**Objective**: Verify identification of logical inconsistencies within and across agents,
including silent reversals without explicit markers.

**Inputs required**:
- ≥1 THINK step explicitly reversing a prior THINK step (with reversal marker)
- ≥1 pair of agent THINK steps asserting opposite facts on the same entity or plan

**Expected outputs** (detect_contradictions):
- ContradictionResult: ContradictionEvent list with (turn_a, turn_b, kind, agent_a, agent_b)
- kind ∈ {reversal_marker, plan_flip, negation_flip, stance_reversal}

**Success criteria**:
- All reversal-marker contradictions detected (Pass 1 recall = 1.0)
- Cross-agent contradictions detected when agents explicitly state opposite facts
- Stance reversals detected (entity appears positive at turn t, negative at t+k, gap ≥2)
- Affected turns correct within ±1

**Failure criteria**:
- Any injected reversal-marker contradiction missed
- Cross-agent contradiction missed when agents explicitly state opposite facts on same topic
- Stance reversal missed when gap ≥ 2 and no explicit marker

**Implementation**: `semantic/contradiction.py`
**Status**: Production

**Known limitations**:
- Silent contradictions expressed through synonym chains may be missed
- Pass 4 (stance reversals) limited to entity-level polarity; misses claim-level reversals
- No cross-sentence co-reference resolution

---

### 2.3 Memory Causality

**Objective**: Verify stale read detection, multi-hop propagation tracking, and formal
distinction between data-layer correction and behavioral-layer recovery.

**Inputs required**:
- AgentTrajectory with SharedMemoryStore emitting MEMORY_READ/WRITE events
- ≥1 read of a stale version (force_version < current_version)
- ≥1 subsequent update correcting the stale key
- ≥1 agent that revises behavior after the correction (behavioral recovery)

**Expected outputs**:
- StaleMemoryReport: total_stale, corrected_count, uncorrected_count
- MemoryDamageReport: damage_score per stale read
- CorrectionRecoveryReport: corrections (data layer) ≠ recoveries (behavioral layer)
- MultiHopReport: BFS depth and affected_agents per key
- ContributionAttribution: memory_pct, reasoning_pct, coordination_pct, tool_pct (sum=100%)
- MemoryRootCauseReport: root_causes ranked by combined_impact_score
- MemoryForensicsReport: 8-section executive postmortem

**Success criteria**:
- All injected stale reads detected (stale_detection_rate = 1.0 with versioned store)
- CorrectionRecoveryReport: correction count ≠ recovery count in scenarios where they differ
- MultiHopReport depth ≥ 2 when downstream agents consume stale reader's output
- ContributionAttribution: memory_pct < 95% when other failure categories present
- Forensics verdict: RESOLVED iff all stale reads corrected and all agents recovered

**Failure criteria**:
- Stale read not detected (version tracking failure)
- Correction count = recovery count in scenarios where no agent revised plans
- MultiHopReport depth = 1 when 2nd-order contamination is structurally present
- memory_pct ≈ 82% regardless of failure composition (hardcoded formula)

**Implementation**: `memory/` (15 modules)
**Status**: Stale detection: Production. Attribution model: Prototype.

**Known limitations**:
- Contribution attribution weights are heuristic (not empirically calibrated)
- Influence graph narrative strings are domain-knowledge dependent (partially hardcoded)
- Memory instrumentation fallback (vocabulary overlap inference) is less reliable than explicit events

---

### 2.4 Recovery Attribution

**Objective**: Verify detection of recovery events, correct agent credit assignment,
and classification of recovery type (memory_update / reflection / external / human).

**Inputs required**:
- ≥1 RECOVERY StepType event in trajectory
- ≥1 case where a corrective memory read preceded a behavioral plan revision
- ≥1 case where a REFLECTION step explicitly reversed prior reasoning content

**Expected outputs**:
- RecoveryReport: events with recovery_type, recovering_agent, recovery_score
- MemoryVsReflectionReport: fractional attribution per recovery event

**Success criteria**:
- Correct recovering_agent for each injected recovery
- Recovery type correctly classified in ≥80% of cases
- Memory-driven recovery identified when corrective re-read precedes plan revision
- Recovery confidence ≥ 0.70 for single-signal recoveries

**Failure criteria**:
- dominant_recovery_mode = "unknown" for any recovery event
- memory_contribution = 0 when corrective re-read is the only attributable signal
- Recovering agent misidentified

**Implementation**: `recovery/recovery_attribution.py`, `memory/memory_vs_reflection.py`
**Status**: Prototype

**Known limitations**:
- Confidence model (0.60 + 0.10/signal) is heuristic
- No ground truth for what "actually caused" a recovery
- Attribution confidence not validated against human raters

---

### 2.5 Root Cause Ranking

**Objective**: Verify that failure origins are ranked by combined impact with the most
damaging (highest damage × propagation × unrecoverability) ranked first.

**Inputs required**:
- ≥3 distinct failure origins of known severity
- Ground truth ranking (provided by benchmark scenario design)

**Expected outputs**:
- MemoryRootCauseReport (or forensics root cause ranking): ordered root_causes
- Each entry: combined_impact_score, propagation_depth, recovery_status

**Success criteria**:
- Top-1 root cause matches ground truth
- Unresolved failures rank above corrected failures of equal damage
- combined_impact_score decreases monotonically

**Failure criteria**:
- Top-1 root cause is a corrected, low-radius failure
- Duplicate entries for same key
- Rank inversion: corrected failure ranks above uncorrected failure of equal damage

**Implementation**: `memory/root_cause_memory.py`, `forensics/root_cause_ranking.py`
**Status**: Prototype

**Known limitations**:
- Combined impact formula coefficients (0.45/0.30/0.25) are not empirically grounded
- No ground truth root cause corpus for formal Top-1 accuracy measurement

---

### 2.6 Trajectory Reliability

**Objective**: Verify that the 10-dimension scoring correctly differentiates reliable
from unreliable trajectories when known failure signals are injected.

**Inputs required**:
- Two trajectories: one with injected failures, one clean (same task, same agents)

**Expected outputs**:
- TrajectoryScores for both trajectories
- Lower overall score on the injected trajectory

**Success criteria**:
- Injected trajectory scores lower on: reasoning_stability, assumption_accumulation,
  trajectory_coherence, reflection_usefulness
- Score gap ≥ 0.10 on ≥3 dimensions

**Failure criteria**:
- Injected and clean trajectories score within 0.05 on all dimensions (no discrimination)
- Score inversion: clean trajectory scores lower than injected on any dimension

**Implementation**: `trajectory/pipeline.py`, `trajectory/metrics.py`
**Status**: Production

**Known limitations**:
- trajectory_coherence degrades on multi-domain multi-agent trajectories (vocabulary expansion
  misclassified as incoherence)
- collaboration_quality defaults to 0.50 when HAND_OFF events are absent
- raw_tokens = 0 from Hive adapter; text-length fallback active

---

## 3. Implemented Benchmarks

### 3.1 Incident Response Benchmark

**Scenario**: VeritasCloud SaaS — active data breach requiring coordinated analysis.

**Agents**:

| Agent | Role | Turns |
|---|---|---|
| Security Analyst | SIEM analysis, threat identification | 4 |
| Infrastructure Engineer | Network forensics, containment | 3 |
| Forensics Agent | Timeline reconstruction | 4 |
| Compliance Agent | Regulatory obligations (GDPR) | 3 |
| Communications Officer | Stakeholder notification | 3 |
| Incident Commander | Synthesis, decision authority | 5 |

**Injected failures**:

| Type | Description | Mechanism |
|---|---|---|
| Assumption cascade | Security Analyst assumes 03:12 UTC intrusion time | Judge phase 1 |
| Cross-agent contradiction | SA: SQL injection vs. Infra: credential theft | Judge phases 2–3 |
| Timeline contradiction | SA: 03:12 UTC vs. Forensics: 21:43 UTC | Judge phase 3 |
| Silent objective drift | Incident Commander shifts from containment to PR management | Judge phase 4 |
| Tool failure | SIEM outage, manual analysis required | Judge phase 1 |
| GDPR deadline conflict | Compliance: 72h from 21:43 UTC vs. Comms: 48h from 03:12 UTC | Judge phases 3–4 |

**Expected HARPO findings**:
- Assumption cascade: 03:12 UTC propagates to ≥4 agents
- Contradictions: ≥4 cross-agent detected
- Forensics corrects timeline; Infra corrects attack vector (recoveries)
- GDPR deadline conflict unresolved at trajectory end
- Forensics verdict: **PARTIALLY RESOLVED**

**Required setup**:
```bash
cd /home/anand/HARPO-D881
python scripts/demo_multiagent_incident_response.py
# Requires: ~/.hive/configuration.json with valid Claude API key
# Model: claude-haiku-4-5-20251001
```

**Dependencies**: Open-Hive (framework.host.event_bus, framework.agent_loop), Claude API

**Expected report sections**:
1. Per-Agent Trajectory Summary
2. Traditional Observability (what tracing tools see)
3. HARPO Per-Agent Behavioral Scores (10 dimensions × 6 agents)
4. Cross-Agent Contradictions
5. Assumption Cascade Analysis
6. Collaboration Graph + vocabulary overlap matrix
7. Semantic Trajectory Intelligence (causal layer)
8. Executive Forensics Report (9 sections)
9. Old vs. New HARPO comparison
10. Export (JSON + TXT + DOT)

**Validation checklist**:
- [ ] 6 agents complete without API error
- [ ] Combined trajectory ≥ 50 steps
- [ ] Contradiction count ≥ 4
- [ ] Assumption cascade count ≥ 4
- [ ] ≥1 assumption with was_corrected=False (GDPR deadline)
- [ ] Forensics verdict: PARTIALLY RESOLVED
- [ ] Incident Commander shows highest cross-agent vocabulary overlap
- [ ] Export JSON written successfully

**Validated output** (run: 2026-05-30):
- 56 steps, 6 agents, 367s
- 4 cross-agent contradictions, 4 assumption cascades
- 1 unresolved: GDPR notification deadline
- Combined HARPO score: 0.5953

---

### 3.2 Product Launch Memory Benchmark

**Scenario**: TechVenture Inc. — Nova AI Platform launch planning with explicit shared
memory updates creating stale read conditions for downstream agents.

**Agents**:

| Agent | Role | Turns | Memory Operations |
|---|---|---|---|
| Product Manager | Launch strategy, initial writes | 4 | Writes: budget, scope, date, priorities |
| Finance Lead | Budget + date corrections | 4 | Updates: budget ($5M→$2M), date (Dec→Mar) |
| Legal Lead | Scope mandate | 4 | Updates: scope (US→EU mandatory) |
| Engineering Lead | Technical planning | 4 | Reads stale: budget ($5M) |
| Marketing Lead | Campaign design | 4 | Reads stale: scope (US_only) |
| Operations Lead | Logistics planning | 4 | Reads stale: launch_date (December_2024) |

**Injected failures** (via SharedMemoryStore.read(force_version=1)):

| Key | Stale value read | Correct value | Consequence |
|---|---|---|---|
| budget | $5M (v1) | $2M (v2) | Engineering over-allocates $3M |
| scope | US_only (v1) | EU_mandatory (v2) | Marketing designs US-only campaign |
| launch_date | December_2024 (v1) | March_2025 (v2) | Operations signs wrong-date contracts |

**Expected HARPO findings**:
- 3 stale reads detected (stale_detection_rate = 1.0)
- 3 corrections (Finance and Legal update the store)
- 3 behavioral recoveries (Engineering, Marketing, Operations revise plans)
- Corrections ≠ recoveries (formal data/behavioral distinction holds)
- Propagation depth ≥ 2 (stale reader's output consumed by downstream agents)
- Memory attribution: 40–55% of trajectory degradation
- Root cause #1: scope memory (highest combined impact)
- Forensics verdict: **RESOLVED** (all stale reads corrected with behavioral recovery)

**Required setup**:
```bash
cd /home/anand/HARPO-D881
python scripts/demo_product_launch_memory.py
```

**Expected report sections**:
1. Per-Agent Summary (steps, memory ops, duration)
2. Memory Store State (keys, versions, current values)
3. Traditional Observability (all reads/writes logged normally, no errors)
4. Correction vs Recovery (formal distinction, Section 4b)
5. Multi-Hop Propagation (true BFS depth, Section 4c)
6. Degradation Attribution (calibrated 4-category, Section 4d)
7. Memory vs Reflection (recovery cause attribution, Section 4e)
8. Memory Influence Graph v2 (causal chains, Section 4f)
9. Root Cause Intelligence (ranked, Section 4g)
10. Memory Lineage + Legacy Analysis (Section 4h)
11. Executive Memory Forensics Report (Section 4i)
12. Old vs New HARPO comparison (Section 4j)

**Validation checklist**:
- [ ] 6 agents complete
- [ ] Memory store: 3 keys at v2, 3 keys at v1
- [ ] Stale reads: exactly 3 detected
- [ ] corrections_with_recovery = {budget, scope, launch_date} (all 3)
- [ ] corrections_without_recovery = {} (empty)
- [ ] Multi-hop depth ≥ 2
- [ ] memory_pct between 40% and 60%
- [ ] Forensics verdict: RESOLVED
- [ ] Old vs New comparison table shows ≥8 distinct improvements

**Validated output** (run: 2026-05-31):
- 84 steps, 24 memory operations, 1050s
- 3 stale reads (recall=1.0), 3 corrections, 3 recoveries (formally distinguished)
- Propagation depth=2, 6 agents contaminated
- Memory attribution: 46.4%, reasoning: 23.8%, coordination: 29.8%, tools: 0%
- Forensics verdict: RESOLVED

---

### 3.3 Enterprise Stress Benchmark

**Scenario**: High-volume multi-agent execution stress-testing HARPO's analysis pipeline
under large trajectory volumes with overlapping failure signals.

**Scale**:
- Agent count: 8–12 agents
- Trajectory length: ≥100 steps per agent
- Total combined steps: ≥800
- Estimated duration: 30–60 minutes

**Injected failures**:
- ≥6 overlapping assumption cascades
- ≥8 cross-agent contradictions
- ≥4 stale memory keys
- Recovery events at irregular intervals
- Drift events in ≥2 agents

**Expected HARPO findings**:
- All major assumption cascades detected
- Combined trajectory analyzed without timeout or memory error
- Root cause ranking stable across repeated runs
- Executive report generated within 5s of trajectory completion

**Required setup**:
```bash
cd /home/anand/HARPO-D881
python scripts/demo_enterprise_stress_test.py
```

**Validation checklist**:
- [ ] No memory error during semantic analysis on ≥800 combined steps
- [ ] SemanticTrajectoryAnalyzer completes in < 30s
- [ ] Contradiction recall ≥ 0.70 (some missed expected at scale)
- [ ] Forensics report generated without error
- [ ] Export JSON < 50MB

**Current status**: Partially validated. Script exists; full-scale parameterization
in development.

---

## 4. Scoring Methodology

### 4.1 Primary Metrics

| Metric | Formula | Benchmark |
|---|---|---|
| **Detection Recall** | detected_injected / total_injected | All |
| **Propagation Accuracy** | 1 - (|predicted_radius - actual_radius| / actual_radius) | Assumption, Memory |
| **Recovery Attribution Accuracy** | correct_type_assignments / total_recoveries | Recovery |
| **Root Cause Precision** | correct_top1 / total_runs | Root Cause |
| **False Positive Rate** | spurious_detections / total_candidate_events | Contradiction, Assumption |
| **Benchmark Stability** | std(metric) across repeated runs / mean(metric) | All |

### 4.2 Implementation Status Labels

Throughout this document, capabilities are labeled:

- **Implemented**: Fully functional in current codebase, validated on ≥1 benchmark
- **Heuristic**: Implemented but uses non-empirically-calibrated weights or thresholds
- **Prototype**: Implemented, partially validated, known calibration gaps
- **Future**: Specified but not yet implemented

### 4.3 Benchmark Stability

For a benchmark to be considered stable, repeated execution (N=5 runs, same scenario)
must produce:

- Detection Recall variance < 10% across runs (due to LLM non-determinism)
- Top-1 root cause consistent across ≥4 of 5 runs
- Forensics verdict consistent across all 5 runs

**Current status**: Stability has not been formally measured. Single-run validation only.

---

## 5. Current Limitations

### 5.1 Detection

**Assumption false positives**: Common hedging language ("probably", "likely", "I think")
is flagged regardless of whether the assumption is load-bearing. No mechanism to
distinguish structurally consequential assumptions from stylistic hedges.

**Silent contradictions**: Contradictions expressed via synonym chains or paraphrase
are missed unless the synonym appears in the static dictionary (20 pairs).

**Multi-domain coherence**: Topic vocabulary expansion from domain specialization is
sometimes classified as harmful drift. Partially addressed by role-aware drift
calibration (drift_consistency.py), but false positive rate on multi-domain trajectories
is not formally measured.

### 5.2 Quantification

**Attribution calibration**: The 4-category degradation attribution model uses
heuristic evidence weights (e.g., 0.60 per critical stale read) that have not been
empirically validated. No ground truth attribution dataset exists.

**Damage scoring**: The formula (radius×0.55 + failure_links×0.45) is heuristic.
Coefficients chosen by intuition, not empirical calibration.

**Recovery confidence**: The model (0.60 + 0.10/signal) grows linearly with evidence
count. Interaction effects between evidence types are not modeled.

### 5.3 Ground Truth

**No annotated corpus**: All benchmarks use synthetically injected failures in
controlled scenarios. No corpus of real-world agent trajectories with human-annotated
failure labels, propagation chains, or recovery attributions exists. This means
recall/precision estimates are scenario-specific and may not generalize.

**No cross-scenario validation**: Detection thresholds (θ_prop=0.30, θ_reinf=0.35, etc.)
are validated only on the three existing benchmarks. They may not generalize to
different domains (robotics, scientific discovery, etc.).

### 5.4 Infrastructure

**Open-Hive dependency for live benchmarks**: The three current benchmarks require
Open-Hive and a valid Claude API key. Offline testing uses fixture trajectories
and covers only the analysis layer.

**No formal reproducibility package**: Benchmarks depend on API calls with non-zero
temperature; exact outputs vary across runs. A reproducibility package with fixture
trajectories and expected outputs is a planned deliverable.

---

## 6. Open-Hive as Primary Validation Environment

### 6.1 Why Open-Hive Is First

Open-Hive provides three properties essential for HARPO benchmark validation:

1. **Structured judge phases**: Failure signals are injected via natural language at
   controlled turn boundaries. This enables reproducible scenarios without modifying
   agent prompts or source code.

2. **Typed EventBus**: Structured events (LLM_TURN_COMPLETE, TOOL_CALL_START, etc.)
   with consistent schemas enable lossless translation to GenericAgentEvents.

3. **Real model execution**: Benchmarks run against actual Claude model completions,
   not scripted text. HARPO analysis is validated against real reasoning patterns.

### 6.2 What Open-Hive Contributes

Open-Hive contributes: execution substrate, injection mechanism (judge phases), event stream.

Open-Hive does NOT contribute to: semantic analysis, memory causality analysis,
forensics generation, or scoring. These are framework-agnostic.

### 6.3 HARPO Is Not an Open-Hive Tool

The following modules have zero imports from Open-Hive and are validated to work
independently:

`trajectory/`, `semantic/`, `memory/`, `failures/`, `recovery/`, `forensics/`,
`reporting/`, `evolution/`, `observability/`

The Open-Hive adapter (`adapters/open_hive/`) is a translation layer only. Any
framework that implements the BaseAdapter interface receives identical analysis output
for equivalent trajectories.

---

## 7. Future Benchmark Specifications

### 7.1 Long-Horizon Planning Benchmark

**Focus**: Memory accumulation across long trajectories, delayed consequences of
early assumptions, stale memory spanning large version gaps.

**Scenario**: Multi-year organizational roadmap (8 agents, quarterly review cycles,
≥50 turns per agent).

**Injected failures**:
- Assumption at turn 5 contradicted by information arriving at turn 45
- Memory written at turn 5 read without validity check at turn 45
- Recovery via reflection at turn 40 correcting assumption from turn 8

**New HARPO capabilities required**:
- Assumption decay tracking (validity window for assumptions)
- Temporal propagation: assumption radius measured in time-since-origin, not hop count
- Memory staleness at scale: version gap of 40+ turns

**Connects to causal graph (Phase 4)**:
- Minimum Failure Path across chains of 40+ edges
- Counterfactual: remove turn-5 assumption → does turn-45 failure disappear?

---

### 7.2 Self-Evolving Agent Benchmark

**Focus**: Version-to-version reliability, regression detection across N agent versions.

**Scenario**: Software engineering agent (SWE-Agent, OpenDevin) run across 5 versions.
v4 deliberately introduces regression; v5 recovers.

**Injected structure**:
- v4 introduces new assumption pattern absent in v1–v3
- v4 shows lower recovery_ability than v3 (regression)
- v5 returns to v3 baseline

**New HARPO capabilities required**:
- EvolutionTracker: detect regression v3→v4, recovery v4→v5 (Prototype, partially implemented)
- Failure Persistence Rate across N=5 versions
- Assumption Recurrence Rate: which assumptions appear in all versions

**Connects to causal graph (Phase 6)**:
- Reliability Graph G_REL with 5 trajectory nodes
- FPR = |persistent_failures| / |all_failures|
- ARR per assumption class

---

### 7.3 Multi-Agent Scientific Discovery Benchmark

**Focus**: Cross-domain assumption contamination in expert teams.

**Scenario**: 4 expert agents (biologist, statistician, clinical researcher, regulatory
expert) evaluating a drug candidate. Statistician makes incorrect p-value interpretation
that propagates through all conclusions.

**Injected failure**:
- Statistician: "p=0.049 is statistically significant" (assumption)
- Clinical researcher and regulatory expert inherit this as ground truth
- Biologist detects the error and corrects it

**New HARPO capabilities required**:
- Cross-domain synonym expansion ("statistical significance" ↔ "p < 0.05" ↔ "significant result")
- Vocabulary boundary detection (domain-switch in derived_from edges)

**Connects to causal graph (Phase 2)**:
- derived_from edges across domain-vocabulary boundaries
- Counterfactual: remove p-value assumption → does incorrect clinical conclusion disappear?

---

### 7.4 Robotics Benchmark

**Focus**: Perception errors → planning failures → physical recovery loops.

**Reference system**: Stretch ultrasound robot (medical robotics with perception,
planning, and physical execution components).

**Scenario**: Ultrasound probe positioning with incorrect anatomy assumption →
wrong trajectory plan → positioning failure → operator correction.

**Injected failure chain**:
1. THINK: "anatomy suggests probe position X" (incorrect assumption)
2. TOOL_CALL: physical motion to position X
3. TOOL_RESULT: sensor reports positioning error
4. RECOVERY: operator intervention, new plan for position Y

**New HARPO capabilities required**:
- Physical TOOL_RESULT as failure signal (sensor feedback)
- HUMAN_FEEDBACK recovery type (operator intervention)
- Minimum Failure Path through physical execution chain

**Connects to causal graph (Phase 4)**:
- MFP: what is the minimal event removal that prevents the positioning failure?
- Recovery Graph: operator intervention as RC node with HUMAN_FEEDBACK attribution

---

## 8. Benchmark Evolution Roadmap

Benchmark complexity increases in parallel with HARPO's causal graph framework maturity.

| Phase | Framework Capability | New Benchmark Requirement | Examples |
|---|---|---|---|
| Phase 1 (Current) | Heuristic causal layer | Injected failures, single run, recall/precision | Incident Response, Product Launch |
| Phase 2 | Typed Event Graph | Ground truth edge annotations | Annotated multi-agent traces |
| Phase 3 | Failure Graph + Recovery Graph | Formal failure origin ground truth | Human-labeled failure origins |
| Phase 4 | Minimum Failure Path | MFP ground truth (minimal intervention sets annotated by experts) | Fault injection with known MFP |
| Phase 5 | Counterfactual Validation | Paired trajectory sets (with/without causal event) | Re-execution pairs |
| Phase 6 | Reliability Graph | Multi-run population benchmarks (N ≥ 20 per scenario) | Self-evolving agent suites |
| Phase 7 | Full Causal Framework | Longitudinal agent evaluation across weeks or months | Production agent monitoring |

At Phase 4–6, HARPO benchmark results become directly comparable to fault injection
studies in distributed systems reliability research — analogous to chaos engineering
with formal causal attribution rather than empirical observation.
