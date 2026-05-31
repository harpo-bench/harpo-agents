# Product Launch Memory Benchmark

6-agent product launch planning with explicit shared memory stale reads.

**Scenario**: TechVenture Inc. — Nova AI Platform launch; Finance and Legal issue memory updates that other agents miss.

**Injected failures**: 3 stale reads (budget $5M→$2M, scope US→EU, date Dec→Mar)

**Expected verdict**: RESOLVED (all 3 corrected and recovered)

## Run

```bash
export HIVE_CORE=/path/to/hive/core
python demo_product_launch_memory.py
```
