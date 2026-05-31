"""Unit tests for memory causality analysis."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import pytest
from harpo.memory.memory_store import SharedMemoryStore
from harpo.memory.stale_memory_detector import build_stale_memory_report
from harpo.memory.correction_vs_recovery import build_correction_recovery_report
from harpo.memory.multi_hop_propagation import build_multi_hop_propagation_report


def _make_store_with_stale():
    store = SharedMemoryStore()
    store.write("budget", "$5M", "product-manager")
    store.update("budget", "$2M", "finance-lead")
    store.read("budget", "engineering-lead", force_version=1)  # stale read
    return store


def test_stale_read_detected():
    store = _make_store_with_stale()
    report = build_stale_memory_report(store)
    assert report.total_stale == 1
    assert report.records[0].reader_agent == "engineering-lead"
    assert report.records[0].stale_value == "$5M"


def test_correction_recovery_distinct():
    store = _make_store_with_stale()
    stale = build_stale_memory_report(store, corrections={"budget:engineering-lead": "finance-lead"})
    cr = build_correction_recovery_report(store, stale)
    assert len(cr.corrections) == 1
    # corrections and recoveries are separate objects
    assert cr.corrections is not cr.recoveries


def test_multi_hop_depth():
    store = _make_store_with_stale()
    stale = build_stale_memory_report(store)
    mh = build_multi_hop_propagation_report(store, stale)
    assert mh.max_depth >= 1
    assert mh.total_agents_affected >= 1
