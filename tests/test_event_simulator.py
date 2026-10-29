from pathlib import Path

from apex.core import load_config
from apex.policies import AlwaysLocal
from apex.workload import make_transition_trace
from replay.event_simulator import (ClosedLoopSimulator, load_local_execution_trace,
                                    stage_a_metrics)


ROOT = Path(__file__).resolve().parents[1]


def test_transition_trace_has_frozen_phases():
    trace = make_transition_trace(307)
    assert len(trace) == 200
    counts = {}
    for row in trace:
        counts[row["workload_phase"]] = counts.get(row["workload_phase"], 0) + 1
    assert counts == {"low_factual": 40, "short_burst": 40,
                      "reasoning_heavy": 60, "recovery": 20,
                      "cloud_degradation": 40}
    assert sum(x["cloud_degraded"] for x in trace) == 40


def test_stage_a_reconstructs_existing_all_local_trace():
    path = ROOT / "data/events/shadow_synthetic_bursty_n300_s7.jsonl"
    rows = load_local_execution_trace(path)
    simulated = ClosedLoopSimulator(load_config(), rows, AlwaysLocal(),
                                     threshold=0.0, seed=7).run()
    report = stage_a_metrics(rows, simulated)
    assert report["status"] == "pass"
    assert report["metrics"]["queue_wait_mae_ms"] <= 100.0
