import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apex.core import load_config
from soc4ai.v4 import PROBE_TAG, make_v4_governor


def _gov():
    cfg = load_config()
    return cfg, make_v4_governor(cfg, episode_requests=300, frozen_cloud_p95_ms=3070.0)


def test_half_open_probe_is_not_charged_to_the_quota():
    _, gov = _gov()
    gov.circuit.state = "half-open"
    gov.circuit.half_open_probe_inflight = False
    d = gov._cloud_or_local("tail_hard_breach", 1.0, "full", safety=True)
    assert d.route == "cloud" and d.reason_code.endswith(PROBE_TAG)
    assert gov.budget.total_cloud == 0 and gov.budget.exempt_probes == 1
    # a second probe while one is in flight is refused
    d2 = gov._cloud_or_local("tail_hard_breach", 1.0, "full", safety=True)
    assert d2.route == "local"


def test_failed_charged_call_is_refunded_and_successful_call_is_not():
    _, gov = _gov()
    d = gov._cloud_or_local("tail_hard_breach", 1.0, "full", safety=True)
    assert d.route == "cloud" and gov.budget.total_cloud == 1
    gov.record_outcome(d, False, None)
    assert gov.budget.total_cloud == 0 and gov.budget.refunds == 1
    d_ok = gov._cloud_or_local("tail_hard_breach", 1.0, "full", safety=True)
    gov.record_outcome(d_ok, True, 800.0)
    assert gov.budget.total_cloud == 1 and gov.budget.refunds == 1


def test_failed_probe_is_not_refunded_twice():
    _, gov = _gov()
    gov.circuit.state = "half-open"
    d = gov._cloud_or_local("tail_hard_breach", 1.0, "full", safety=True)
    gov.record_outcome(d, False, None)
    assert gov.budget.total_cloud == 0 and gov.budget.refunds == 0


@pytest.fixture
def workdir():
    d = ROOT / "soc4ai_out" / "test_tmp" / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_audit_billable_counting_excludes_probes_and_failures(workdir):
    import yaml
    from soc4ai.audit import audit_run
    events = []
    for i in range(10):
        route = "cloud" if i < 5 else "local"
        reason = f"x:{PROBE_TAG}" if i == 0 else "x:budget_admitted"
        ok = i != 1
        events.append({"trace_pos": i, "decision": {"route": route, "reason_code": reason, "circuit_state": "closed"},
                       "outcome": {"executed_on": route, "ok": ok, "end_to_end_ttft_ms": 500.0}})
    run = workdir / "run.jsonl"
    run.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    run.with_suffix(".manifest.json").write_text(json.dumps({"run_id": "t", "policy": "v4_governor"}), encoding="utf-8")
    base = {"plans": {"local_first": {"guarantees": {"ttft_slo": {"threshold_ms": 3839.5, "max_violation_rate": 0.1}},
                                      "quotas": {"cloud_calls": {"max_fraction_of_episode": 0.3},
                                                 "cloud_spend": {"max_usd_study": 20.0}}}},
            "governance": {"artifact": {"sha256": "x"}}}
    all_d = audit_run(run, base, None)["terms"]["cloud_call_quota"]
    assert all_d["observed"] == 5 and not all_d["compliant"] and all_d["enforced"]
    base["plans"]["local_first"]["quotas"]["cloud_calls"]["counts"] = "billable"
    billable = audit_run(run, base, None)["terms"]["cloud_call_quota"]
    assert billable["observed"] == 3 and billable["compliant"]
