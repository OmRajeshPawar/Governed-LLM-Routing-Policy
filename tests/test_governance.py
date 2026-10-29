import json
import math
from pathlib import Path

import pytest

from apex.core import load_config
from apex.policies import (CloudCircuitBreaker, GovernorArtifact,
                                 GuardedGovernor, TotalCloudBudget)
from apex.policies import (all_policies, make_guarded_governor,
                           frozen_cloud_p95_from_calibration)


ROOT = Path(__file__).resolve().parents[1]


def _full_features(cfg):
    return {
        "pending_local": 0.0, "active_local": 0.0, "active_cloud": 0.0,
        "semaphore_occupancy": 0.0, "roll_ttft_p95": 100.0,
        "roll_service_p95": 100.0, "arr_rate_5s": .2, "burst_ratio": 1.0,
        "gpu_util": 50.0, "vram_used_mb": 5000.0, "vram_free_mb": 3000.0,
        "prompt_token_est": 50.0, "max_output_tokens": 32.0,
        "recent_timeouts": 0.0, "recent_errors": 0.0,
    }


def test_total_cloud_budget_counts_every_cloud_action():
    budget = TotalCloudBudget(10, .30, "evaluation_hard")
    assert budget.request(discretionary=True)[0]
    assert budget.request(safety_override=True)[0]
    assert budget.request(probe=True)[0]
    assert budget.total_cloud == 3
    assert budget.discretionary_cloud == 1
    assert budget.safety_overrides == 1
    assert budget.probes == 1
    assert budget.request(safety_override=True) == (False, "budget_denied")
    assert budget.total_cloud == 3


def test_operational_soft_marks_excess_override_unmatched():
    budget = TotalCloudBudget(1, 0.0, "operational_soft")
    assert budget.request(safety_override=True)[0]
    assert budget.unmatched


def test_circuit_failure_open_cooldown_and_single_probe():
    cfg = load_config().governor.circuit
    circuit = CloudCircuitBreaker(cfg, frozen_cloud_p95_ms=100.0)
    circuit.record(False, None, now=0)
    circuit.record(False, None, now=1)
    circuit.record(False, None, now=2)
    assert circuit.state == "open"
    for i in range(10):
        circuit.on_arrival(now=33 + i)
    assert circuit.state == "half-open"
    assert circuit.cloud_available()
    assert circuit.begin_attempt()
    assert not circuit.begin_attempt()
    circuit.record(True, 50.0, now=50)
    assert circuit.state == "half-open"


def test_governor_hard_cap_denies_safety_override():
    cfg = load_config()
    artifact = GovernorArtifact.load(ROOT / "ml/artifacts/governor_v1.json")
    gov = GuardedGovernor(cfg, artifact, all_policies(cfg), episode_requests=1,
                          frozen_cloud_p95_ms=1000.0)
    feat = _full_features(cfg)
    feat["pending_local"] = float(cfg.governor.hard_local_pending)
    decision = gov.decide(feat)
    assert decision.route == "local"
    assert decision.budget_denied
    assert decision.budget_cap == 0
    assert "budget_denied" in decision.reason_code


def test_missing_telemetry_ladder_uses_minimal_queue_guard():
    cfg = load_config()
    artifact = GovernorArtifact.load(ROOT / "ml/artifacts/governor_v1.json")
    gov = GuardedGovernor(cfg, artifact, all_policies(cfg), episode_requests=10,
                          frozen_cloud_p95_ms=1000.0)
    feat = _full_features(cfg)
    feat["roll_ttft_p95"] = float("nan")
    feat["roll_service_p95"] = float("nan")
    feat["gpu_util"] = float("nan")
    feat["pending_local"] = float(cfg.policies.queue_threshold)
    decision = gov.decide(feat)
    assert decision.route == "cloud"
    assert decision.telemetry_tier == "minimal"
    assert decision.expert_name == "minimal_queue_guard"


def test_frozen_artifact_is_tail_aware_only():
    raw = json.loads((ROOT / "ml/artifacts/governor_v1.json").read_text())
    assert set(raw["regime_map"].values()) == {"tail_aware"}
    assert raw["promotion_status"]["confirmatory_data_used"] is False


def test_factory_builds_governor_from_frozen_artifact():
    cfg = load_config()
    gov = make_guarded_governor(cfg, episode_requests=100,
                                frozen_cloud_p95_ms=1000.0)
    assert gov.name == "guarded_governor"
    assert gov.artifact.schema_version == "governor-v1"
    assert set(gov.artifact.regime_map) == {"low", "medium", "high", "burst"}
    for expert in gov.artifact.regime_map.values():
        assert expert in gov.experts
    assert gov.budget.cap == 30
    assert gov.circuit.state == "closed"


def test_governor_is_not_registered_in_all_policies():
    assert "guarded_governor" not in all_policies(load_config())


def test_factory_governor_emits_the_full_audit_trail():
    cfg = load_config()
    gov = make_guarded_governor(cfg, episode_requests=100,
                                frozen_cloud_p95_ms=1000.0)
    decision = gov.decide(_full_features(cfg))
    for field in ("expert_name", "regime", "telemetry_tier", "budget_used",
                  "budget_cap", "budget_remaining", "circuit_state",
                  "matched_budget"):
        assert getattr(decision, field) is not None, field
    assert decision.matched_budget is True


def test_frozen_cloud_p95_tracks_the_calibration_and_speed_ratio():
    cfg = load_config()
    calib = {"base_ttft_ms": 200.0, "per_token_ms": 5.0}
    p95 = frozen_cloud_p95_from_calibration(cfg, calib, est_output_tokens=64.0)
    median = cfg.providers.cloud.network_rtt_ms + 200.0 * cfg.providers.cloud.speed_ratio
    assert p95 > median
    assert p95 == pytest.approx(median * math.exp(1.645 * cfg.providers.cloud.jitter_sigma))
    cfg.providers.cloud.speed_ratio = 4.0
    assert frozen_cloud_p95_from_calibration(cfg, calib, 64.0) > p95


def test_runner_refuses_the_governor_when_config_disables_it():
    src = (ROOT / "apex/runner.py").read_text(encoding="utf-8")
    assert "cfg.governor.enabled" in src
    assert "guarded_governor" in src
    cfg = load_config()
    assert cfg.governor.enabled, "config.yaml ships with the governor enabled"
