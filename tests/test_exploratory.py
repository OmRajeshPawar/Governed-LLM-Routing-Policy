import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from apex.core import load_config
from soc4ai.paced import (FreshTailAware, PacedCloudBudget, allowance_for,
                          local_busy, make_paced_governor)


def _feat(**over):
    base = {"pending_local": 0.0, "active_local": 0.0, "active_cloud": 0.0,
            "semaphore_occupancy": 0.0, "roll_ttft_p95": 20000.0, "roll_service_p95": 1000.0,
            "arr_rate_5s": 0.2, "burst_ratio": 1.0, "gpu_util": 50.0, "vram_used_mb": 5000.0,
            "vram_free_mb": 3000.0, "prompt_token_est": 50.0, "max_output_tokens": 32.0,
            "recent_timeouts": 0.0, "recent_errors": 0.0}
    base.update(over)
    return base


def _tuned_cfg():
    cfg = load_config()
    meta = json.loads((ROOT / "ml" / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    cfg.targets.sla_ttft_ms = meta["frozen_thresholds"]["sla_ttft_ms"]
    cfg.policies.tail_aware_pressure_threshold = 4.986486535529362
    return cfg, meta["heuristic_weights"]


def test_paced_budget_releases_in_proportion_to_arrivals():
    b = PacedCloudBudget(300, 0.30, "evaluation_hard", allowance_for(300))
    b.on_arrival()
    assert b.released == 15
    admitted = sum(b.request(safety_override=True)[0] for _ in range(20))
    assert admitted == 15 and b.denials == 5
    for _ in range(49):
        b.on_arrival()
    assert b.released == 30
    for _ in range(1000):
        b.on_arrival()
    assert b.released == b.cap == 90


def test_fresh_override_is_suppressed_while_local_tier_is_idle():
    cfg, weights = _tuned_cfg()
    expert = FreshTailAware(cfg, weights)
    idle = expert.decide(_feat(), 4.986486535529362)
    assert idle.route == "local"
    busy = expert.decide(_feat(pending_local=2.0, active_local=2.0), 4.986486535529362)
    assert busy.route == "cloud" and busy.reason_code == "tail_hard_breach"


def test_local_busy_defaults_to_busy_when_unobservable():
    cfg, _ = _tuned_cfg()
    assert local_busy(cfg, _feat(pending_local=float("nan"), active_local=float("nan")))
    assert not local_busy(cfg, _feat())


def test_paced_governor_keeps_queue_cap_breach_but_not_stale_p95():
    cfg, weights = _tuned_cfg()
    gov = make_paced_governor(cfg, episode_requests=300, frozen_cloud_p95_ms=3070.0, heuristic_weights=weights)
    assert gov.name == "paced_governor"
    assert gov._local_state(_feat(), True) == "healthy"
    cap = float(cfg.governor.hard_local_pending)
    assert gov._local_state(_feat(pending_local=cap), True) == "hard_breach"
