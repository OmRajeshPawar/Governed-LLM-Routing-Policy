from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from apex.core import load_config
from apex.policies import make_guarded_governor

NAN = float("nan")


def healthy_features(cfg) -> dict:
    return {
        "pending_local": 0.0, "active_local": 0.0, "active_cloud": 0.0,
        "semaphore_occupancy": 0.0, "roll_ttft_p95": 100.0,
        "roll_service_p95": 100.0, "arr_rate_5s": 0.2, "burst_ratio": 1.0,
        "gpu_util": 50.0, "vram_used_mb": 5000.0, "vram_free_mb": 3000.0,
        "prompt_token_est": 50.0, "max_output_tokens": 32.0,
        "recent_timeouts": 0.0, "recent_errors": 0.0,
    }


def _governor(cfg, episode_requests: int):
    return make_guarded_governor(cfg, episode_requests=episode_requests,
                                 frozen_cloud_p95_ms=1000.0)


def scenarios(cfg) -> list[dict]:
    rows = []

    def run(name, note, *, episode_requests=100, mutate=None, prime=None):
        gov = _governor(cfg, episode_requests)
        feat = healthy_features(cfg)
        if mutate:
            mutate(feat)
        if prime:
            prime(gov)
        d = gov.decide(feat)
        rows.append({
            "scenario": name, "note": note,
            "route": d.route, "reason_code": d.reason_code,
            "telemetry_tier": d.telemetry_tier, "regime": d.regime,
            "expert_name": d.expert_name, "safety_override": d.safety_override,
            "budget_denied": d.budget_denied, "circuit_state": d.circuit_state,
            "budget": f"{d.budget_used}/{d.budget_cap}",
            "matched_budget": d.matched_budget,
        })

    run("healthy", "full telemetry, idle queue -> the artifact's expert decides")

    run("gpu signal lost", "gpu_util missing -> demote to the tail-only expert",
        mutate=lambda f: f.update(gpu_util=NAN, vram_used_mb=NAN))

    run("minimal telemetry", "only queue depth left -> deterministic queue guard",
        mutate=lambda f: f.update(roll_ttft_p95=NAN, roll_service_p95=NAN,
                                  gpu_util=NAN,
                                  pending_local=float(cfg.policies.queue_threshold)))

    run("telemetry unusable", "cannot observe the local tier -> fail out to cloud",
        mutate=lambda f: f.update(pending_local=NAN, active_local=NAN,
                                  roll_ttft_p95=NAN, gpu_util=NAN))

    run("local hard breach", "queue past the hard cap -> safety override, not a vote",
        mutate=lambda f: f.update(
            pending_local=float(cfg.governor.hard_local_pending)))

    run("cloud circuit open", "3 consecutive cloud failures -> stay local",
        prime=lambda g: [g.circuit.record(False, None, now=i) for i in range(3)])

    run("dual degradation", "local breached AND cloud circuit open -> local anyway",
        mutate=lambda f: f.update(
            pending_local=float(cfg.governor.hard_local_pending)),
        prime=lambda g: [g.circuit.record(False, None, now=i) for i in range(3)])

    run("cloud budget exhausted", "cap is floor(0.30*1)=0 -> even a safety override is denied",
        episode_requests=1,
        mutate=lambda f: f.update(
            pending_local=float(cfg.governor.hard_local_pending)))

    return rows


def regime_hysteresis(cfg) -> list[dict]:
    gov = _governor(cfg, 100)
    feat = healthy_features(cfg)
    feat.update(burst_ratio=3.0, arr_rate_5s=2.0)
    out = []
    for i in range(1, 4):
        d = gov.decide(dict(feat))
        out.append({"arrival": i, "regime": d.regime,
                    "expert_name": d.expert_name, "route": d.route})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    args = ap.parse_args()

    cfg = load_config()
    gov = _governor(cfg, 100)
    art = gov.artifact
    rows = scenarios(cfg)
    hyst = regime_hysteresis(cfg)

    if args.json:
        print(json.dumps({"artifact_id": art.artifact_id,
                          "regime_map": art.regime_map,
                          "promotion_status": art.promotion_status,
                          "scenarios": rows,
                          "regime_hysteresis": hyst}, indent=2))
        return 0

    print()
    print("FROZEN GOVERNOR ARTIFACT")
    print(f"  id             {art.artifact_id}")
    print(f"  schema         {art.schema_version}")
    print(f"  regime map     {art.regime_map}")
    print(f"  promotion      {art.promotion_status.get('status')}")
    print(f"  dev seeds      {art.development_seeds}")
    print("  nothing online-learns: the map is loaded from disk and never updated")
    print()

    print("DETERMINISTIC GUARDS  (each row is one pre-decision snapshot)")
    print()
    hdr = f"{'scenario':<24} {'route':<6} {'tier':<9} {'expert':<20} {'circuit':<9} {'budget':<8} reason"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['scenario']:<24} {r['route']:<6} {r['telemetry_tier']:<9} "
              f"{str(r['expert_name']):<20} {r['circuit_state']:<9} "
              f"{r['budget']:<8} {r['reason_code']}")
    print()
    for r in rows:
        print(f"  {r['scenario']:<24} {r['note']}")
    print()

    print("REGIME HYSTERESIS  (burst conditions held constant)")
    print()
    for h in hyst:
        print(f"  arrival {h['arrival']}  regime={h['regime']:<7} "
              f"expert={h['expert_name']}")
    print("  two consecutive arrivals are required before the regime flips,")
    print("  so a single spike cannot swing the routing policy.")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
