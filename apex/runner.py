from __future__ import annotations

import argparse
import asyncio
import re
import statistics
import time
from pathlib import Path

from .core import Config, Request, GatewayState, load_config, ensure_dirs, _repo_root
from .events import EventLog, write_manifest, events_to_parquet
from .gateway import Gateway
from .policies import (all_policies, Policy, make_guarded_governor,
                       frozen_cloud_p95_from_calibration)
from .providers import OllamaProvider, CalibratedCloudProvider, calibrate_cloud


def _threshold(cfg: Config, policy_name: str) -> float:
    p = cfg.policies
    return {
        "queue_threshold": float(p.queue_threshold),
        "tail_aware": p.tail_aware_pressure_threshold,
        "ml_ttft": p.ml_ttft_threshold,
        "ml_dual_risk": p.ml_occupancy_threshold,
        "semantic_embedding": p.semantic_embedding_threshold,
        "local_llm_probe": p.local_llm_probe_threshold,
        "thinking_probe": p.thinking_probe_threshold,
    }.get(policy_name, 0.5)


def sanitize_variant(value: str | None) -> str | None:
    if not value:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    safe = safe.strip("._-")
    return safe or None


async def build_providers(cfg: Config, *, allow_cloud_calibration: bool = True):
    local = OllamaProvider("local", cfg.providers.local.model,
                           cfg.providers.local.keep_alive, cfg.providers.local.num_gpu,
                           cfg.generation.temperature,
                           num_ctx=cfg.providers.local.num_ctx)
    cc = cfg.providers.cloud
    calib = {"mode": cc.mode}
    if cc.mode == "live":
        cloud = OllamaProvider("cloud", cc.calibration_model, cc.keep_alive,
                               cc.num_gpu, cfg.generation.temperature,
                               cost_per_1k_tokens=cc.cost_per_1k_tokens,
                               num_ctx=cc.num_ctx)
    else:
        if cc.base_ttft_ms is None or cc.per_token_ms is None:
            if allow_cloud_calibration:
                fit = await calibrate_cloud(cc.calibration_model, cc.num_gpu,
                                            cfg.generation.temperature,
                                            num_ctx=cc.num_ctx)
            else:
                fit = {
                    "calibration_model": cc.calibration_model,
                    "base_ttft_ms": 200.0,
                    "per_token_ms": 5.0,
                    "jitter_sigma": cc.jitter_sigma,
                    "n_samples": 0,
                    "fallback": True,
                    "skipped": "shadow_route_forced_local",
                }
            base, per_tok = fit["base_ttft_ms"], fit["per_token_ms"]
            calib.update(fit)
        else:
            base, per_tok = cc.base_ttft_ms, cc.per_token_ms
            calib.update({"base_ttft_ms": base, "per_token_ms": per_tok, "source": "config"})
        cloud = CalibratedCloudProvider(cc, base, per_tok, cfg.seed)
    return local, cloud, calib


def load_risk_model(cfg: Config):
    import sys
    sys.path.insert(0, str(_repo_root()))
    from ml.risk_model import CalibratedRiskModel
    model_dir = _repo_root() / cfg.paths.models_dir
    if (model_dir / "ttft_model.joblib").exists():
        return CalibratedRiskModel.load(model_dir)
    return None


async def run_trace(cfg: Config, trace: list[dict], policy: Policy, *,
                    mode: str, run_id: str, out_jsonl: Path,
                    ml=None, route_override=None,
                    calib: dict | None = None,
                    local=None, cloud=None, variant: str | None = None) -> dict:
    if local is None or cloud is None:
        local, cloud, calib = await build_providers(
            cfg,
            allow_cloud_calibration=(route_override != "local"),
        )
    state = GatewayState(cfg)
    log = EventLog(out_jsonl)
    await log.start()
    gw = Gateway(cfg, local, cloud, state, log, run_id, trace_id=f"{cfg.seed}")

    load_s = await local.warmup()
    thr = _threshold(cfg, policy.name)

    t0 = time.perf_counter()
    tasks = []
    for spec in trace:
        target = t0 + spec["arrival_offset"]
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        req = Request(spec["request_id"], spec["prompt"], spec["max_tokens"],
                      spec["task_category"], time.perf_counter())
        req.trace_pos = spec["trace_pos"]
        req.scheduled_arrival_offset_s = spec.get("arrival_offset")
        req.workload_phase = spec.get("workload_phase")
        req.cloud_degraded = bool(spec.get("cloud_degraded", False))
        override = route_override(spec) if callable(route_override) else route_override
        tasks.append(asyncio.create_task(
            gw.handle(req, policy, threshold=thr, ml=ml, route_override=override)))
    results = await asyncio.gather(*tasks)
    await log.aclose()

    n_cloud = sum(1 for r in results if r["decision"].route == "cloud")
    manifest = {"run_id": run_id, "mode": mode, "policy": policy.name,
                "variant": variant,
                "n_requests": len(trace), "n_offloaded": n_cloud,
                "warm_load_s": round(load_s, 2), "calibration": calib,
                "wall_clock_s": round(time.perf_counter() - t0, 2),
                "created_unix": time.time()}
    write_manifest(out_jsonl.with_suffix(".manifest.json"), cfg.model_dump(), manifest)
    events_to_parquet(out_jsonl, out_jsonl.with_suffix(".parquet"))
    return manifest


async def _main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["shadow", "replay"], default="shadow")
    ap.add_argument("--policy", default="tail_aware")
    ap.add_argument("--pattern", default="bursty")
    ap.add_argument("--mix", default="synthetic",
                    choices=["synthetic", "real_sample", "reasoning_coding"])
    ap.add_argument("--n", type=int, default=160)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--speed-ratio", type=float, default=None)
    ap.add_argument("--variant", default=None)
    ap.add_argument("--local-model", default=None)
    ap.add_argument("--num-ctx", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg.seed = args.seed
    if args.speed_ratio is not None:
        cfg.providers.cloud.speed_ratio = args.speed_ratio
    if args.local_model:
        cfg.providers.local.model = args.local_model
    if args.num_ctx is not None:
        cfg.providers.local.num_ctx = args.num_ctx
    ensure_dirs(cfg)
    variant = sanitize_variant(args.variant)

    from .workload import make_trace
    trace = make_trace(n=args.n, pattern=args.pattern, seed=cfg.seed, mix=args.mix)
    events_dir = _repo_root() / cfg.paths.events_dir

    if args.mode == "shadow":
        policy = all_policies(cfg)["always_local"]
        prefix = f"shadow_{variant}_" if variant else "shadow_"
        run_id = f"{prefix}{args.mix}_{args.pattern}_n{args.n}_s{cfg.seed}"
        out = events_dir / f"{run_id}.jsonl"
        m = await run_trace(cfg, trace, policy, mode="shadow", run_id=run_id,
                            out_jsonl=out, route_override="local", variant=variant)
    elif args.policy == "guarded_governor":
        if not cfg.governor.enabled:
            raise SystemExit(
                "governor.enabled is false in config.yaml - refusing to run "
                "the guarded governor from a config that disables it.")
        local, cloud, calib = await build_providers(cfg)
        est_tokens = statistics.median(float(s["max_tokens"]) for s in trace)
        frozen_p95 = frozen_cloud_p95_from_calibration(cfg, calib, est_tokens)
        calib["frozen_cloud_p95_ms"] = round(frozen_p95, 2)
        calib["frozen_cloud_p95_est_output_tokens"] = est_tokens
        policy = make_guarded_governor(cfg, episode_requests=len(trace),
                                       frozen_cloud_p95_ms=frozen_p95)
        ml = (load_risk_model(cfg)
              if any(e.startswith("ml_") for e in policy.artifact.regime_map.values())
              else None)
        prefix = f"replay_{variant}_" if variant else "replay_"
        run_id = f"{prefix}{args.policy}_{args.mix}_{args.pattern}_n{args.n}_s{cfg.seed}_R{cfg.providers.cloud.speed_ratio}"
        out = events_dir / f"{run_id}.jsonl"
        m = await run_trace(cfg, trace, policy, mode="replay", run_id=run_id,
                            out_jsonl=out, ml=ml, variant=variant,
                            calib=calib, local=local, cloud=cloud)
        print(f"  governor artifact {policy.artifact.artifact_id} | "
              f"regimes {policy.artifact.regime_map} | "
              f"cloud budget {policy.budget.cap}/{len(trace)} | "
              f"frozen cloud p95 {frozen_p95:.0f}ms")
    else:
        pool = all_policies(cfg)
        if args.policy not in pool:
            raise SystemExit(
                f"unknown policy {args.policy!r}. available: "
                f"{', '.join(sorted(pool))}, guarded_governor")
        policy = pool[args.policy]
        ml = load_risk_model(cfg) if args.policy.startswith("ml_") else None
        if args.policy.startswith("ml_") and ml is None:
            print("WARNING: no trained model found; run ml/train.py first.")
        prefix = f"replay_{variant}_" if variant else "replay_"
        run_id = f"{prefix}{args.policy}_{args.mix}_{args.pattern}_n{args.n}_s{cfg.seed}_R{cfg.providers.cloud.speed_ratio}"
        out = events_dir / f"{run_id}.jsonl"
        m = await run_trace(cfg, trace, policy, mode="replay", run_id=run_id,
                            out_jsonl=out, ml=ml, variant=variant)

    print(f"[{m['policy']}] {m['n_requests']} reqs | offloaded {m['n_offloaded']} | "
          f"wall {m['wall_clock_s']}s | -> {out.name}")


if __name__ == "__main__":
    asyncio.run(_main())
