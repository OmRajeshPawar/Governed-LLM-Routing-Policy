"""SOC4AI live study driver.

Stages:
  recalibrate  derive governor_v2.json: tail-aware threshold set to the matched
               30% offload point (70th score percentile) on the July development
               shadow corpus; v1's threshold (1.0) offloads every request on the
               tuned score scale. No confirmatory data is used.
  calibrate    30 live Gemini calls on representative prompts -> frozen cloud P95
  dryrun       throwaway n=24 live runs (governed with faults, ungoverned) to
               validate the pipeline; never analysed as results
  freeze       write protocol.json + protocol.sha256
  batch        execute the frozen matrix A -> B -> C, blind, checkpointed,
               with the USD 20 kill switch

Usage (from the repository root):  SOC4AI_GCP_KEY=<path> python soc4ai/study.py <stage>
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soc4ai import creds  # noqa: E402

# Force credentials before anything else can touch them. Analysis-only stages run
# without a key; any live Gemini call still refuses to start without one.
if os.environ.get(creds.KEY_ENV):
    creds.force_credentials()

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from apex.core import FEATURE_NAMES, GatewayState, Request, load_config  # noqa: E402
from apex.events import EventLog, events_to_parquet, load_events, write_manifest  # noqa: E402
from apex.gateway import Gateway  # noqa: E402
from apex.policies import TailAwareHeuristic, all_policies, make_guarded_governor  # noqa: E402
from apex.runner import _threshold  # noqa: E402
from apex.workload import make_trace, make_transition_trace  # noqa: E402
from soc4ai.ledger import BudgetKillSwitch, SpendLedger  # noqa: E402
from soc4ai.providers import (CURRENT_DEGRADED, CURRENT_TRACE_POS,  # noqa: E402
                              FaultInjectingProvider, LiveGeminiProvider,
                              TimedOllamaProvider)
from soc4ai.telemetry import GpuSampler, attach_tracing_breaker, gpu_temp_c  # noqa: E402

OUT = ROOT / "soc4ai_out"
LEDGER_PATH = OUT / "spend_ledger.json"
PROTOCOL_PATH = OUT / "protocol.json"
PROTOCOL_HASH = OUT / "protocol.sha256"
CALIB_PATH = OUT / "calibration.json"
PROGRESS = OUT / "progress.log"
V1_ARTIFACT = ROOT / "ml" / "artifacts" / "governor_v1.json"
V2_ARTIFACT = ROOT / "ml" / "artifacts" / "governor_v2.json"
# Source tree whose git state is recorded in the protocol (default: this repository).
ORIGINAL_REPO = Path(os.environ.get("APEX_ORIGINAL_REPO", str(ROOT)))
SMOKE_SPEND_USD = 0.000292  # 3-call smoke test made before the ledger existed
COOLDOWN_C = 58.0
COOLDOWN_MAX_S = 180.0
MIXES = ("synthetic", "real_sample", "reasoning_coding")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    with open(PROGRESS, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    print(line, flush=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ollama_fingerprint(model: str) -> dict:
    version = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
    listing = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
    digest = None
    for line in listing.splitlines():
        parts = line.split()
        if parts and parts[0] == model:
            digest = parts[1]
    return {"ollama_version": version.splitlines()[-1] if version else None, "model": model, "model_id": digest}


def gpu_driver() -> str | None:
    r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
                       capture_output=True, text=True)
    return r.stdout.strip() or None


def original_repo_status() -> dict:
    head = subprocess.run(["git", "-C", str(ORIGINAL_REPO), "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(ORIGINAL_REPO), "status", "--porcelain"],
                           capture_output=True, text=True).stdout.strip()
    return {"head": head, "clean": dirty == ""}


def code_hashes() -> dict:
    files = sorted((ROOT / "soc4ai").glob("*.py")) + [
        ROOT / "apex" / "gateway.py", ROOT / "apex" / "providers.py", ROOT / "apex" / "policies.py",
        ROOT / "apex" / "workload.py", ROOT / "config.yaml", V1_ARTIFACT, V2_ARTIFACT,
        ROOT / "ml" / "artifacts" / "meta.json"]
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_file(p) for p in files if p.exists()}


def base_config(*, use_v2: bool = True):
    cfg = load_config()
    meta = json.loads((ROOT / "ml" / "artifacts" / "meta.json").read_text(encoding="utf-8"))
    fr = meta["frozen_thresholds"]
    cfg.targets.sla_ttft_ms = fr["sla_ttft_ms"]
    cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]
    if use_v2:
        if not V2_ARTIFACT.exists():
            raise SystemExit("governor_v2.json missing; run the recalibrate stage first")
        v2 = json.loads(V2_ARTIFACT.read_text(encoding="utf-8"))
        cfg.governor.artifact_path = "ml/artifacts/governor_v2.json"
        cfg.policies.tail_aware_pressure_threshold = float(v2["thresholds"]["tail_aware"])
    return cfg, meta


def weights_for(meta: dict, choice: str) -> dict | None:
    return meta.get("heuristic_weights") if choice == "tuned" else None


def pct(values: list[float], q: float) -> float:
    s = sorted(values)
    return float(s[max(0, min(len(s) - 1, math.ceil(q * len(s)) - 1))])


# --------------------------------------------------------------------------- run
async def run_one(cfg, trace: list[dict], *, policy_name: str, weights, cloud, run_id: str,
                  out_dir: Path, frozen_p95: float, extra_manifest: dict) -> dict:
    local = TimedOllamaProvider("local", cfg.providers.local.model, cfg.providers.local.keep_alive,
                                cfg.providers.local.num_gpu, cfg.generation.temperature,
                                num_ctx=cfg.providers.local.num_ctx)
    if policy_name == "guarded_governor":
        policy = make_guarded_governor(cfg, episode_requests=len(trace),
                                       frozen_cloud_p95_ms=frozen_p95, heuristic_weights=weights)
        attach_tracing_breaker(policy)
    else:
        policy = all_policies(cfg, weights)[policy_name]
    out_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = out_dir / f"{run_id}.jsonl"
    state = GatewayState(cfg)
    elog = EventLog(out_jsonl)
    await elog.start()
    gw = Gateway(cfg, local, cloud, state, elog, run_id, trace_id=str(cfg.seed))
    load_s = await local.warmup()
    thr = _threshold(cfg, policy.name)
    sampler = GpuSampler(out_dir / f"{run_id}.gpu.csv")
    sampler.start()
    t0_perf, t0_unix = time.perf_counter(), time.time()
    tasks = []
    try:
        for spec in trace:
            delay = t0_perf + spec["arrival_offset"] - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
            req = Request(spec["request_id"], spec["prompt"], spec["max_tokens"],
                          spec["task_category"], time.perf_counter())
            req.trace_pos = spec["trace_pos"]
            req.scheduled_arrival_offset_s = spec.get("arrival_offset")
            req.workload_phase = spec.get("workload_phase")
            req.cloud_degraded = bool(spec.get("cloud_degraded", False))
            CURRENT_TRACE_POS.set(req.trace_pos)
            CURRENT_DEGRADED.set(req.cloud_degraded)
            tasks.append(asyncio.create_task(gw.handle(req, policy, threshold=thr, ml=None)))
        results = await asyncio.gather(*tasks)
    finally:
        await elog.aclose()
        sampler.stop()
    manifest = {
        "run_id": run_id, "policy": policy.name, "n_requests": len(trace),
        "n_offloaded": sum(1 for r in results if r["decision"].route == "cloud"),
        "warm_load_s": round(load_s, 2), "wall_clock_s": round(time.perf_counter() - t0_perf, 2),
        "t0_perf_s": t0_perf, "t0_unix_s": t0_unix, "frozen_cloud_p95_ms": frozen_p95,
        "tail_aware_threshold": thr if policy_name != "guarded_governor" else
        policy.artifact.thresholds.get("tail_aware"),
        "gpu_samples": sampler.samples, "created_unix": time.time(), **extra_manifest,
    }
    if policy_name == "guarded_governor":
        manifest["governor_artifact"] = {"artifact_id": policy.artifact.artifact_id,
                                         "sha256": sha256_file(ROOT / cfg.governor.artifact_path)}
        manifest["breaker_transitions"] = policy.circuit.transitions
        manifest["budget"] = {"cap": policy.budget.cap, "total_cloud": policy.budget.total_cloud,
                              "discretionary": policy.budget.discretionary_cloud,
                              "safety_overrides": policy.budget.safety_overrides,
                              "denials": policy.budget.denials}
    if isinstance(cloud, FaultInjectingProvider):
        manifest["faults_injected"] = dict(cloud.injected)
    write_manifest(out_jsonl.with_suffix(".manifest.json"), cfg.model_dump(), manifest)
    events_to_parquet(out_jsonl, out_jsonl.with_suffix(".parquet"))
    return manifest


def build_trace(spec: dict) -> list[dict]:
    if spec["trace"] == "transition":
        return make_transition_trace(seed=spec["seed"], n=200)
    return make_trace(n=spec["n"], pattern="bursty", seed=spec["seed"], mix=spec["mix"])


def build_cloud(spec: dict, ledger: SpendLedger, protocol: dict):
    live = LiveGeminiProvider(ledger)
    faults = spec.get("faults") or {}
    if not faults:
        return live
    fm = protocol["fault_model"]
    return FaultInjectingProvider(
        live, outage=faults.get("outage"), spike=faults.get("spike"),
        spike_delay_ms=fm["spike_delay_ms"],
        degraded_delay_ms=fm["degraded_delay_ms"] if faults.get("degraded_phase") else 0.0,
        outage_latency_ms=fm["outage_latency_ms"])


# ----------------------------------------------------------------------- stages
def stage_recalibrate(weights_choice: str) -> None:
    cfg, meta = base_config(use_v2=False)
    files = sorted((ROOT / "data" / "events").glob("shadow_*_bursty_n300_s*.jsonl"))
    df = pd.concat([load_events(f) for f in files], ignore_index=True)
    heuristic = TailAwareHeuristic(cfg, weights_for(meta, weights_choice))
    scores = np.array([heuristic.score(f) for f in df[FEATURE_NAMES].to_dict("records")])
    tau = float(np.quantile(scores, 0.70))
    v1 = json.loads(V1_ARTIFACT.read_text(encoding="utf-8"))
    v1_share = float((scores >= v1["thresholds"]["tail_aware"]).mean())
    v2 = dict(v1)
    v2["artifact_id"] = "apex-guarded-tail-aware-matched30-2026-09-15"
    v2["created_utc"] = now_utc()
    v2["thresholds"] = {**v1["thresholds"], "tail_aware": tau}
    v2["promotion_status"] = {
        **v1["promotion_status"],
        "status": "no_richer_expert_promoted",
        "reason": "Expert unchanged (tail-aware in every regime). Threshold recalibrated: the v1 "
                  "threshold offloads every development request on the tuned score scale.",
        "threshold_recalibration": {
            "from": v1["thresholds"]["tail_aware"], "to": tau,
            "rule": "70th percentile of the tuned tail-aware score on the 4,500-request July "
                    "development shadow corpus (matched 30% offload)",
            "v1_offload_share_on_development_corpus": v1_share,
            "v2_offload_share_on_development_corpus": float((scores >= tau).mean()),
            "heuristic_weights": weights_choice,
            "n_rows": int(len(scores)),
        },
        "derived_from_artifact_sha256": sha256_file(V1_ARTIFACT),
        "confirmatory_data_used": False,
    }
    V2_ARTIFACT.write_text(json.dumps(v2, indent=2), encoding="utf-8")
    log(f"recalibrate: {len(files)} shadow files, {len(scores)} rows, v1 threshold "
        f"{v1['thresholds']['tail_aware']} offloads {v1_share:.1%}; v2 threshold {tau:.4f} "
        f"offloads {(scores >= tau).mean():.1%}; sha256 {sha256_file(V2_ARTIFACT)[:16]}")


async def stage_calibrate(n_calls: int = 30) -> None:
    ledger = SpendLedger(LEDGER_PATH)
    if ledger.n_calls == 0 and ledger.committed == 0.0:
        ledger.add_external(SMOKE_SPEND_USD, n_calls=3)
    live = LiveGeminiProvider(ledger)
    prompts = []
    for i, mix in enumerate(MIXES):
        prompts += make_trace(n=10, pattern="bursty", seed=9001 + i, mix=mix)
    rows = []
    for spec in prompts[:n_calls]:
        res = await live.generate(spec["prompt"], spec["max_tokens"])
        rows.append({"mix": spec.get("mix"), "max_tokens": spec["max_tokens"], "ok": res.ok,
                     "ttft_ms": res.provider_ttft_ms, "service_ms": res.service_ms,
                     "prompt_tokens": res.prompt_tokens, "output_tokens": res.output_tokens,
                     "cost_usd": res.cost_usd, "error": res.error})
    good = [r["ttft_ms"] for r in rows if r["ok"] and r["ttft_ms"] is not None]
    summary = {"created_utc": now_utc(), "n_calls": len(rows), "n_ok": len(good),
               "ttft_p50_ms": statistics.median(good) if good else None,
               "ttft_p95_ms": pct(good, 0.95) if good else None,
               "ttft_max_ms": max(good) if good else None,
               "spend_usd": sum(r["cost_usd"] for r in rows), "rows": rows}
    CALIB_PATH.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    log(f"calibration: {len(good)}/{len(rows)} ok, p50={summary['ttft_p50_ms']:.0f}ms "
        f"p95={summary['ttft_p95_ms']:.0f}ms, ledger ${ledger.committed:.4f}")


async def stage_dryrun(weights_choice: str) -> None:
    cfg, meta = base_config()
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    p95 = float(calib["ttft_p95_ms"])
    ledger = SpendLedger(LEDGER_PATH)
    proto_stub = {"fault_model": {"spike_delay_ms": 2.5 * p95, "degraded_delay_ms": 2.5 * p95,
                                  "outage_latency_ms": 250.0}}
    out_dir = OUT / "dryrun"
    for policy_name, faults in [("guarded_governor", {"outage": [6, 12], "spike": [14, 20]}),
                                ("tail_aware", {})]:
        cfg.seed = 999
        spec = {"trace": "bursty", "n": 24, "seed": 999, "mix": "synthetic", "faults": faults}
        m = await run_one(cfg, build_trace(spec), policy_name=policy_name,
                          weights=weights_for(meta, weights_choice),
                          cloud=build_cloud(spec, ledger, proto_stub), run_id=f"dryrun_{policy_name}",
                          out_dir=out_dir, frozen_p95=p95, extra_manifest={"dryrun": True})
        log(f"dryrun {policy_name}: {m['n_requests']} reqs, offloaded {m['n_offloaded']}, "
            f"wall {m['wall_clock_s']}s, gpu samples {m['gpu_samples']}, ledger ${ledger.committed:.4f}")


def stage_freeze(weights_choice: str) -> None:
    cfg, meta = base_config()
    calib = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
    p95 = float(calib["ttft_p95_ms"])
    spike_delay = 2.5 * p95
    breaker_limit = cfg.governor.circuit.latency_multiplier * p95
    runs = []
    for i, seed in enumerate([211, 223, 227, 229, 233, 239, 241, 251]):
        mix = MIXES[i % 3]
        order = ["guarded_governor", "tail_aware"] if i % 2 == 0 else ["tail_aware", "guarded_governor"]
        for pol in order:
            runs.append({"experiment": "A", "key": f"A_s{seed}_{mix}_{pol}", "trace": "bursty", "n": 300,
                         "seed": seed, "mix": mix, "policy": pol, "condition": "nominal", "faults": {}})
    for i, seed in enumerate([7, 13, 42]):
        mix = MIXES[i]
        order = ["guarded_governor", "tail_aware"] if i % 2 == 0 else ["tail_aware", "guarded_governor"]
        for pol in order:
            runs.append({"experiment": "B", "key": f"B_s{seed}_{mix}_{pol}", "trace": "bursty", "n": 300,
                         "seed": seed, "mix": mix, "policy": pol, "condition": "chaos",
                         "faults": {"outage": [96, 156], "spike": [192, 252]}})
    for i, seed in enumerate([7, 13, 42]):
        order = ["guarded_governor", "tail_aware"] if i % 2 == 0 else ["tail_aware", "guarded_governor"]
        for pol in order:
            runs.append({"experiment": "C", "key": f"C_s{seed}_transition_{pol}", "trace": "transition", "n": 200,
                         "seed": seed, "mix": "phased", "policy": pol, "condition": "drift",
                         "faults": {"degraded_phase": True}})
    v2 = json.loads(V2_ARTIFACT.read_text(encoding="utf-8"))
    protocol = {
        "created_utc": now_utc(),
        "purpose": "SOC4AI 2026 live validation of the governed routing policy",
        "sla_ttft_ms": cfg.targets.sla_ttft_ms,
        "sla_violation_rule": "apex.core.ttft_breach(end_to_end_ttft_ms, sla, ok) for every route; failed requests violate",
        "confirmatory_ttft_rule": "failed requests use end_to_end_ttft_ms = max(end_to_end_ms or 0, sla_ttft_ms)",
        "confirmatory_analysis": "analysis/confirmatory_statistics.py unchanged; seeds 211..251 preregistered July 2026",
        "blind_rule": "progress log carries no outcome metrics; no experiment analysed before its full corpus completes",
        "heuristic_weights": weights_choice,
        "heuristic_weights_values": weights_for(meta, weights_choice),
        "governor_artifact": {"path": "ml/artifacts/governor_v2.json", "sha256": sha256_file(V2_ARTIFACT),
                              "derived_from_sha256": v2["promotion_status"]["derived_from_artifact_sha256"],
                              "threshold_recalibration": v2["promotion_status"]["threshold_recalibration"]},
        "tail_aware_threshold": cfg.policies.tail_aware_pressure_threshold,
        "cloud_tier": {"provider": "Vertex AI", "model": creds.MODEL, "region": creds.LOCATION,
                       "project": creds.PROJECT, "thinking_budget": 0, "temperature": 0.0},
        "calibration": {k: calib[k] for k in ("created_utc", "n_calls", "n_ok", "ttft_p50_ms",
                                              "ttft_p95_ms", "ttft_max_ms", "spend_usd")},
        "frozen_cloud_p95_ms": p95,
        "breaker": {**cfg.governor.circuit.model_dump(), "latency_limit_ms": breaker_limit},
        "fault_model": {"outage_latency_ms": 250.0, "spike_delay_ms": spike_delay,
                        "degraded_delay_ms": spike_delay,
                        "spike_exceeds_breaker_limit_by_ms": spike_delay - breaker_limit,
                        "window_rationale": "bursty cycle is 24 requests (positions i%24<6 arrive at 1.5/s); "
                                            "outage [96,156) and spike [192,252) both start at a burst boundary; "
                                            "drift degradation applies to the transition trace's "
                                            "cloud_degradation phase, which is otherwise only a label"},
        "cooldown": {"gpu_temp_c_below": COOLDOWN_C, "max_wait_s": COOLDOWN_MAX_S},
        "spend_kill_switch_usd": 20.0,
        "environment": {**ollama_fingerprint(cfg.providers.local.model), "gpu_driver": gpu_driver(),
                        "python": sys.version.split()[0]},
        "original_repo": original_repo_status(),
        "code_sha256": code_hashes(),
        "runs": runs,
    }
    if protocol["fault_model"]["spike_exceeds_breaker_limit_by_ms"] <= 0:
        raise SystemExit("spike delay does not exceed the breaker latency limit; refusing to freeze")
    text = json.dumps(protocol, indent=1)
    PROTOCOL_PATH.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    PROTOCOL_HASH.write_text(digest + "\n", encoding="utf-8")
    log(f"protocol frozen: {len(runs)} runs, sha256 {digest[:16]}, weights={weights_choice}, "
        f"threshold {cfg.policies.tail_aware_pressure_threshold:.4f}, frozen p95 {p95:.0f}ms, "
        f"spike {spike_delay:.0f}ms vs breaker limit {breaker_limit:.0f}ms")


async def stage_batch(only: str | None) -> int:
    text = PROTOCOL_PATH.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if PROTOCOL_HASH.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit("protocol.json does not match protocol.sha256")
    protocol = json.loads(text)
    cfg0, _ = base_config()
    if sha256_file(V2_ARTIFACT) != protocol["governor_artifact"]["sha256"]:
        raise SystemExit("governor_v2.json changed after the protocol was frozen")
    frozen_env = protocol["environment"]
    env_now = ollama_fingerprint(cfg0.providers.local.model)
    if (env_now["ollama_version"], env_now["model_id"]) != (frozen_env["ollama_version"], frozen_env["model_id"]):
        raise SystemExit(f"environment drift: {env_now} vs frozen {frozen_env}")
    if not original_repo_status()["clean"]:
        raise SystemExit("source repository has uncommitted changes (commit first, so the protocol records a clean tree); refusing to start")
    ledger = SpendLedger(LEDGER_PATH)
    weights = protocol["heuristic_weights_values"]
    p95 = float(protocol["frozen_cloud_p95_ms"])
    runs = [r for r in protocol["runs"] if only is None or r["experiment"] == only]
    log(f"BATCH START {len(runs)} runs, protocol {digest[:16]}, ledger ${ledger.committed:.4f}")
    for idx, spec in enumerate(runs, 1):
        exp_dir = OUT / "runs" / spec["experiment"]
        done_marker = exp_dir / f"{spec['key']}.done"
        if done_marker.exists():
            log(f"[{idx}/{len(runs)}] skip {spec['key']} (done)")
            continue
        for attempt in (1, 2):
            waited = 0.0
            temp = gpu_temp_c()
            while temp is not None and temp >= COOLDOWN_C and waited < COOLDOWN_MAX_S:
                time.sleep(10)
                waited += 10
                temp = gpu_temp_c()
            env_check = ollama_fingerprint(cfg0.providers.local.model)
            if (env_check["model_id"], env_check["ollama_version"]) != (frozen_env["model_id"], frozen_env["ollama_version"]):
                log(f"HALT environment drift before {spec['key']}: {env_check}")
                return 4
            cfg, _ = base_config()
            cfg.seed = spec["seed"]
            t_start = time.time()
            try:
                m = await run_one(cfg, build_trace(spec), policy_name=spec["policy"], weights=weights,
                                  cloud=build_cloud(spec, ledger, protocol), run_id=spec["key"],
                                  out_dir=exp_dir, frozen_p95=p95,
                                  extra_manifest={"protocol_sha256": digest, "spec": spec,
                                                  "cooldown_waited_s": waited, "gpu_temp_start_c": temp,
                                                  "environment": env_check})
                manifest_path = exp_dir / f"{spec['key']}.manifest.json"
                mf = json.loads(manifest_path.read_text(encoding="utf-8"))
                mf["ledger_committed_usd_end"] = ledger.committed
                manifest_path.write_text(json.dumps(mf, indent=2), encoding="utf-8")
                done_marker.write_text(now_utc(), encoding="utf-8")
                log(f"[{idx}/{len(runs)}] done {spec['key']} attempt {attempt} "
                    f"wall {m['wall_clock_s']:.0f}s cooldown {waited:.0f}s ledger ${ledger.committed:.4f}")
                break
            except BudgetKillSwitch as exc:
                log(f"KILL SWITCH during {spec['key']}: {exc}. Batch halted.")
                (OUT / "HALTED").write_text(now_utc() + " " + str(exc), encoding="utf-8")
                return 3
            except Exception:
                err_dir = OUT / "errors"
                err_dir.mkdir(parents=True, exist_ok=True)
                (err_dir / f"{spec['key']}_attempt{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                log(f"[{idx}/{len(runs)}] ERROR {spec['key']} attempt {attempt} "
                    f"({time.time() - t_start:.0f}s); see errors/")
        else:
            log(f"[{idx}/{len(runs)}] SKIPPED {spec['key']} after 2 failed attempts")
    status = original_repo_status()
    log(f"BATCH END ledger ${ledger.committed:.4f}, original repo clean={status['clean']}")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["recalibrate", "calibrate", "dryrun", "freeze", "batch"])
    ap.add_argument("--weights", choices=["tuned", "default"], default="tuned")
    ap.add_argument("--only", choices=["A", "B", "C"], default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "recalibrate":
        stage_recalibrate(args.weights)
    elif args.stage == "calibrate":
        asyncio.run(stage_calibrate())
    elif args.stage == "dryrun":
        asyncio.run(stage_dryrun(args.weights))
    elif args.stage == "freeze":
        stage_freeze(args.weights)
    else:
        sys.exit(asyncio.run(stage_batch(args.only)))


if __name__ == "__main__":
    main()
