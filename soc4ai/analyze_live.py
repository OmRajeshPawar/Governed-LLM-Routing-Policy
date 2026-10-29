"""Post-batch analysis for the SOC4AI live study.

Blind rule: run on an experiment only after its full corpus has completed
(every run in the frozen protocol for that experiment has a .done marker).
Developed and tested on the throwaway dry run only.

Usage:
  python soc4ai/analyze_live.py --exp A      (confirmatory + nominal descriptives)
  python soc4ai/analyze_live.py --exp B      (chaos episodes)
  python soc4ai/analyze_live.py --exp C      (drift phases)
  python soc4ai/analyze_live.py --exp dry --dir soc4ai_out/dryrun   (pipeline test)
Outputs JSON under soc4ai_out/analysis/.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex.core import ttft_breach  # noqa: E402

OUT = ROOT / "soc4ai_out"
SLA_MS = 3839.5
# Optional: raw live-Gemini TTFTs from the earlier cloud-proxy validity check (not shipped).
JULY_RAW = Path(os.environ.get("APEX_CLOUD_VALIDITY_RAW",
                               str(ROOT / "analysis" / "private" / "cloud_streaming_validity_raw.json")))


# ------------------------------------------------------------------ loading
def load_run(jsonl: Path) -> tuple[pd.DataFrame, dict, pd.DataFrame | None]:
    manifest = json.loads(jsonl.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    spec = manifest.get("spec") or {}
    rows = []
    for line in jsonl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        o, d, x = e["outcome"], e["decision"], (e["outcome"].get("extra") or {})
        e2e = o.get("end_to_end_ttft_ms")
        ok = bool(o.get("ok"))
        rows.append({
            "run": manifest["run_id"], "experiment": spec.get("experiment", "dry"),
            "seed": spec.get("seed", manifest.get("config", {}).get("seed")),
            "mix": spec.get("mix"), "policy": manifest["policy"], "condition": spec.get("condition"),
            "trace_pos": e["trace_pos"], "route": d["route"], "reason": d["reason_code"],
            "executed_on": o.get("executed_on"), "ok": ok, "e2e_ttft_ms": e2e,
            "e2e_ms": o.get("end_to_end_ms"), "queue_wait_ms": o.get("queue_wait_ms"),
            "provider_ttft_ms": o.get("provider_ttft_ms"), "service_ms": o.get("service_ms"),
            "cost_usd": o.get("cost_usd") or 0.0, "circuit_state": d.get("circuit_state"),
            "budget_denied": d.get("budget_denied"), "telemetry_tier": d.get("telemetry_tier"),
            "regime": d.get("regime"), "phase": e.get("workload_phase"),
            "cloud_degraded": e.get("cloud_degraded"), "fault": x.get("fault"),
            "load_ms": x.get("load_ms"), "prompt_eval_ms": x.get("prompt_eval_ms"),
            "eval_ms": x.get("eval_ms"), "decode_tps": x.get("decode_tokens_per_s"),
            "stream_ttft_ms": x.get("stream_ttft_ms"),
            "arrival_unix": manifest["t0_unix_s"] + (o.get("arrival_offset_ms") or 0.0) / 1000.0,
            "violation": ttft_breach(e2e, SLA_MS, ok),
        })
    df = pd.DataFrame(rows)
    gpu_path = jsonl.with_suffix("").with_suffix(".gpu.csv")
    gpu_path = jsonl.parent / f"{jsonl.stem}.gpu.csv"
    gpu = pd.read_csv(gpu_path) if gpu_path.exists() else None
    return df, manifest, gpu


def load_dir(directory: Path) -> tuple[pd.DataFrame, dict, dict]:
    frames, manifests, gpus = [], {}, {}
    for jsonl in sorted(directory.glob("*.jsonl")):
        if not jsonl.with_suffix(".manifest.json").exists():
            continue
        df, m, g = load_run(jsonl)
        frames.append(df)
        manifests[m["run_id"]] = m
        gpus[m["run_id"]] = g
    return pd.concat(frames, ignore_index=True), manifests, gpus


def pct(values, q):
    v = sorted(x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x)))
    return float(v[max(0, min(len(v) - 1, math.ceil(q * len(v)) - 1))]) if v else None


# ---------------------------------------------------------------- telemetry
def energy_and_thermal(run_df: pd.DataFrame, gpu: pd.DataFrame | None) -> dict:
    if gpu is None or gpu.empty or gpu["power_w"].isna().all():
        return {"available": False}
    g = gpu.dropna(subset=["power_w"]).sort_values("unix_s")
    wh = float(np.trapezoid(g["power_w"], g["unix_s"]) / 3600.0) if len(g) > 1 else None
    local = run_df[run_df["executed_on"] == "local"]
    met_local = int(((local["violation"] == 0) & local["ok"]).sum())
    throttle_cols = [c for c in ("sw_power_cap", "hw_slowdown", "sw_thermal", "hw_thermal") if c in g]
    throttled = g[throttle_cols].fillna(0).sum(axis=1) > 0 if throttle_cols else pd.Series(False, index=g.index)
    out = {"available": True, "gpu_energy_wh": wh, "mean_power_w": float(g["power_w"].mean()),
           "max_temp_c": float(g["temp_c"].max()), "mean_temp_c": float(g["temp_c"].mean()),
           "throttled_sample_share": float(throttled.mean()),
           "wh_per_request": wh / len(run_df) if wh is not None and len(run_df) else None,
           "wh_per_slo_met_local_request": wh / met_local if wh is not None and met_local else None}
    # temperature at each local request's arrival vs its TTFT
    if len(local) > 5 and "temp_c" in g:
        temps = np.interp(local["arrival_unix"], g["unix_s"], g["temp_c"])
        ttft = local["e2e_ttft_ms"].astype(float)
        mask = ~np.isnan(ttft)
        if mask.sum() > 5:
            out["spearman_temp_vs_local_ttft"] = float(pd.Series(temps[mask]).rank().corr(pd.Series(ttft[mask].values).rank()))
    return out


def breach_decomposition(df: pd.DataFrame) -> dict:
    local = df[(df["executed_on"] == "local") & df["ok"]].copy()
    if local.empty:
        return {}
    local["prefill_ms"] = local["prompt_eval_ms"].fillna(0) + local["load_ms"].fillna(0)
    out = {}
    for label, sub in (("breached", local[local["violation"] == 1]), ("met", local[local["violation"] == 0])):
        if sub.empty:
            continue
        ttft = sub["e2e_ttft_ms"].astype(float)
        out[label] = {"n": int(len(sub)), "mean_ttft_ms": float(ttft.mean()),
                      "mean_queue_wait_ms": float(sub["queue_wait_ms"].mean()),
                      "mean_prefill_ms": float(sub["prefill_ms"].mean()),
                      "queue_share_of_ttft": float((sub["queue_wait_ms"] / ttft).mean()),
                      "mean_decode_tps": float(sub["decode_tps"].mean()) if sub["decode_tps"].notna().any() else None}
    return out


def gemini_latency(df: pd.DataFrame, calibration: dict | None) -> dict:
    cloud = df[(df["executed_on"] == "cloud") & df["ok"] & df["fault"].isna()]
    live = cloud["stream_ttft_ms"].dropna().astype(float).tolist()
    out = {"n_live": len(live), "p50_ms": pct(live, 0.5), "p95_ms": pct(live, 0.95),
           "p99_ms": pct(live, 0.99), "slo_miss_share": float(np.mean([v > SLA_MS for v in live])) if live else None}
    if JULY_RAW.exists():
        july = [r["ttft_ms"] for r in json.loads(JULY_RAW.read_text(encoding="utf-8"))["rows"]
                if r.get("ok") and r.get("ttft_ms") is not None]
        out["july_n"], out["july_p50_ms"], out["july_p95_ms"] = len(july), pct(july, 0.5), pct(july, 0.95)
    if calibration:
        out["calibration_p50_ms"], out["calibration_p95_ms"] = calibration.get("ttft_p50_ms"), calibration.get("ttft_p95_ms")
    return out


def run_summary(sub: pd.DataFrame) -> dict:
    ttft = sub["e2e_ttft_ms"].astype(float)
    return {"n": int(len(sub)), "violation_rate": float(sub["violation"].mean()),
            "p95_ttft_ms": pct(ttft.tolist(), 0.95), "cloud_share": float((sub["route"] == "cloud").mean()),
            "failed": int((~sub["ok"]).sum()), "cloud_failed": int(((sub["executed_on"] == "cloud") & ~sub["ok"]).sum()),
            "spend_usd": float(sub["cost_usd"].sum()),
            "wasted_spend_usd": float(sub.loc[(sub["executed_on"] == "cloud") & (sub["violation"] == 1), "cost_usd"].sum()),
            "budget_denials": int(sub["budget_denied"].fillna(False).astype(bool).sum())}


# -------------------------------------------------------------- experiments
def analyze_a(df: pd.DataFrame) -> dict:
    spec = importlib.util.spec_from_file_location("confirmatory_statistics", ROOT / "analysis" / "confirmatory_statistics.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    rows = []
    for r in df.itertuples():
        ttft = r.e2e_ttft_ms if (r.ok and r.e2e_ttft_ms is not None and not math.isnan(r.e2e_ttft_ms)) \
            else max(float(r.e2e_ms or 0.0), SLA_MS)
        rows.append({"seed": int(r.seed), "policy": r.policy, "sla_violation": bool(r.violation),
                     "end_to_end_ttft_ms": float(ttft)})
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    with open(OUT / "analysis" / "confirmatory_request_results.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    confirmatory = mod.analyze(rows)
    per_policy = {p: run_summary(g) for p, g in df.groupby("policy")}
    per_seed = {f"{s}_{p}": run_summary(g) for (s, p), g in df.groupby(["seed", "policy"])}
    return {"confirmatory_statistics": confirmatory, "per_policy": per_policy, "per_seed_policy": per_seed}


SEGMENTS_B = [("pre", 0, 96), ("outage", 96, 156), ("between", 156, 192), ("spike", 192, 252), ("post", 252, 300)]


def analyze_b(df: pd.DataFrame, manifests: dict) -> dict:
    out = {"segments": {}, "breaker": {}}
    for p, g in df.groupby("policy"):
        out["segments"][p] = {name: run_summary(g[(g["trace_pos"] >= a) & (g["trace_pos"] < b)])
                              for name, a, b in SEGMENTS_B}
        out["segments"][p]["all"] = run_summary(g)
    for run_id, m in manifests.items():
        if m.get("policy") != "guarded_governor":
            continue
        run_df = df[df["run"] == run_id]
        outage_start = run_df.loc[run_df["trace_pos"] == 96, "arrival_unix"]
        spike_start = run_df.loc[run_df["trace_pos"] == 192, "arrival_unix"]
        tr = m.get("breaker_transitions", [])
        info = {"transitions": [(t["from"], t["to"], t["reason"], round(t["unix_s"] - m["t0_unix_s"], 1)) for t in tr]}
        for label, start in (("outage", outage_start), ("spike", spike_start)):
            if start.empty:
                continue
            t0 = float(start.iloc[0])
            opens = [t for t in tr if t["to"] == "open" and t["unix_s"] >= t0]
            if opens:
                info[f"{label}_detect_s"] = round(opens[0]["unix_s"] - t0, 2)
                closes = [t for t in tr if t["to"] == "closed" and t["unix_s"] > opens[0]["unix_s"]]
                if closes:
                    info[f"{label}_recover_s"] = round(closes[0]["unix_s"] - opens[0]["unix_s"], 2)
        out["breaker"][run_id] = info
    return out


def analyze_c(df: pd.DataFrame) -> dict:
    return {p: {ph: run_summary(g[g["phase"] == ph]) for ph in g["phase"].dropna().unique()}
            for p, g in df.groupby("policy")}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "dry"], required=True)
    ap.add_argument("--dir", default=None)
    args = ap.parse_args()
    directory = Path(args.dir) if args.dir else OUT / "runs" / args.exp
    if args.exp != "dry":
        protocol = json.loads((OUT / "protocol.json").read_text(encoding="utf-8"))
        keys = [r["key"] for r in protocol["runs"] if r["experiment"] == args.exp]
        missing = [k for k in keys if not (directory / f"{k}.done").exists()]
        if missing:
            raise SystemExit(f"blind rule: experiment {args.exp} incomplete, {len(missing)} runs missing")
    df, manifests, gpus = load_dir(directory)
    calib_path = OUT / "calibration.json"
    calibration = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else None
    result = {"experiment": args.exp, "n_runs": len(manifests), "n_requests": int(len(df)),
              "gemini_latency": gemini_latency(df, calibration),
              "breach_decomposition": breach_decomposition(df),
              "telemetry": {rid: energy_and_thermal(df[df["run"] == rid], gpus.get(rid)) for rid in manifests}}
    if args.exp == "A":
        result["A"] = analyze_a(df)
    elif args.exp == "B":
        result["B"] = analyze_b(df, manifests)
    elif args.exp == "C":
        result["C"] = analyze_c(df)
    else:
        result["per_policy"] = {p: run_summary(g) for p, g in df.groupby("policy")}
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    out_path = OUT / "analysis" / f"live_{args.exp}.json"
    out_path.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(f"wrote {out_path} ({len(manifests)} runs, {len(df)} requests)")


if __name__ == "__main__":
    main()
