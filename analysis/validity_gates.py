from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex.core import load_config
from apex.policies import AlwaysLocal
from replay.event_simulator import (ClosedLoopSimulator, load_local_execution_trace,
                                    stage_a_metrics)


def _rel(a: float, b: float) -> float:
    return abs(a - b) / max(abs(b), 1e-9)


def evaluate_stage_a(event_path: Path) -> dict:
    cfg = load_config()
    rows = load_local_execution_trace(event_path)
    sim = ClosedLoopSimulator(cfg, rows, AlwaysLocal(), threshold=0.0,
                              seed=int(rows[0]["seed"])).run()
    report = stage_a_metrics(rows, sim)
    report["source"] = str(event_path.relative_to(ROOT))
    report["n_requests"] = len(rows)
    return report


def evaluate_stage_b(path: Path) -> dict:
    if not path.exists():
        return {
            "stage": "B_contention_change_validity", "status": "not_run",
            "required_input": str(path.relative_to(ROOT)),
            "claim_mode": "fixed_service_time_scenario_analysis",
            "interpretation": "No routed same-request local validation file exists. Stored all-local service times are not yet accepted as empirical counterfactuals.",
        }
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    ratios, apes = [], []
    residual_inputs = {k: [] for k in ("prompt_tokens", "output_cap", "active_local_jobs",
                                       "gpu_util", "gpu_temp_c")}
    residuals = []
    strata = {0: [], 1: []}
    for r in rows:
        base, routed = float(r["all_local_service_ms"]), float(r["routed_service_ms"])
        ratio = routed / max(base, 1e-9)
        ratios.append(ratio)
        apes.append(abs(routed - base) / max(base, 1e-9))
        residuals.append(routed - base)
        active = min(1, int(r["active_local_jobs"]))
        strata[active].append(ratio)
        for k in residual_inputs:
            residual_inputs[k].append(float(r[k]))
    try:
        from scipy.stats import spearmanr
        correlations = {k: abs(float(spearmanr(v, residuals).statistic))
                        for k, v in residual_inputs.items()}
    except Exception:
        correlations = {k: None for k in residual_inputs}
    metrics = {
        "median_service_ratio": float(np.median(ratios)),
        "p90_absolute_percentage_error": float(np.quantile(apes, .90)),
        "median_ratio_zero_competitor": float(np.median(strata[0])) if strata[0] else None,
        "median_ratio_one_competitor": float(np.median(strata[1])) if strata[1] else None,
        "absolute_spearman_residual_correlations": correlations,
    }
    gates = {
        "median_ratio": .90 <= metrics["median_service_ratio"] <= 1.10,
        "p90_ape": metrics["p90_absolute_percentage_error"] <= .25,
        "zero_competitor": metrics["median_ratio_zero_competitor"] is not None and .85 <= metrics["median_ratio_zero_competitor"] <= 1.15,
        "one_competitor": metrics["median_ratio_one_competitor"] is not None and .85 <= metrics["median_ratio_one_competitor"] <= 1.15,
        "residual_correlations": all(v is not None and v <= .30 for v in correlations.values()),
    }
    aggregate = json.loads(path.with_suffix(".aggregate.json").read_text(encoding="utf-8")) if path.with_suffix(".aggregate.json").exists() else {}
    aggregate_gates = {
        "route_agreement": aggregate.get("route_agreement", 0) >= .95,
        "offload_rate_difference": aggregate.get("offload_rate_difference_pp", 999) <= 2.0,
        "mean_queue_wait_error": aggregate.get("mean_queue_wait_relative_error", 999) <= .10,
        "p95_ttft_error": aggregate.get("p95_ttft_relative_error", 999) <= .10,
        "completion_order": aggregate.get("completion_order_kendall", 0) >= .95,
    }
    passed = all(gates.values()) and all(aggregate_gates.values())
    return {
        "stage": "B_contention_change_validity", "status": "pass" if passed else "fail",
        "n_paired_local_requests": len(rows), "metrics": metrics,
        "gates": gates, "aggregate_metrics": aggregate,
        "aggregate_gates": aggregate_gates,
        "claim_mode": "bounded_local_counterfactual" if passed else "fixed_service_time_scenario_analysis",
        "interpretation": "Stage B tests whether routing-induced contention changes invalidate stored local service times.",
    }


def evaluate_cloud(path: Path) -> dict:
    if not path.exists():
        return {
            "stage": "cloud_proxy_validity", "status": "not_run",
            "required_input": str(path.relative_to(ROOT)),
            "claim_mode": "frozen_analytical_cloud_scenario_model",
            "freshness": "not_established",
            "interpretation": "Existing July 6 calls measured full wall latency, not the preregistered streaming TTFT cells; they cannot promote the proxy.",
        }
    raw = json.loads(path.read_text(encoding="utf-8"))
    metrics = raw.get("validity_metrics", {})
    gates = {
        "p50_ttft_error": metrics.get("p50_ttft_relative_error", 999) <= .20,
        "p95_ttft_error": metrics.get("p95_ttft_relative_error", 999) <= .30,
        "failure_rate_error": metrics.get("failure_rate_absolute_error_pp", 999) <= 5.0,
        "prompt_band_errors": all(v <= .25 for v in metrics.get("median_ttft_relative_error_by_prompt_band", {}).values()) and len(metrics.get("median_ttft_relative_error_by_prompt_band", {})) == 3,
        "concurrency_independence": metrics.get("live_median_concurrency_difference", 999) <= .20,
        "predictive_interval_coverage": .80 <= metrics.get("nominal_90_interval_coverage", -1) <= 1.00,
        "complete_cells": raw.get("n_attempted") == 60,
        "exact_model_identifier_recorded": bool(raw.get("model_resolution_verified", False)),
    }
    passed = all(gates.values())
    return {
        "stage": "cloud_proxy_validity", "status": "pass" if passed else "fail",
        "provider": raw.get("provider"), "model": raw.get("resolved_model_identifier"),
        "requested_model": raw.get("requested_model"),
        "region": raw.get("region"), "created_utc": raw.get("created_utc"),
        "metrics": metrics, "gates": gates,
        "claim_mode": "bounded_calibrated_cloud_counterfactual" if passed else "frozen_analytical_cloud_scenario_model",
        "freshness_rule": "Recalibrate after seven days or any model, region, SDK, output, thinking, or endpoint change.",
    }


def _write_report(path: Path, title: str, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [f"# {title}", "", f"**Status:** `{report['status']}`", "",
             report.get("interpretation", ""), "", "```json",
             json.dumps(report, indent=2), "```", ""]
    path.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage-a-event", default="data/events/shadow_synthetic_bursty_n300_s7.jsonl")
    ap.add_argument("--stage-b-pairs", default="data/validity/stage_b_local_pairs.jsonl")
    ap.add_argument("--cloud-summary",
                    default="analysis/results/cloud_streaming_validity_summary.json")
    ap.add_argument("--out-dir", default="deliverables/APEX_LLM_FINAL/evidence")
    args = ap.parse_args()
    out = ROOT / args.out_dir
    stage_a = evaluate_stage_a(ROOT / args.stage_a_event)
    stage_b = evaluate_stage_b(ROOT / args.stage_b_pairs)
    cloud = evaluate_cloud(ROOT / args.cloud_summary)
    local = {"status": "pass" if stage_a["status"] == stage_b["status"] == "pass" else
             ("fail" if "fail" in {stage_a["status"], stage_b["status"]} else "not_run"),
             "stage_a": stage_a, "stage_b": stage_b,
             "claim_mode": stage_b.get("claim_mode")}
    _write_report(out / "LocalCounterfactualValidityReport.md", "Local Counterfactual Validity Report", local)
    _write_report(out / "CloudProxyValidityReport.md", "Cloud Proxy Validity Report", cloud)
    print(json.dumps({"local": local["status"], "cloud": cloud["status"]}, indent=2))


if __name__ == "__main__":
    main()
