"""Offline feasibility frontier and a prompt-only static baseline. CPU only, read-only.

Question the live study cannot answer without more GPU time: at which cloud-call
quota does each router first meet the 10% TTFT-SLO violation target, and what
does that quota cost relative to always-cloud? Answered by replaying the July
shadow corpus (the same offline corpus used for the paper's offline results).

Adds a prompt-only static baseline: a router that scores requests using only
features visible before execution (prompt length, requested output length,
code flag), with no telemetry and no quality labels. This is the closest
analogue to learned quality routers (RouteLLM, CARROT) that is implementable
here, since this study has no per-request quality labels.

Run with:  python frontier_eval.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "frontier_eval.json"
os.environ["APEX_SHADOW_GLOB"] = "shadow_*_bursty_n300_s*.jsonl"
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import evaluate as ev  # noqa: E402

from apex.core import _repo_root, load_config  # noqa: E402
from apex.events import load_events, resolve_event_files  # noqa: E402
from ml.risk_model import CalibratedRiskModel  # noqa: E402

SLO_TARGET = 0.10
BUDGETS = np.round(np.arange(0.10, 0.701, 0.05), 2)
BOOT = 500
POLICIES = ["tail_aware", "queue_threshold", "ml_dual_risk", "prompt_static"]

cfg = load_config()
files, _ = resolve_event_files(_repo_root() / cfg.paths.events_dir)
df = pd.concat([load_events(f) for f in files], ignore_index=True)
meta = json.loads((_repo_root() / cfg.paths.models_dir / "meta.json").read_text(encoding="utf-8"))
fr = meta["frozen_thresholds"]
sla = float(fr["sla_ttft_ms"])
cfg.targets.sla_ttft_ms = sla
cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]
hw = meta.get("heuristic_weights")
ml = CalibratedRiskModel.load(_repo_root() / cfg.paths.models_dir)

local = df["end_to_end_ttft_ms"].values.astype(float)
base, per_tok = ev.meta_calib(cfg, meta)
scores = ev.policy_scores(df, cfg, ml, hw)
overhead, _router_cost = ev.policy_overhead_vectors(df, cfg, hw)
cloud, cost = ev.build_cloud_outcomes(df, cfg, base, per_tok, 1.0)
N = len(df)


def z(values) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    s = v.std()
    return (v - v.mean()) / s if s > 0 else np.zeros_like(v)


# Prompt-only static baseline: no telemetry, no labels, no model. Offload the
# requests that are longest to generate and most likely to be code.
scores["prompt_static"] = (z(df["max_output_tokens"].values) + 0.5 * z(df["prompt_chars"].values)
                           + df["is_code"].fillna(0).values.astype(float))
overhead["prompt_static"] = np.zeros(N)

runs = df["run_id"].astype(str).values
run_ids = sorted(set(runs))
run_idx = {r: np.where(runs == r)[0] for r in run_ids}
mixes = {r: re.search(r"(synthetic|real_sample|reasoning_coding)", r).group(1) for r in run_ids}
by_mix: dict[str, list[str]] = {}
for r in run_ids:
    by_mix.setdefault(mixes[r], []).append(r)

rng = np.random.default_rng(0)
IDX = []
for _ in range(BOOT):
    parts = []
    for rs in by_mix.values():
        for p in rng.integers(0, len(rs), len(rs)):
            parts.append(run_idx[rs[p]])
    IDX.append(np.concatenate(parts).astype(np.int32))


def realized(name: str, budget: float):
    k = int(round(budget * N))
    order = np.argsort(-scores[name], kind="stable")
    off = np.zeros(N, dtype=bool)
    off[order[:k]] = True
    return np.where(off, cloud, local) + overhead[name], off


out = {"sla_ttft_ms": sla, "slo_target_violation_rate": SLO_TARGET, "n_requests": int(N),
       "n_runs": len(run_ids), "bootstrap_resamples": BOOT, "budgets": BUDGETS.tolist(),
       "corpus": "July shadow corpus, bursty n=300 runs (offline replay; cloud tier is the calibrated model)",
       "prompt_static_definition": "z(max_output_tokens) + 0.5 z(prompt_chars) + is_code; no telemetry, no labels",
       "frontier": {}, "feasible_quota": {}}

for name in POLICIES:
    rows = []
    for b in BUDGETS:
        t, off = realized(name, float(b))
        viol = (t > sla).mean()
        band = [float(np.percentile([(t[i] > sla).mean() for i in IDX], q)) for q in (2.5, 97.5)]
        rows.append({"budget": float(b), "violation_rate": float(viol),
                     "violation_ci95": band,
                     "cost_share_of_always_cloud": float(cost[off].sum() / cost.sum())})
    out["frontier"][name] = rows
    feasible = next((r for r in rows if r["violation_rate"] <= SLO_TARGET), None)
    out["feasible_quota"][name] = feasible
    tag = (f"quota {feasible['budget']*100:.0f}%, {feasible['violation_rate']*100:.2f}% viol, "
           f"{feasible['cost_share_of_always_cloud']:.3f}x always-cloud cost") if feasible else "never within 70%"
    print(f"{name:<16} first meets 10% SLO at {tag}")

print("\nviolation rate (%) by quota")
print("quota  " + "  ".join(f"{b*100:4.0f}" for b in BUDGETS))
for name in POLICIES:
    print(f"{name:<14} " + "  ".join(f"{r['violation_rate']*100:4.1f}" for r in out["frontier"][name]))

# LaTeX rows for the paper
lines = []
for name in POLICIES:
    f = out["feasible_quota"][name]
    label = {"tail_aware": "Telemetry heuristic (T0)", "queue_threshold": "Queue threshold",
             "ml_dual_risk": "Tabular ML dual-risk (T1)", "prompt_static": "Prompt-only static"}[name]
    if f is None:
        lines.append(f"{label} & $>70$ & n/a & n/a \\\\")
    else:
        lines.append(f"{label} & {f['budget']*100:.0f} & {f['violation_rate']*100:.1f} & "
                     f"{f['cost_share_of_always_cloud']:.2f} \\\\")
out["latex_rows"] = lines
print("\n".join(lines))
OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\nwrote", OUT)
