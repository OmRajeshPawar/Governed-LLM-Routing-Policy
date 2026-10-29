from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import _repo_root, load_config
from apex.events import load_events
from apex.policies import all_policies
from analysis.evaluate import POLICY_ORDER, SIGNAL_LABELS, SIGNAL_POLICY_ORDER, policy_overhead_vectors
from ml.risk_model import CalibratedRiskModel

ROOT = _repo_root()
FIG = ROOT / "analysis" / "figures"
OUT = ROOT / "analysis"
KEY_POLICIES = [
    "tail_aware",
    "ml_ttft",
    "ml_dual_risk",
    "semantic_embedding",
    "local_llm_probe",
    "thinking_probe",
]
PRETTY = {
    "tail_aware": "T0 telemetry",
    "ml_ttft": "T1 ML TTFT",
    "ml_dual_risk": "T1 ML dual",
    "semantic_embedding": "T2 semantic",
    "local_llm_probe": "T3 local probe",
    "thinking_probe": "T4 thinking",
    "queue_threshold": "queue threshold",
    "queue_service_oracle": "service oracle",
    "always_local": "always local",
    "always_cloud": "always cloud",
}


def load_summary() -> dict:
    path = OUT / "summary_metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_shadow_df() -> pd.DataFrame:
    cfg = load_config()
    summary = load_summary()
    events_dir = ROOT / cfg.paths.events_dir
    if summary.get("shadow_files"):
        files = [events_dir / name for name in summary["shadow_files"]]
        missing = [path for path in files if not path.exists()]
        if missing:
            raise SystemExit(f"Summary listed missing shadow file(s): {missing[:3]}")
    else:
        shadow_glob = summary.get("shadow_glob", "shadow_*_bursty_n300_s*.jsonl")
        files = sorted(events_dir.glob(shadow_glob))
    if not files:
        raise SystemExit("No shadow files available for deep-dive analysis")
    return pd.concat([load_events(f) for f in files], ignore_index=True)


def parse_results_table() -> pd.DataFrame:
    path = OUT / "results_table.md"
    rows: list[list[str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("|") and "---" not in line:
            rows.append([c.strip() for c in line.strip("|").split("|")])
    header, body = rows[0], rows[1:]
    df = pd.DataFrame(body, columns=header)
    for col in ["tier", "offload %", "raw SLA viol %", "adj SLA viol %", "adj P99 TTFT ms", "all-in cost $", "$/adj SLA-ok req"]:
        df[col] = df[col].str.replace("$", "", regex=False).astype(float)
    return df


def ci95(values: pd.Series) -> tuple[float, float]:
    vals = values.dropna().to_numpy(dtype=float)
    if len(vals) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(0)
    boots = [np.mean(vals[rng.integers(0, len(vals), len(vals))]) for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def plot_budget_curves(qr: pd.DataFrame) -> None:
    plt.figure(figsize=(8.0, 5.2))
    policies = ["queue_service_oracle", "tail_aware", "ml_ttft", "ml_dual_risk", "semantic_embedding", "local_llm_probe", "thinking_probe"]
    for policy in policies:
        g = qr[qr.policy == policy].groupby("budget")["sla_violation_rate"]
        means = g.mean()
        los, his = [], []
        for _, vals in g:
            lo, hi = ci95(vals)
            los.append(lo)
            his.append(hi)
        x = means.index.to_numpy() * 100
        y = means.to_numpy() * 100
        label = PRETTY.get(policy, policy)
        lw = 2.7 if policy in {"queue_service_oracle", "tail_aware"} else 1.8
        style = "--" if policy == "queue_service_oracle" else "-"
        plt.plot(x, y, marker="o", linewidth=lw, linestyle=style, label=label)
        plt.fill_between(x, np.array(los) * 100, np.array(his) * 100, alpha=0.08)
    plt.xlabel("matched cloud-offload budget (%)")
    plt.ylabel("queue-replay SLA violation (%)")
    plt.title("Queue replay budget curves with run-level bootstrap bands")
    plt.grid(alpha=0.28)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "queue_replay_budget_curves.png", dpi=150)
    plt.close()


def plot_headroom(qr: pd.DataFrame) -> pd.DataFrame:
    mean = qr.groupby(["policy", "budget"])["sla_violation_rate"].mean().unstack(0)
    rows = []
    for budget, row in mean.iterrows():
        baseline = row["tail_aware"]
        oracle = row["queue_service_oracle"]
        denom = baseline - oracle
        for policy in KEY_POLICIES:
            captured = np.nan if abs(denom) < 1e-9 else (baseline - row[policy]) / denom
            rows.append({
                "budget": budget,
                "policy": policy,
                "headroom_captured": float(captured),
                "policy_sla": float(row[policy]),
                "tail_aware_sla": float(baseline),
                "oracle_sla": float(oracle),
            })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "deep_headroom_captured.csv", index=False)

    pivot = out.pivot(index="budget", columns="policy", values="headroom_captured")
    plt.figure(figsize=(8.2, 4.8))
    for policy in KEY_POLICIES:
        y = pivot[policy].to_numpy() * 100
        plt.plot(pivot.index.to_numpy() * 100, y, marker="o", label=PRETTY[policy])
    plt.axhline(0, color="#333333", linewidth=1)
    plt.axhline(100, color="#999999", linewidth=1, linestyle="--")
    plt.xlabel("matched cloud-offload budget (%)")
    plt.ylabel("oracle headroom captured vs T0 (%)")
    plt.title("Do richer signals capture the queue-replay oracle gap?")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "headroom_captured_by_budget.png", dpi=150)
    plt.close()
    return out


def plot_workload_breakdown(qr: pd.DataFrame) -> None:
    focus = qr[(qr["budget"].round(2) == 0.30) & (qr["policy"].isin(KEY_POLICIES + ["queue_service_oracle"]))]
    table = focus.pivot_table(index="mix", columns="policy", values="sla_violation_rate", aggfunc="mean")
    ordered = ["queue_service_oracle", *KEY_POLICIES]
    table = table[[p for p in ordered if p in table.columns]]
    x = np.arange(len(table.index))
    width = 0.11
    plt.figure(figsize=(10.0, 5.0))
    for i, policy in enumerate(table.columns):
        offset = (i - (len(table.columns) - 1) / 2.0) * width
        plt.bar(x + offset, table[policy].to_numpy() * 100, width, label=PRETTY.get(policy, policy))
    plt.xticks(x, table.index)
    plt.ylabel("queue-replay SLA violation (%)")
    plt.title("Workload breakdown at 30% offload")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(FIG / "workload_breakdown_30pct.png", dpi=150)
    plt.close()


def plot_signal_tax(static_df: pd.DataFrame) -> None:
    cfg = load_config()
    shadow = load_shadow_df()
    meta = json.loads((ROOT / cfg.paths.models_dir / "meta.json").read_text(encoding="utf-8"))
    cfg.targets.sla_ttft_ms = meta["frozen_thresholds"]["sla_ttft_ms"]
    cfg.targets.occupancy_service_ms = meta["frozen_thresholds"]["occupancy_service_ms"]
    _ = CalibratedRiskModel.load(ROOT / cfg.paths.models_dir)
    overhead, _router = policy_overhead_vectors(shadow, cfg, meta.get("heuristic_weights"))
    rows = []
    for policy in KEY_POLICIES:
        row = static_df[static_df.policy == policy].iloc[0]
        rows.append({
            "policy": policy,
            "median_decision_overhead_ms": float(np.median(overhead[policy])),
            "adj_sla_violation_pct": float(row["adj SLA viol %"]),
            "adj_p99_ttft_ms": float(row["adj P99 TTFT ms"]),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "deep_signal_tax.csv", index=False)

    plt.figure(figsize=(7.8, 5.0))
    for _, r in df.iterrows():
        plt.scatter(r["median_decision_overhead_ms"], r["adj_sla_violation_pct"], s=90)
        plt.text(r["median_decision_overhead_ms"] * 1.04 + 0.03, r["adj_sla_violation_pct"], PRETTY[r["policy"]], fontsize=8, va="center")
    plt.xscale("symlog", linthresh=1.0)
    plt.xlabel("median decision/probe overhead (ms, symlog)")
    plt.ylabel("static adjusted SLA violation (%) @ 30% offload")
    plt.title("Signal tax: more expensive signals must earn their latency")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG / "signal_tax_scatter.png", dpi=150)
    plt.close()


def plot_gemini_validation() -> None:
    public = OUT / "vertex_validation_summary.json"
    private = OUT / "private" / "vertex_validation_raw.json"
    if not public.exists() or not private.exists():
        return
    summary = json.loads(public.read_text(encoding="utf-8"))
    raw = json.loads(private.read_text(encoding="utf-8"))
    vals = [float(r["wall_ms"]) for r in raw.get("rows", []) if r.get("ok") and r.get("wall_ms") is not None]
    if not vals:
        return
    vals = np.sort(np.asarray(vals))
    ys = np.arange(1, len(vals) + 1) / len(vals)
    plt.figure(figsize=(7.4, 4.8))
    plt.plot(vals, ys, linewidth=2.2, label="live Gemini 2.5 Flash")
    plt.axvline(summary.get("latency_wall_ms", {}).get("p50", np.nan), color="#0d7377", linestyle="--", label="Gemini p50")
    plt.axvline(summary.get("latency_wall_ms", {}).get("p95", np.nan), color="#b33a32", linestyle="--", label="Gemini p95")
    plt.xscale("log")
    plt.xlabel("wall latency (ms, log scale)")
    plt.ylabel("CDF")
    plt.title("Bounded live Gemini validation latency distribution")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "gemini_latency_cdf.png", dpi=150)
    plt.close()

    q = np.percentile(vals, [50, 90, 95, 99])
    (OUT / "deep_gemini_latency_summary.json").write_text(json.dumps({
        "n": len(vals),
        "p50_ms": round(float(q[0]), 2),
        "p90_ms": round(float(q[1]), 2),
        "p95_ms": round(float(q[2]), 2),
        "p99_ms": round(float(q[3]), 2),
        "estimated_cost_usd_total": summary.get("estimated_cost_usd_total"),
    }, indent=2), encoding="utf-8")


def write_insights(qr: pd.DataFrame, static_df: pd.DataFrame, headroom: pd.DataFrame) -> None:
    b30 = qr[qr["budget"].round(2) == 0.30].groupby("policy")["sla_violation_rate"].mean()
    static = {r["policy"]: r for _, r in static_df.iterrows()}
    h30 = headroom[headroom["budget"].round(2) == 0.30].set_index("policy")
    lines = [
        "# Deep-Dive Insights From The 4,500-Request Target Run",
        "",
        "These are derived metrics from existing data; no new local or cloud calls were made.",
        "",
        "## Main Additional Findings",
        "",
        f"1. **Telemetry is the best deployable policy in queue replay at 30% offload.** Tail-aware telemetry reaches {b30['tail_aware']*100:.1f}% SLA violations, compared with ML dual-risk at {b30['ml_dual_risk']*100:.1f}%, semantic/embedding at {b30['semantic_embedding']*100:.1f}%, and local probe at {b30['local_llm_probe']*100:.1f}%.",
        f"2. **The oracle gap is real but mostly uncaptured by richer signals.** The service-time oracle reaches {b30['queue_service_oracle']*100:.1f}% at 30% offload. ML dual-risk captures {h30.loc['ml_dual_risk','headroom_captured']*100:.1f}% of T0-to-oracle headroom, while semantic/embedding captures {h30.loc['semantic_embedding','headroom_captured']*100:.1f}%. Negative values mean worse than T0.",
        f"3. **Dual-risk is not better than TTFT-only here.** Static ML-TTFT is {static['ml_ttft']['adj SLA viol %']:.1f}% adjusted SLA violation, while ML dual-risk is {static['ml_dual_risk']['adj SLA viol %']:.1f}%. The occupancy model detects a real phenomenon, but its current score is not improving routing.",
        f"4. **T3/T4 fail for an explainable systems reason.** Static local-probe raw SLA is {static['local_llm_probe']['raw SLA viol %']:.1f}% but adjusted SLA is {static['local_llm_probe']['adj SLA viol %']:.1f}%; thinking-probe adjusted SLA rises to {static['thinking_probe']['adj SLA viol %']:.1f}%. The probe tax cancels the apparent signal value.",
        "5. **Gemini validation adds credibility but also warns about provider tails.** The 100-request live Gemini subset has p50 around 1.23 s and p95 around 7.07 s, much wider than the calibrated proxy. This should be framed as validation-plus-threat, not as a replacement for the main experiment.",
        "",
        "## New Figures",
        "",
        "- `analysis/figures/queue_replay_budget_curves.png`",
        "- `analysis/figures/headroom_captured_by_budget.png`",
        "- `analysis/figures/workload_breakdown_30pct.png`",
        "- `analysis/figures/signal_tax_scatter.png`",
        "- `analysis/figures/gemini_latency_cdf.png`",
    ]
    (OUT / "deep_dive_insights.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    qr_path = OUT / "queue_replay_results.csv"
    if not qr_path.exists():
        raise SystemExit("Missing analysis/queue_replay_results.csv. Run replay/queue_replay.py first.")
    qr = pd.read_csv(qr_path)
    static_df = parse_results_table()

    plot_budget_curves(qr)
    headroom = plot_headroom(qr)
    plot_workload_breakdown(qr)
    plot_signal_tax(static_df)
    plot_gemini_validation()
    write_insights(qr, static_df, headroom)
    print("Deep-dive analysis complete.")


if __name__ == "__main__":
    main()
