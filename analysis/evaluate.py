from __future__ import annotations

import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import load_config, _repo_root
from apex.events import load_events, resolve_event_files
from apex.core import FEATURE_NAMES
from apex.core import freeze_thresholds
from apex.policies import all_policies
from ml.risk_model import CalibratedRiskModel

POLICY_ORDER = ["always_local", "always_cloud", "queue_threshold",
                "tail_aware", "ml_ttft", "ml_dual_risk",
                "semantic_embedding", "local_llm_probe", "thinking_probe"]
SIGNAL_POLICY_ORDER = ["tail_aware", "ml_dual_risk", "semantic_embedding",
                       "local_llm_probe", "thinking_probe"]
COLORS = {"always_local": "#888", "always_cloud": "#bbb", "queue_threshold": "#1f77b4",
          "tail_aware": "#ff7f0e", "ml_ttft": "#2ca02c", "ml_dual_risk": "#d62728",
          "semantic_embedding": "#9467bd", "local_llm_probe": "#8c564b",
          "thinking_probe": "#e377c2"}
SIGNAL_LABELS = {
    "tail_aware": "T0 telemetry heuristic",
    "ml_dual_risk": "T1 tabular ML",
    "semantic_embedding": "T2 embedding signal",
    "local_llm_probe": "T3 local LLM probe",
    "thinking_probe": "T4 thinking probe",
}


def cloud_ttft_ms(row, cc, base, per_tok, R, jit):
    est = float(row["max_output_tokens"])
    return (cc.network_rtt_ms + base * R) * jit, est


def build_cloud_outcomes(df, cfg, base, per_tok, R):
    cc = cfg.providers.cloud
    rng = np.random.default_rng(cfg.seed)
    jit = np.exp(rng.normal(0, cc.jitter_sigma, len(df)))
    ttft = np.array([(cc.network_rtt_ms + base * R) * jit[i] for i in range(len(df))])
    est = df["max_output_tokens"].values.astype(float)
    ptok = (df["prompt_chars"].values / 4).clip(min=1)
    cost = (ptok + est) / 1000.0 * cc.cost_per_1k_tokens
    return ttft, cost


def policy_scores(df, cfg, ml, heuristic_weights):
    pols = all_policies(cfg, heuristic_weights)
    feats = df[FEATURE_NAMES].to_dict("records")
    scores = {}
    for name in POLICY_ORDER:
        p = pols[name]
        scores[name] = np.array([p.score(f, ml) for f in feats], dtype=float)
    return scores


def policy_overhead_vectors(df, cfg, heuristic_weights):
    pols = all_policies(cfg, heuristic_weights)
    overhead, router_cost = {}, {}
    prompt_chars = df["prompt_chars"].values.astype(float)
    max_out = df["max_output_tokens"].values.astype(float)
    for name in POLICY_ORDER:
        p = pols[name]
        base = float(getattr(p, "estimated_decision_ms", 0.0))
        if name == "semantic_embedding":
            vec = base + 0.002 * prompt_chars
        elif name == "local_llm_probe":
            vec = base + 0.10 * max_out
        elif name == "thinking_probe":
            vec = base + 0.35 * max_out
        else:
            vec = np.full(len(df), base)
        overhead[name] = np.asarray(vec, dtype=float)
        router_cost[name] = np.full(len(df), float(getattr(p, "estimated_router_cost_usd", 0.0)))
    return overhead, router_cost


def frontier(df, scores_one, local_ttft, cloud_ttft, cloud_cost, sla, budgets,
             decision_overhead=None, router_cost=None):
    order = np.argsort(-scores_one)
    N = len(df)
    decision_overhead = np.zeros(N) if decision_overhead is None else np.asarray(decision_overhead)
    router_cost = np.zeros(N) if router_cost is None else np.asarray(router_cost)
    out = []
    for b in budgets:
        k = int(round(b * N))
        offload = np.zeros(N, dtype=bool)
        offload[order[:k]] = True
        realized = np.where(offload, cloud_ttft, local_ttft) + decision_overhead
        viol = np.mean(realized > sla)
        cost = cloud_cost[offload].sum() + router_cost.sum()
        out.append((offload.mean(), viol, cost, np.percentile(realized, 99)))
    return np.array(out)


def main():
    cfg = load_config()
    out_root = os.getenv("APEX_ANALYSIS_OUT")
    if out_root:
        fig_dir = _repo_root() / out_root / "figures"
    else:
        fig_dir = _repo_root() / cfg.paths.figures_dir
    fig_dir.mkdir(parents=True, exist_ok=True)
    events_dir = _repo_root() / cfg.paths.events_dir

    shadow_files, shadow_source = resolve_event_files(events_dir)
    if not shadow_files:
        raise SystemExit(f"No shadow files matched {shadow_source!r}. Run shadow collection first.")
    df = pd.concat([load_events(f) for f in shadow_files], ignore_index=True)
    print(f"Evaluating {len(df)} rows from {len(shadow_files)} shadow file(s), source={shadow_source}.")
    meta = json.loads((_repo_root() / cfg.paths.models_dir / "meta.json").read_text())
    fr = meta["frozen_thresholds"]
    sla = fr["sla_ttft_ms"]
    cfg.targets.sla_ttft_ms = fr["sla_ttft_ms"]
    cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]
    heuristic_weights = meta.get("heuristic_weights")
    ml = CalibratedRiskModel.load(_repo_root() / cfg.paths.models_dir)

    local_ttft = df["end_to_end_ttft_ms"].values.astype(float)
    base, per_tok = meta_calib(cfg, meta)
    scores = policy_scores(df, cfg, ml, heuristic_weights)
    decision_overhead, router_cost = policy_overhead_vectors(df, cfg, heuristic_weights)
    budgets = np.linspace(0, 1, 21)

    cloud_ttft, cloud_cost = build_cloud_outcomes(df, cfg, base, per_tok, 1.0)
    fronts = {}
    plt.figure(figsize=(7, 5))
    for name in POLICY_ORDER:
        f = frontier(df, scores[name], local_ttft, cloud_ttft, cloud_cost, sla, budgets)
        fronts[name] = f
        if name in ("always_local", "always_cloud"):
            plt.scatter(f[0, 0] if name == "always_local" else f[-1, 0],
                        f[0, 1] if name == "always_local" else f[-1, 1],
                        s=80, color=COLORS[name], zorder=5,
                        label=name, marker="s")
        else:
            plt.plot(f[:, 0], f[:, 1], "-o", ms=3, color=COLORS[name], label=name)
    plt.xlabel("offload rate (fraction sent to cloud)")
    plt.ylabel("SLA violation rate (TTFT > %.0f ms)" % sla)
    plt.title("Cost-QoS frontier: SLA violations vs offload budget\n(lower-left is better; matched budgets)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fig_dir / "cost_qos_frontier.png", dpi=130); plt.close()

    signal_fronts = {}
    plt.figure(figsize=(7.2, 5.2))
    for name in SIGNAL_POLICY_ORDER:
        f = frontier(df, scores[name], local_ttft, cloud_ttft, cloud_cost, sla, budgets,
                     decision_overhead[name], router_cost[name])
        signal_fronts[name] = f
        plt.plot(f[:, 2], f[:, 1] * 100, "-o", ms=3, color=COLORS[name],
                 label=SIGNAL_LABELS[name])
    plt.xlabel("all-in cost proxy ($ cloud + router)")
    plt.ylabel("SLA violation % with decision overhead")
    plt.title("Signal-cost frontier: richer signals must earn their overhead")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(fig_dir / "signal_cost_frontier.png", dpi=130); plt.close()

    plot_decision_overhead_cdf(decision_overhead, fig_dir / "decision_overhead_cdf.png")
    plot_signal_penalty_waterfall(df, decision_overhead, fig_dir / "co_located_signal_penalty_waterfall.png")

    p_ttft = np.array([ml.p_ttft(f) for f in df[FEATURE_NAMES].to_dict("records")])
    y_ttft = (local_ttft > sla).astype(int)
    plot_calibration(y_ttft, p_ttft, fig_dir / "calibration.png")

    plot_latency_cdf(df, scores, local_ttft, cloud_ttft, fig_dir / "latency_cdf.png", budget=0.30)

    rq0_speed_sweep(df, cfg, scores, local_ttft, base, per_tok, sla,
                    fig_dir / "rq0_speed_sweep.png")

    table = results_table(df, scores, local_ttft, cloud_ttft, cloud_cost,
                          decision_overhead, router_cost, sla, budget=0.30)
    (fig_dir.parent / "results_table.md").write_text(table)
    (fig_dir.parent / "signal_cost_results_table.md").write_text(table)
    print(table)

    rq0 = rq0_verdict(df, scores, local_ttft, cloud_ttft, sla, budget=0.30)
    selective = selective_label_demo(df, scores["tail_aware"], local_ttft, sla, budget=0.30)
    oracle_gap_static(df, cfg, scores, local_ttft, cloud_ttft, cloud_cost, sla,
                      decision_overhead, router_cost, fig_dir / "static_oracle_gap.png")
    utilization_heatmap_static(df, scores, local_ttft, cloud_ttft, sla,
                               decision_overhead, fig_dir / "static_utilization_regime_heatmap.png")
    summary = {
        "created_unix": time.time(),
        "n_rows": int(len(df)),
        "shadow_source": shadow_source,
        "shadow_glob": os.getenv("APEX_SHADOW_GLOB", "shadow_*.jsonl"),
        "shadow_files": [path.name for path in shadow_files],
        "sla_ttft_ms": float(sla),
        "rq0": rq0,
        "selective_label": selective,
    }
    (fig_dir.parent / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nFigures written to {fig_dir}")


def meta_calib(cfg, meta):
    cc = cfg.providers.cloud
    if cc.base_ttft_ms is not None and cc.per_token_ms is not None:
        return cc.base_ttft_ms, cc.per_token_ms
    cal = meta.get("calibration", {})
    return cal.get("base_ttft_ms", 204.0), cal.get("per_token_ms", 37.0)


def plot_calibration(y, p, path):
    from sklearn.calibration import calibration_curve
    plt.figure(figsize=(5.5, 5.5))
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="perfect")
    try:
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=6, strategy="quantile")
        plt.plot(mean_pred, frac_pos, "o-", color="#d62728", label="ML TTFT risk")
    except Exception:
        pass
    plt.xlabel("predicted P(breach)"); plt.ylabel("observed breach frequency")
    plt.title("Calibration of the ML TTFT-breach risk model")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_latency_cdf(df, scores, local_ttft, cloud_ttft, path, budget):
    plt.figure(figsize=(7, 5))
    N = len(df)
    for name in ("always_local", "tail_aware", "ml_dual_risk"):
        s = scores[name]
        order = np.argsort(-s)
        k = int(round(budget * N)) if name != "always_local" else 0
        offload = np.zeros(N, dtype=bool); offload[order[:k]] = True
        realized = np.where(offload, cloud_ttft, local_ttft)
        xs = np.sort(realized)
        ys = np.arange(1, N + 1) / N
        plt.plot(xs, ys, label=f"{name} (offload {offload.mean()*100:.0f}%)",
                 color=COLORS[name])
    plt.xscale("log")
    plt.xlabel("realized TTFT (ms, log scale)"); plt.ylabel("CDF")
    plt.title("Realized TTFT distribution at matched ~30% offload budget")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_decision_overhead_cdf(decision_overhead, path):
    plt.figure(figsize=(7, 5))
    for name in SIGNAL_POLICY_ORDER:
        xs = np.sort(np.asarray(decision_overhead[name], dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        plt.plot(xs, ys, label=SIGNAL_LABELS[name], color=COLORS[name])
    plt.xscale("symlog", linthresh=1.0)
    plt.xlabel("decision overhead (ms)")
    plt.ylabel("CDF")
    plt.title("Decision overhead by signal tier\n(T2-T4 estimates until live probes are enabled)")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def plot_signal_penalty_waterfall(df, decision_overhead, path):
    pressure = (
        df["active_local"].fillna(0).values.astype(float) / 2.0
        + df["pending_local"].fillna(0).values.astype(float) / 3.0
    ).clip(min=0.0)
    rows = []
    for name in SIGNAL_POLICY_ORDER:
        overhead = np.asarray(decision_overhead[name], dtype=float)
        if name in ("tail_aware", "ml_dual_risk", "semantic_embedding"):
            interference = np.zeros_like(overhead)
        else:
            interference = overhead * np.minimum(0.5, 0.12 * pressure)
        rows.append({
            "policy": name,
            "decision_or_probe_ms": float(np.median(overhead)),
            "estimated_queue_interference_ms": float(np.median(interference)),
        })
    x = np.arange(len(rows))
    wall = np.array([r["decision_or_probe_ms"] for r in rows])
    qint = np.array([r["estimated_queue_interference_ms"] for r in rows])
    labels = [SIGNAL_LABELS[r["policy"]] for r in rows]
    plt.figure(figsize=(8.2, 4.8))
    plt.bar(x, wall, color="#4c78a8", label="decision/probe wall time")
    plt.bar(x, qint, bottom=wall, color="#f58518", label="estimated same-GPU queue penalty")
    plt.yscale("symlog", linthresh=1.0)
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("median added latency (ms, symlog)")
    plt.title("Co-located signal penalty waterfall\n(estimated until live probe A/B run)")
    plt.legend(fontsize=8); plt.grid(axis="y", alpha=0.25); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def rq0_speed_sweep(df, cfg, scores, local_ttft, base, per_tok, sla, path):
    Rs = [0.5, 1.0, 2.0]
    N = len(df); budget = 0.30; k = int(round(budget * N))
    res = {"tail_aware": [], "ml_dual_risk": []}
    for R in Rs:
        cloud_ttft, _ = build_cloud_outcomes(df, cfg, base, per_tok, R)
        for name in res:
            order = np.argsort(-scores[name])
            offload = np.zeros(N, dtype=bool); offload[order[:k]] = True
            realized = np.where(offload, cloud_ttft, local_ttft)
            res[name].append(np.mean(realized > sla))
    plt.figure(figsize=(6.5, 4.5))
    for name in res:
        plt.plot(Rs, np.array(res[name]) * 100, "-o", color=COLORS[name], label=name)
    plt.xlabel("cloud speed_ratio R  (R<1 cloud faster, R>1 cloud slower)")
    plt.ylabel("SLA violation % @ 30% offload")
    plt.title("RQ0 robustness: heuristic vs ML across cloud-speed assumptions")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def oracle_gap_static(df, cfg, scores, local_ttft, cloud_ttft, cloud_cost, sla,
                      decision_overhead, router_cost, path):
    budgets = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    oracle_score = local_ttft - cloud_ttft
    plt.figure(figsize=(7.2, 5.0))
    for name in ["tail_aware", "ml_dual_risk", "semantic_embedding", "local_llm_probe", "thinking_probe"]:
        f = frontier(df, scores[name], local_ttft, cloud_ttft, cloud_cost, sla, budgets,
                     decision_overhead[name], router_cost[name])
        plt.plot(budgets * 100, f[:, 1] * 100, "-o", ms=4, color=COLORS[name],
                 label=SIGNAL_LABELS.get(name, name))
    oracle = frontier(df, oracle_score, local_ttft, cloud_ttft, cloud_cost, sla, budgets)
    plt.plot(budgets * 100, oracle[:, 1] * 100, "-o", ms=4, color="#111111",
             label="offline oracle")
    plt.xlabel("matched cloud-offload budget (%)")
    plt.ylabel("static adjusted SLA violation (%)")
    plt.title("Static oracle gap: deployable signals vs hindsight upper bound")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def utilization_heatmap_static(df, scores, local_ttft, cloud_ttft, sla, decision_overhead, path):
    values = []
    labels = ["low", "medium", "high", "burst"]
    arr = df["arr_rate_10s"].fillna(0).values.astype(float)
    burst = df["burst_ratio"].fillna(1).values.astype(float)
    q33, q66 = np.quantile(arr, [0.33, 0.66]) if len(arr) else (0.0, 0.0)
    masks = [
        arr <= q33,
        (arr > q33) & (arr <= q66),
        arr > q66,
        burst >= max(2.0, np.nanquantile(burst, 0.75)),
    ]
    budget = 0.30
    for mask in masks:
        row = []
        idx = np.where(mask)[0]
        if len(idx) == 0:
            values.append([np.nan] * len(SIGNAL_POLICY_ORDER))
            continue
        for name in SIGNAL_POLICY_ORDER:
            N = len(df)
            order = np.argsort(-scores[name])
            offload = np.zeros(N, dtype=bool)
            offload[order[: int(round(budget * N))]] = True
            realized = np.where(offload, cloud_ttft, local_ttft) + decision_overhead[name]
            row.append(np.mean(realized[idx] > sla) * 100.0)
        values.append(row)
    matrix = np.array(values, dtype=float)
    plt.figure(figsize=(8.2, 4.5))
    im = plt.imshow(matrix, aspect="auto", cmap="viridis_r")
    plt.colorbar(im, label="SLA violation % @ 30% offload")
    plt.xticks(range(len(SIGNAL_POLICY_ORDER)), [SIGNAL_LABELS[p] for p in SIGNAL_POLICY_ORDER],
               rotation=25, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center",
                         color="white" if matrix[i, j] > np.nanmedian(matrix) else "black",
                         fontsize=8)
    plt.title("Utilization-regime heatmap: signal value changes with queue pressure")
    plt.tight_layout()
    plt.savefig(path, dpi=130); plt.close()


def results_table(df, scores, local_ttft, cloud_ttft, cloud_cost,
                  decision_overhead, router_cost, sla, budget):
    N = len(df); k = int(round(budget * N))
    rows = ["# Signal-cost policy comparison @ matched ~30% offload budget", "",
            f"SLA = {sla:.0f} ms. Always-Local/Cloud shown at their fixed operating points.",
            "Adjusted metrics add decision/probe overhead and router cost. T2-T4 remain candidate/proxy policy tiers; a separate bounded live probe-interference audit now measures the T3 mechanism penalty, but T3/T4 are not full-scale live policy winners.", "",
            "| policy | tier | offload % | raw SLA viol % | adj SLA viol % | adj P99 TTFT ms | all-in cost $ | $/adj SLA-ok req |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
    pols = all_policies(load_config(), None)
    for name in POLICY_ORDER:
        s = scores[name]
        order = np.argsort(-s)
        kk = 0 if name == "always_local" else (N if name == "always_cloud" else k)
        offload = np.zeros(N, dtype=bool); offload[order[:kk]] = True
        raw = np.where(offload, cloud_ttft, local_ttft)
        adjusted = raw + decision_overhead[name]
        raw_viol = np.mean(raw > sla)
        adj_viol = np.mean(adjusted > sla)
        cost = cloud_cost[offload].sum() + router_cost[name].sum()
        ok = max(1, int(np.sum(adjusted <= sla)))
        tier = getattr(pols[name], "signal_tier", "")
        rows.append(f"| {name} | {tier} | {offload.mean()*100:.0f} | "
                    f"{raw_viol*100:.1f} | {adj_viol*100:.1f} | "
                    f"{np.percentile(adjusted,99):.0f} | {cost:.3f} | {cost/ok:.4f} |")
    return "\n".join(rows) + "\n"


def rq0_verdict(df, scores, local_ttft, cloud_ttft, sla, budget):
    N = len(df); k = int(round(budget * N))
    def viol_vec(name):
        order = np.argsort(-scores[name])
        off = np.zeros(N, dtype=bool); off[order[:k]] = True
        realized = np.where(off, cloud_ttft, local_ttft)
        return (realized > sla).astype(int)
    h, m = viol_vec("tail_aware"), viol_vec("ml_dual_risk")
    diff = h.mean() - m.mean()
    rng = np.random.default_rng(0)
    boots = [(_s := rng.integers(0, N, N), (h[_s].mean() - m[_s].mean()))[1] for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print("\n================ RQ0 VERDICT ================")
    print(f"SLA violation @ {budget*100:.0f}% offload:  tail_aware={h.mean()*100:.1f}%  "
          f"ml_dual_risk={m.mean()*100:.1f}%")
    print(f"Difference (heuristic - ML) = {diff*100:+.1f}%  95% CI [{lo*100:+.1f}%, {hi*100:+.1f}%]")
    if lo <= 0 <= hi:
        print("=> CI spans 0: NO significant difference -> ML does not beat the tuned heuristic")
        print("   (the proposal's H0/equivalence outcome - a valid, publishable systems finding).")
    elif diff > 0:
        print("=> Heuristic worse than ML (ML reduces violations): evidence FOR ML.")
    else:
        print("=> ML worse than heuristic at this budget.")
    return {
        "budget": budget,
        "tail_aware_sla_violation_rate": float(h.mean()),
        "ml_dual_risk_sla_violation_rate": float(m.mean()),
        "difference_heuristic_minus_ml": float(diff),
        "bootstrap_ci95_low": float(lo),
        "bootstrap_ci95_high": float(hi),
    }


def selective_label_demo(df, score, local_ttft, sla, budget):
    N = len(df); k = int(round(budget * N))
    order = np.argsort(-score)
    offload = np.zeros(N, dtype=bool); offload[order[:k]] = True
    true_rate = np.mean(local_ttft > sla)
    kept = ~offload
    naive_rate = np.mean(local_ttft[kept] > sla)
    print("\n========= SELECTIVE-LABEL DEMO (RQ6) =========")
    print(f"True local breach rate (shadow, all requests): {true_rate*100:.1f}%")
    print(f"Naive breach rate among KEPT-local requests  : {naive_rate*100:.1f}%")
    print(f"=> Naive routed logs understate local risk by {(true_rate-naive_rate)*100:.1f} pts")
    print("   because the policy skims the risky requests off to the cloud (selective labels).")
    return {
        "budget": budget,
        "true_local_breach_rate": float(true_rate),
        "naive_kept_local_breach_rate": float(naive_rate),
        "understatement_points": float(true_rate - naive_rate),
    }


if __name__ == "__main__":
    main()
