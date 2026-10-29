from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import Config, _repo_root, load_config
from apex.events import load_events, resolve_event_files
from apex.core import FEATURE_NAMES
from apex.policies import all_policies
from apex.workload import make_trace
from analysis.evaluate import (
    COLORS,
    POLICY_ORDER,
    SIGNAL_LABELS,
    SIGNAL_POLICY_ORDER,
    build_cloud_outcomes,
    meta_calib,
    policy_overhead_vectors,
    policy_scores,
)
from ml.risk_model import CalibratedRiskModel

RUN_ID_RE = re.compile(
    r"^shadow_(?:(?P<variant>.+?)_)?(?P<mix>synthetic|real_sample|reasoning_coding)_"
    r"(?P<pattern>steady|bursty|saturation)_n(?P<n>\d+)_s(?P<seed>\d+)$"
)
LEGACY_RUN_ID_RE = re.compile(
    r"^shadow_(?P<pattern>steady|bursty|saturation)_n(?P<n>\d+)_s(?P<seed>\d+)$"
)


@dataclass(frozen=True)
class TraceMeta:
    mix: str
    pattern: str
    n: int
    seed: int


def parse_trace_meta(run_id: str) -> TraceMeta:
    match = RUN_ID_RE.match(run_id)
    if not match:
        match = LEGACY_RUN_ID_RE.match(run_id)
    if not match:
        raise ValueError(f"Unsupported shadow run_id format for replay: {run_id}")
    groups = match.groupdict()
    return TraceMeta(
        mix=groups.get("mix") or "synthetic",
        pattern=groups["pattern"],
        n=int(groups["n"]),
        seed=int(groups["seed"]),
    )


def load_shadow_df(cfg: Config) -> pd.DataFrame:
    events_dir = _repo_root() / cfg.paths.events_dir
    files, shadow_source = resolve_event_files(events_dir)
    if not files:
        raise SystemExit(f"No shadow files matched {shadow_source!r}. Run shadow collection first.")
    print(f"Queue replay using {len(files)} shadow file(s), source={shadow_source}.")
    return pd.concat([load_events(f) for f in files], ignore_index=True)


def _offload_mask(scores: np.ndarray, policy_name: str, budget: float) -> np.ndarray:
    n = len(scores)
    if policy_name == "always_local":
        return np.zeros(n, dtype=bool)
    if policy_name == "always_cloud":
        return np.ones(n, dtype=bool)
    k = int(round(budget * n))
    order = np.argsort(-scores)
    out = np.zeros(n, dtype=bool)
    out[order[:k]] = True
    return out


def replay_one_trace(
    df: pd.DataFrame,
    cfg: Config,
    policy_name: str,
    score: np.ndarray,
    cloud_ttft: np.ndarray,
    cloud_cost: np.ndarray,
    decision_overhead: np.ndarray,
    router_cost: np.ndarray,
    budget: float,
    sla: float,
) -> dict:
    meta = parse_trace_meta(str(df["run_id"].iloc[0]))
    trace = make_trace(n=meta.n, pattern=meta.pattern, seed=meta.seed, mix=meta.mix)
    arrivals = np.array([row["arrival_offset"] for row in trace], dtype=float)

    order = np.argsort(df["trace_pos"].values.astype(int))
    df = df.iloc[order].reset_index(drop=True)
    score = np.asarray(score)[order]
    cloud_ttft = np.asarray(cloud_ttft)[order]
    cloud_cost = np.asarray(cloud_cost)[order]
    decision_overhead = np.asarray(decision_overhead)[order]
    router_cost = np.asarray(router_cost)[order]

    provider_ttft = df["provider_ttft_ms"].fillna(df["end_to_end_ttft_ms"]).values.astype(float)
    service_ms = df["service_ms"].fillna(df["provider_ttft_ms"]).fillna(0.0).values.astype(float)
    service_s = service_ms / 1000.0

    offload = _offload_mask(score, policy_name, budget)
    servers = np.zeros(int(cfg.local_concurrency), dtype=float)
    realized_ttft_ms: list[float] = []
    queue_wait_ms: list[float] = []
    local_count = 0
    for idx, arrival in enumerate(arrivals):
        if offload[idx]:
            realized_ttft_ms.append(float(cloud_ttft[idx] + decision_overhead[idx]))
            queue_wait_ms.append(0.0)
            continue
        server_idx = int(np.argmin(servers))
        start = max(arrival, float(servers[server_idx]))
        wait_ms = max(0.0, (start - arrival) * 1000.0)
        servers[server_idx] = start + max(0.0, service_s[idx])
        realized_ttft_ms.append(float(wait_ms + provider_ttft[idx] + decision_overhead[idx]))
        queue_wait_ms.append(wait_ms)
        local_count += 1

    realized = np.asarray(realized_ttft_ms)
    qwait = np.asarray(queue_wait_ms)
    return {
        "run_id": str(df["run_id"].iloc[0]),
        "mix": meta.mix,
        "pattern": meta.pattern,
        "seed": meta.seed,
        "policy": policy_name,
        "budget": budget,
        "offload_rate": float(offload.mean()),
        "local_requests": local_count,
        "sla_violation_rate": float(np.mean(realized > sla)),
        "p50_ttft_ms": float(np.percentile(realized, 50)),
        "p95_ttft_ms": float(np.percentile(realized, 95)),
        "p99_ttft_ms": float(np.percentile(realized, 99)),
        "mean_queue_wait_ms": float(np.mean(qwait)),
        "sum_queue_wait_ms": float(np.sum(qwait)),
        "all_in_cost_usd": float(np.sum(cloud_cost[offload]) + np.sum(router_cost)),
    }


def replay_all() -> pd.DataFrame:
    cfg = load_config()
    df = load_shadow_df(cfg)
    meta_path = _repo_root() / cfg.paths.models_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    fr = meta["frozen_thresholds"]
    cfg.targets.sla_ttft_ms = fr["sla_ttft_ms"]
    cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]
    sla = fr["sla_ttft_ms"]
    heuristic_weights = meta.get("heuristic_weights")
    ml = CalibratedRiskModel.load(_repo_root() / cfg.paths.models_dir)
    base, per_tok = meta_calib(cfg, meta)

    scores = policy_scores(df, cfg, ml, heuristic_weights)
    cloud_ttft, cloud_cost = build_cloud_outcomes(df, cfg, base, per_tok, 1.0)
    decision_overhead, router_cost = policy_overhead_vectors(df, cfg, heuristic_weights)
    scores["queue_service_oracle"] = df["service_ms"].fillna(0).values.astype(float)
    decision_overhead["queue_service_oracle"] = np.zeros(len(df))
    router_cost["queue_service_oracle"] = np.zeros(len(df))

    budgets = [0.10, 0.20, 0.30, 0.40, 0.50]
    rows = []
    for run_id, group in df.groupby("run_id", sort=True):
        idx = group.index.to_numpy()
        for policy_name in [*POLICY_ORDER, "queue_service_oracle"]:
            for budget in budgets:
                rows.append(
                    replay_one_trace(
                        group,
                        cfg,
                        policy_name,
                        scores[policy_name][idx],
                        cloud_ttft[idx],
                        cloud_cost[idx],
                        decision_overhead[policy_name][idx],
                        router_cost[policy_name][idx],
                        budget,
                        sla,
                    )
                )
    return pd.DataFrame(rows)


def write_outputs(replay_df: pd.DataFrame) -> None:
    root = _repo_root()
    out_dir = root / "analysis"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    replay_df.to_csv(out_dir / "queue_replay_results.csv", index=False)

    summary = (
        replay_df.groupby(["policy", "budget"], as_index=False)
        .agg(
            sla_violation_rate=("sla_violation_rate", "mean"),
            p95_ttft_ms=("p95_ttft_ms", "mean"),
            p99_ttft_ms=("p99_ttft_ms", "mean"),
            mean_queue_wait_ms=("mean_queue_wait_ms", "mean"),
            all_in_cost_usd=("all_in_cost_usd", "mean"),
        )
        .sort_values(["budget", "policy"])
    )
    lines = [
        "# Queue replay policy comparison",
        "",
        "Static queue replay on identical arrival traces. Metrics are averaged across shadow runs.",
        "",
        "| policy | budget | SLA viol % | P95 TTFT ms | P99 TTFT ms | mean queue wait ms | all-in cost $ |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in summary.iterrows():
        lines.append(
            f"| {r.policy} | {r.budget:.0%} | {r.sla_violation_rate*100:.1f} | "
            f"{r.p95_ttft_ms:.0f} | {r.p99_ttft_ms:.0f} | {r.mean_queue_wait_ms:.0f} | "
            f"{r.all_in_cost_usd:.3f} |"
        )
    (out_dir / "queue_replay_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    plot_oracle_gap(replay_df, fig_dir / "oracle_gap.png")
    plot_mix_heatmap(replay_df, fig_dir / "utilization_regime_heatmap.png")


def plot_oracle_gap(df: pd.DataFrame, path: Path) -> None:
    pivot = (
        df.groupby(["policy", "budget"])["sla_violation_rate"]
        .mean()
        .unstack(0)
        .sort_index()
    )
    budgets = pivot.index.to_numpy()
    plt.figure(figsize=(7.2, 5.0))
    for policy in ["tail_aware", "ml_dual_risk", "semantic_embedding", "local_llm_probe", "thinking_probe", "queue_service_oracle"]:
        if policy not in pivot:
            continue
        label = "queue service oracle" if policy == "queue_service_oracle" else SIGNAL_LABELS.get(policy, policy)
        color = "#111111" if policy == "queue_service_oracle" else COLORS.get(policy, "#777777")
        plt.plot(budgets * 100, pivot[policy] * 100, "-o", label=label, color=color, ms=4)
    plt.xlabel("matched cloud-offload budget (%)")
    plt.ylabel("queue-replay SLA violation (%)")
    plt.title("Oracle gap: how much headroom exists beyond deployable policies?")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()

    if {"tail_aware", "queue_service_oracle"}.issubset(pivot.columns):
        headroom_rows = []
        denom = pivot["tail_aware"] - pivot["queue_service_oracle"]
        for policy in ["ml_dual_risk", "semantic_embedding", "local_llm_probe", "thinking_probe"]:
            if policy not in pivot:
                continue
            captured = (pivot["tail_aware"] - pivot[policy]) / denom.replace(0, np.nan)
            for budget, value in captured.items():
                headroom_rows.append({"policy": policy, "budget": budget, "headroom_captured": value})
        pd.DataFrame(headroom_rows).to_csv(path.parent.parent / "oracle_headroom_captured.csv", index=False)


def plot_mix_heatmap(df: pd.DataFrame, path: Path) -> None:
    focus = df[(df["budget"].round(2) == 0.30) & (df["policy"].isin(SIGNAL_POLICY_ORDER))]
    if focus.empty:
        return
    table = focus.pivot_table(
        index="mix",
        columns="policy",
        values="sla_violation_rate",
        aggfunc="mean",
    )
    ordered_cols = [p for p in SIGNAL_POLICY_ORDER if p in table.columns]
    table = table[ordered_cols]
    plt.figure(figsize=(8.0, 3.8))
    values = table.values * 100
    im = plt.imshow(values, cmap="viridis_r", aspect="auto")
    plt.colorbar(im, label="SLA violation % @ 30% offload")
    plt.xticks(range(len(table.columns)), [SIGNAL_LABELS.get(c, c) for c in table.columns], rotation=25, ha="right")
    plt.yticks(range(len(table.index)), table.index)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            plt.text(j, i, f"{values[i, j]:.1f}", ha="center", va="center", color="white" if values[i, j] > np.nanmedian(values) else "black", fontsize=8)
    plt.title("Workload-regime heatmap: richer signals are only useful if they move this table")
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def main() -> None:
    replay_df = replay_all()
    write_outputs(replay_df)
    print("Queue replay complete.")
    print(replay_df.groupby(["policy", "budget"])["sla_violation_rate"].mean().unstack(0).round(3))


if __name__ == "__main__":
    main()
