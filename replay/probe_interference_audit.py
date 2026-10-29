from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path("analysis") / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import _repo_root, ensure_dirs, load_config
from apex.events import EventLog, write_manifest
from apex.core import Request
from apex.gateway import Gateway
from apex.policies import AlwaysLocal
from apex.providers import CalibratedCloudProvider, OllamaProvider
from apex.core import GatewayState
from apex.workload import _make_prompt

PROBE_PROMPT = (
    "You are a tiny local routing probe. Classify this incoming request as "
    "LOCAL or CLOUD using only likely latency risk. Answer one word: LOCAL."
)


def apply_frozen_thresholds(cfg) -> None:
    meta_path = _repo_root() / cfg.paths.models_dir / "meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    frozen = meta.get("frozen_thresholds", {})
    if "sla_ttft_ms" in frozen:
        cfg.targets.sla_ttft_ms = float(frozen["sla_ttft_ms"])
    if "occupancy_service_ms" in frozen:
        cfg.targets.occupancy_service_ms = float(frozen["occupancy_service_ms"])


def build_trace(seed: int, n: int, spacing_s: float, max_tokens: int) -> list[dict]:
    rng = random.Random(seed)
    categories = ["qa", "summarize", "code", "extract_json", "longform", "qa"]
    sizes = ["short", "medium", "short", "medium", "medium", "short"]
    trace: list[dict] = []
    for i in range(n):
        category = categories[i % len(categories)]
        size = sizes[i % len(sizes)]
        trace.append(
            {
                "request_id": f"req-{seed}-{i}",
                "trace_pos": i,
                "task_category": category,
                "size": size,
                "max_tokens": max_tokens,
                "arrival_offset": i * spacing_s,
                "prompt": _make_prompt(rng, category, size),
            }
        )
    return trace


def make_local_provider(cfg) -> OllamaProvider:
    local_cfg = cfg.providers.local
    return OllamaProvider(
        "local",
        local_cfg.model,
        local_cfg.keep_alive,
        local_cfg.num_gpu,
        cfg.generation.temperature,
        local_cfg.cost_per_1k_tokens,
        local_cfg.extra_latency_ms,
        num_ctx=local_cfg.num_ctx,
    )


def make_dummy_cloud(cfg) -> CalibratedCloudProvider:
    cloud_cfg = cfg.providers.cloud
    return CalibratedCloudProvider(
        cloud_cfg,
        cloud_cfg.base_ttft_ms or 200.0,
        cloud_cfg.per_token_ms or 5.0,
        cfg.seed,
    )


async def smoke_local(local: OllamaProvider) -> None:
    await local.warmup()
    res = await local.generate("Return the word ok.", 1)
    if not res.ok:
        raise SystemExit(f"Local Ollama provider failed during smoke test: {res.error}")


async def run_condition(
    cfg,
    local: OllamaProvider,
    cloud: CalibratedCloudProvider,
    trace: list[dict],
    *,
    seed: int,
    condition: str,
    probe_tokens: int,
) -> list[dict]:
    state = GatewayState(cfg)
    root = _repo_root()
    out = root / cfg.paths.events_dir / f"probe_interference_{condition}_s{seed}.jsonl"
    log = EventLog(out)
    await log.start()
    gw = Gateway(cfg, local, cloud, state, log, run_id=f"probe_interference_{condition}_s{seed}", trace_id="probe_interference")
    policy = AlwaysLocal()
    t0 = time.perf_counter()
    rows: list[dict] = []

    async def _run_one(spec: dict) -> dict:
        target = t0 + float(spec["arrival_offset"])
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)

        received = time.perf_counter()
        req = Request(
            spec["request_id"],
            spec["prompt"],
            int(spec["max_tokens"]),
            spec["task_category"],
            received,
            trace_pos=int(spec["trace_pos"]),
        )

        probe_wall_ms = 0.0
        probe_ttft_ms = None
        probe_service_ms = None
        probe_ok = None
        if condition == "co_located_probe":
            p0 = time.perf_counter()
            probe = await local.generate(PROBE_PROMPT, probe_tokens)
            probe_wall_ms = (time.perf_counter() - p0) * 1000.0
            probe_ttft_ms = probe.provider_ttft_ms
            probe_service_ms = probe.service_ms
            probe_ok = probe.ok

        result = await gw.handle(req, policy, threshold=0.0, route_override="local")
        outcome = result["outcome"]
        return {
            "seed": seed,
            "condition": condition,
            "request_id": spec["request_id"],
            "trace_pos": int(spec["trace_pos"]),
            "arrival_offset_s": float(spec["arrival_offset"]),
            "task_category": spec["task_category"],
            "max_tokens": int(spec["max_tokens"]),
            "probe_tokens": probe_tokens if condition == "co_located_probe" else 0,
            "probe_wall_ms": probe_wall_ms,
            "probe_ttft_ms": probe_ttft_ms,
            "probe_service_ms": probe_service_ms,
            "probe_ok": probe_ok,
            "queue_wait_ms": outcome["queue_wait_ms"],
            "provider_ttft_ms": outcome["provider_ttft_ms"],
            "end_to_end_ttft_ms": outcome["end_to_end_ttft_ms"],
            "service_ms": outcome["service_ms"],
            "end_to_end_ms": outcome["end_to_end_ms"],
            "ok": outcome["ok"],
            "sla_breach": bool(outcome["end_to_end_ttft_ms"] and outcome["end_to_end_ttft_ms"] > cfg.targets.sla_ttft_ms),
        }

    try:
        rows = await asyncio.gather(*[asyncio.create_task(_run_one(spec)) for spec in trace])
    finally:
        await log.aclose()
    return sorted(rows, key=lambda r: r["trace_pos"])


def summarize_rows(rows: list[dict], sla_ms: float) -> dict:
    df = pd.DataFrame(rows)
    ttft = pd.to_numeric(df["end_to_end_ttft_ms"], errors="coerce").dropna()
    qwait = pd.to_numeric(df["queue_wait_ms"], errors="coerce").dropna()
    probe_wall = pd.to_numeric(df["probe_wall_ms"], errors="coerce")
    condition = str(df["condition"].iloc[0])
    seed = int(df["seed"].iloc[0])
    return {
        "seed": seed,
        "condition": condition,
        "n_user_requests": int(len(df)),
        "n_ok": int(df["ok"].sum()),
        "n_probes": int((probe_wall > 0).sum()),
        "sla_ms": float(sla_ms),
        "sla_violation_rate": float(np.mean(ttft.to_numpy() > sla_ms)) if len(ttft) else np.nan,
        "mean_ttft_ms": float(ttft.mean()) if len(ttft) else np.nan,
        "p50_ttft_ms": float(np.percentile(ttft, 50)) if len(ttft) else np.nan,
        "p95_ttft_ms": float(np.percentile(ttft, 95)) if len(ttft) else np.nan,
        "p99_ttft_ms": float(np.percentile(ttft, 99)) if len(ttft) else np.nan,
        "mean_queue_wait_ms": float(qwait.mean()) if len(qwait) else np.nan,
        "p95_queue_wait_ms": float(np.percentile(qwait, 95)) if len(qwait) else np.nan,
        "sum_queue_wait_ms": float(qwait.sum()) if len(qwait) else np.nan,
        "mean_probe_wall_ms": float(probe_wall[probe_wall > 0].mean()) if (probe_wall > 0).any() else 0.0,
        "p95_probe_wall_ms": float(np.percentile(probe_wall[probe_wall > 0], 95)) if (probe_wall > 0).any() else 0.0,
    }


def paired_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sla_violation_rate",
        "mean_ttft_ms",
        "p95_ttft_ms",
        "p99_ttft_ms",
        "mean_queue_wait_ms",
        "p95_queue_wait_ms",
        "sum_queue_wait_ms",
    ]
    rows = []
    for seed, group in summary.groupby("seed", sort=True):
        pivot = group.set_index("condition")
        if not {"baseline", "co_located_probe"}.issubset(pivot.index):
            continue
        row = {"seed": seed}
        for metric in metrics:
            row[f"{metric}_baseline"] = float(pivot.loc["baseline", metric])
            row[f"{metric}_probe"] = float(pivot.loc["co_located_probe", metric])
            row[f"{metric}_delta"] = float(pivot.loc["co_located_probe", metric] - pivot.loc["baseline", metric])
        row["mean_probe_wall_ms"] = float(pivot.loc["co_located_probe", "mean_probe_wall_ms"])
        row["p95_probe_wall_ms"] = float(pivot.loc["co_located_probe", "p95_probe_wall_ms"])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_results(rows: pd.DataFrame, summary: pd.DataFrame, deltas: pd.DataFrame, path: Path) -> None:
    pretty = {"baseline": "Baseline\n(no probe)", "co_located_probe": "Co-located\nlocal probe"}
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2))

    ax = axes[0]
    data = [
        rows.loc[rows["condition"] == "baseline", "end_to_end_ttft_ms"].dropna().to_numpy(),
        rows.loc[rows["condition"] == "co_located_probe", "end_to_end_ttft_ms"].dropna().to_numpy(),
    ]
    try:
        ax.boxplot(data, tick_labels=[pretty["baseline"], pretty["co_located_probe"]], showfliers=False)
    except TypeError:
        ax.boxplot(data, labels=[pretty["baseline"], pretty["co_located_probe"]], showfliers=False)
    ax.set_ylabel("user end-to-end TTFT (ms)")
    ax.set_title("User latency distribution")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    grouped = summary.groupby("condition", sort=False)[["mean_queue_wait_ms", "p95_ttft_ms"]].mean()
    x = np.arange(2)
    width = 0.36
    ax.bar(x - width / 2, [grouped.loc["baseline", "mean_queue_wait_ms"], grouped.loc["co_located_probe", "mean_queue_wait_ms"]], width, label="mean queue wait")
    ax.bar(x + width / 2, [grouped.loc["baseline", "p95_ttft_ms"], grouped.loc["co_located_probe", "p95_ttft_ms"]], width, label="P95 TTFT")
    ax.set_xticks(x, [pretty["baseline"], pretty["co_located_probe"]])
    ax.set_ylabel("ms")
    ax.set_title("Queue and tail impact")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[2]
    metric_labels = ["mean TTFT", "P95 TTFT", "mean queue", "SLA viol."]
    vals = [
        deltas["mean_ttft_ms_delta"].mean(),
        deltas["p95_ttft_ms_delta"].mean(),
        deltas["mean_queue_wait_ms_delta"].mean(),
        deltas["sla_violation_rate_delta"].mean() * 100.0,
    ]
    colors = ["#355c7d", "#c06c84", "#6c5b7b", "#f67280"]
    ax.bar(metric_labels, vals, color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Paired probe penalty")
    ax.set_ylabel("delta (ms; SLA in points)")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Live co-located local-probe interference audit", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_summary_md(summary: pd.DataFrame, deltas: pd.DataFrame, args, elapsed_s: float) -> None:
    out = _repo_root() / "analysis" / "probe_interference_summary.md"
    aggregate = summary.groupby("condition", as_index=False).agg(
        n_user_requests=("n_user_requests", "sum"),
        n_probes=("n_probes", "sum"),
        sla_violation_rate=("sla_violation_rate", "mean"),
        mean_ttft_ms=("mean_ttft_ms", "mean"),
        p95_ttft_ms=("p95_ttft_ms", "mean"),
        p99_ttft_ms=("p99_ttft_ms", "mean"),
        mean_queue_wait_ms=("mean_queue_wait_ms", "mean"),
        p95_queue_wait_ms=("p95_queue_wait_ms", "mean"),
        mean_probe_wall_ms=("mean_probe_wall_ms", "mean"),
        p95_probe_wall_ms=("p95_probe_wall_ms", "mean"),
    )

    d = deltas.mean(numeric_only=True)
    lines = [
        "# Live Co-Located Probe Interference Audit",
        "",
        "This is a bounded live A/B audit, not the full 4,500-request study. It replays identical local-only traces twice: once with no routing probe and once where each request first runs a short local LLM probe on the same Ollama model.",
        "",
        f"- Seeds: `{', '.join(str(s) for s in args.seeds)}`",
        f"- Requests per condition per seed: `{args.n}`",
        f"- User max tokens: `{args.max_tokens}`",
        f"- Probe tokens: `{args.probe_tokens}`",
        f"- Arrival spacing: `{args.spacing_s:.2f}s`",
        f"- Runtime: `{elapsed_s:.1f}s`",
        "",
        "## Aggregate Results",
        "",
        "| condition | user requests | probes | SLA viol % | mean TTFT ms | P95 TTFT ms | P99 TTFT ms | mean queue wait ms | P95 queue wait ms | mean probe wall ms | P95 probe wall ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in aggregate.iterrows():
        lines.append(
            f"| {r.condition} | {int(r.n_user_requests)} | {int(r.n_probes)} | "
            f"{r.sla_violation_rate*100:.1f} | {r.mean_ttft_ms:.0f} | {r.p95_ttft_ms:.0f} | "
            f"{r.p99_ttft_ms:.0f} | {r.mean_queue_wait_ms:.0f} | {r.p95_queue_wait_ms:.0f} | "
            f"{r.mean_probe_wall_ms:.0f} | {r.p95_probe_wall_ms:.0f} |"
        )
    lines.extend(
        [
            "",
            "## Mean Paired Delta: Co-Located Probe Minus Baseline",
            "",
            "| metric | delta |",
            "|---|---:|",
            f"| SLA violation | {d['sla_violation_rate_delta']*100:.1f} points |",
            f"| mean TTFT | {d['mean_ttft_ms_delta']:.0f} ms |",
            f"| P95 TTFT | {d['p95_ttft_ms_delta']:.0f} ms |",
            f"| P99 TTFT | {d['p99_ttft_ms_delta']:.0f} ms |",
            f"| mean queue wait | {d['mean_queue_wait_ms_delta']:.0f} ms |",
            f"| P95 queue wait | {d['p95_queue_wait_ms_delta']:.0f} ms |",
            f"| cumulative queue wait per trace | {d['sum_queue_wait_ms_delta']:.0f} ms |",
            f"| mean probe wall latency | {d['mean_probe_wall_ms']:.0f} ms |",
            f"| P95 probe wall latency | {d['p95_probe_wall_ms']:.0f} ms |",
            "",
            "## Interpretation",
            "",
            "A positive delta means the co-located local probe made user-facing local service worse on the same trace. It measures a mechanism behind the signal-cost frontier: a probe can add direct decision latency and contend with the constrained local LLM resource.",
            "",
            "## Scope",
            "",
            "A controlled live A/B on the testbed GPU that isolates the probe-contention mechanism behind the signal-cost frontier. The policy comparison itself comes from the 4,500-request replay corpus.",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


async def main_async(args) -> None:
    start = time.perf_counter()
    cfg = load_config()
    apply_frozen_thresholds(cfg)
    ensure_dirs(cfg)
    local = make_local_provider(cfg)
    cloud = make_dummy_cloud(cfg)
    await smoke_local(local)

    all_rows: list[dict] = []
    all_summaries: list[dict] = []
    for i, seed in enumerate(args.seeds):
        if (time.perf_counter() - start) > args.time_budget_s:
            print(f"[stop] time budget reached before seed {seed}")
            break
        trace = build_trace(seed, args.n, args.spacing_s, args.max_tokens)
        order = ["baseline", "co_located_probe"] if i % 2 == 0 else ["co_located_probe", "baseline"]
        for condition in order:
            print(f"[run] seed={seed} condition={condition}")
            rows = await run_condition(cfg, local, cloud, trace, seed=seed, condition=condition, probe_tokens=args.probe_tokens)
            all_rows.extend(rows)
            all_summaries.append(summarize_rows(rows, cfg.targets.sla_ttft_ms))
            await asyncio.sleep(args.cooldown_s)

    if not all_rows:
        raise SystemExit("No probe-interference rows were collected.")

    root = _repo_root()
    out_dir = root / "analysis"
    fig_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    rows_df = pd.DataFrame(all_rows)
    summary_df = pd.DataFrame(all_summaries)
    delta_df = paired_deltas(summary_df)
    if delta_df.empty:
        raise SystemExit("Missing paired baseline/probe conditions; cannot compute deltas.")

    rows_df.to_csv(out_dir / "probe_interference_rows.csv", index=False)
    summary_df.to_csv(out_dir / "probe_interference_summary.csv", index=False)
    delta_df.to_csv(out_dir / "probe_interference_deltas.csv", index=False)
    plot_results(rows_df, summary_df, delta_df, fig_dir / "probe_interference_ab.png")
    write_summary_md(summary_df, delta_df, args, time.perf_counter() - start)
    write_manifest(
        root / "data" / "manifests" / "probe_interference_manifest.json",
        cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict(),
        {
            "audit": "probe_interference",
            "seeds": args.seeds,
            "n": args.n,
            "spacing_s": args.spacing_s,
            "max_tokens": args.max_tokens,
            "probe_tokens": args.probe_tokens,
            "elapsed_s": round(time.perf_counter() - start, 2),
            "outputs": [
                "analysis/probe_interference_rows.csv",
                "analysis/probe_interference_summary.csv",
                "analysis/probe_interference_deltas.csv",
                "analysis/probe_interference_summary.md",
                "analysis/figures/probe_interference_ab.png",
            ],
        },
    )

    d = delta_df.mean(numeric_only=True)
    print("Probe-interference audit complete.")
    print(f"Mean paired delta: SLA {d['sla_violation_rate_delta']*100:.1f} points, "
          f"mean TTFT {d['mean_ttft_ms_delta']:.0f} ms, P95 TTFT {d['p95_ttft_ms_delta']:.0f} ms, "
          f"mean queue wait {d['mean_queue_wait_ms_delta']:.0f} ms.")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[7, 13, 42])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--spacing-s", type=float, default=0.25)
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--probe-tokens", type=int, default=8)
    ap.add_argument("--cooldown-s", type=float, default=0.75)
    ap.add_argument("--time-budget-s", type=float, default=3300.0)
    return ap.parse_args()


def main() -> None:
    asyncio.run(main_async(parse_args()))


if __name__ == "__main__":
    main()
