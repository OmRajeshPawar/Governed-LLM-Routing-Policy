"""Results figure for the SOC4AI paper (figures/results.pdf). CPU only, read-only on run data.

Panel (a): offline feasibility frontier, SLO violation vs cloud-call quota (frontier_eval.json).
Panel (b): nominal live runs (V4N), cumulative cloud calls vs request position, mean of 5 seeds,
           with the agreement's 90-call quota.
Panel (c): fault live runs (V4F), rolling SLO violation vs request position, mean of 5 seeds,
           injected outage and latency-spike windows shaded.
Palette: validated categorical slots 1-3 (+7) of the dataviz reference palette; every series also
carries a distinct line style and a direct label, so identity survives grayscale print.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RUNS = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
sys.path.insert(0, str(RUNS))
from soc4ai import analyze_live as al  # noqa: E402

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ARMS = [("v4_governor", "candidate", BLUE, "-"),
        ("guarded_governor", "incumbent", ORANGE, "--"),
        ("tail_aware", "ungoverned", AQUA, ":")]
N, WIN = 300, 24  # bursty cycle is 24 requests

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["CMU Serif", "Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.linewidth": 0.6,
    "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.labelcolor": INK,
    "pdf.fonttype": 42, "lines.linewidth": 1.4,
})


def per_position(block: str, policy: str, metric) -> np.ndarray:
    curves = []
    for path in sorted((RUNS / "soc4ai_out" / "runs" / block).glob(f"*_{policy}.jsonl")):
        df, _, _ = al.load_run(path)
        df = df.sort_values("trace_pos")
        curves.append(metric(df))
    assert len(curves) == 5, (block, policy, len(curves))
    return np.mean(curves, axis=0)


def cum_cloud(df) -> np.ndarray:
    v = np.zeros(N)
    v[df["trace_pos"].values] = (df["route"] == "cloud").astype(float).values
    return np.cumsum(v)


def rolling_violation(df) -> np.ndarray:
    v = np.zeros(N)
    v[df["trace_pos"].values] = df["violation"].astype(float).values
    kernel = np.ones(WIN) / WIN
    padded = np.pad(v, (WIN // 2, WIN - WIN // 2 - 1), mode="edge")
    return np.convolve(padded, kernel, mode="valid") * 100


def style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(length=2, width=0.5)


fig, axes = plt.subplots(1, 3, figsize=(4.8, 1.75), gridspec_kw={"wspace": 0.42})
report = {}

# (a) feasibility frontier
fr = json.loads((HERE / "frontier_eval.json").read_text(encoding="utf-8"))
ax = axes[0]
series = [("tail_aware", "heuristic", BLUE, "-"), ("ml_dual_risk", "classifier", VIOLET, "--"),
          ("queue_threshold", "queue thr.", ORANGE, "-."), ("prompt_static", "prompt-only", AQUA, ":")]
for key, label, color, ls in series:
    rows = fr["frontier"][key]
    x = [r["budget"] * 100 for r in rows]
    y = [r["violation_rate"] * 100 for r in rows]
    ax.plot(x, y, color=color, ls=ls, label=label)
ax.axhline(10, color=INK, lw=0.7, ls=(0, (1, 1.5)))
ax.axvline(30, color=MUTED, lw=0.6, ls="--"); ax.text(31, 14.5, "quota\n30%", color=MUTED, fontsize=5.6, va="top"); ax.text(46, 9.0, "SLO 10%", ha="left", va="top", color=INK, fontsize=6)
ax.set_xlim(10, 70); ax.set_ylim(0, 30)
ax.set_xlabel("cloud-call quota (%)"); ax.set_ylabel("SLO violation (%)")
ax.set_title("(a) replay: choosing the quota", loc="left")
ax.legend(frameon=False, fontsize=5.6, loc="upper right", handlelength=1.8, borderaxespad=0.1,
          labelspacing=0.3, bbox_to_anchor=(1.03, 1.0))
style(ax)

# (b) nominal: cumulative cloud calls
ax = axes[1]
x = np.arange(N)
ends = {}
for policy, label, color, ls in ARMS:
    y = per_position("V4N", policy, cum_cloud)
    ax.plot(x, y, color=color, ls=ls, label=label)
    ends[label] = y
    report[f"V4N_cum_cloud_end_{policy}"] = float(y[-1])
    if policy == "guarded_governor":
        report["V4N_v2_mean_cum_reaches_89_at"] = int(np.argmax(y >= 89)) if (y >= 89).any() else None
ax.axhline(90, color=INK, lw=0.7, ls=(0, (1, 1.5)))
ax.text(4, 96, "quota 90", color=INK, fontsize=6, va="bottom")
ax.legend(frameon=False, fontsize=5.6, loc="upper left", handlelength=1.8, labelspacing=0.3,
          borderaxespad=0.1, bbox_to_anchor=(0.0, 1.0), title="(b) and (c)", title_fontsize=5.6)
ax.set_xlim(0, N); ax.set_ylim(0, 260)
ax.set_xlabel("request position"); ax.set_ylabel("cumulative cloud calls")
ax.set_title("(b) live, nominal load", loc="left")
style(ax)

# (c) faults: rolling violation
ax = axes[2]
for lo, hi, name in ((96, 156, "outage"), (192, 252, "spike")):
    ax.axvspan(lo, hi, color="#efeeea", lw=0)
    ax.text((lo + hi) / 2, 103, name, ha="center", va="bottom", color=MUTED, fontsize=6)
for policy, label, color, ls in ARMS:
    y = per_position("V4F", policy, rolling_violation)
    ax.plot(x, y, color=color, ls=ls, label=label)
    report[f"V4F_peak_rolling_violation_{policy}"] = float(y.max())
ax.set_xlim(0, N); ax.set_ylim(0, 102)
ax.set_xlabel("request position"); ax.set_ylabel("SLO violation (%)")
ax.set_title("(c) live, injected faults", loc="left", pad=9)
style(ax)

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "results.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(OUT / "results_preview.png", dpi=220, bbox_inches="tight", pad_inches=0.01)
(OUT / "make_figures.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
print(json.dumps(report, indent=1))
