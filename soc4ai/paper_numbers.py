"""Turn live-study analysis JSON into LaTeX table rows and key sentences.

Every number placed in the paper's live-validation section comes from this
script's output, so each value traces to soc4ai_out/analysis/live_<exp>.json.
Usage: python soc4ai/paper_numbers.py --exp A|B|C  (prints LaTeX to stdout and
writes soc4ai_out/analysis/paper_<exp>.tex)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "soc4ai_out" / "analysis"
LABEL = {"guarded_governor": "Governed", "tail_aware": "Ungoverned"}


def pct(x):
    return "--" if x is None else f"{100 * x:.1f}"


def ms(x):
    return "--" if x is None else f"{x:,.0f}"


def usd(x):
    return "--" if x is None else f"{x:.3f}"


def mean_telemetry(result: dict, policy: str, key: str):
    runs = [r for r in result["telemetry"] if policy in r]
    vals = [result["telemetry"][r].get(key) for r in runs if result["telemetry"][r].get("available")]
    vals = [v for v in vals if v is not None]
    return statistics.mean(vals) if vals else None


def exp_a(result: dict) -> str:
    a = result["A"]
    lines = ["% Experiment A: nominal, 8 seeds x 2 routers",
             r"Router & SLO viol.\ (\%) & P95 TTFT (ms) & Cloud (\%) & Failed & Spend (USD) & GPU Wh/req \\",
             r"\midrule"]
    for pol in ("guarded_governor", "tail_aware"):
        s = a["per_policy"][pol]
        lines.append(f"{LABEL[pol]} & {pct(s['violation_rate'])} & {ms(s['p95_ttft_ms'])} & "
                     f"{pct(s['cloud_share'])} & {s['failed']} & {usd(s['spend_usd'])} & "
                     f"{mean_telemetry(result, pol, 'wh_per_request') or 0:.3f} \\\\")
    c = a["confirmatory_statistics"]
    if c.get("status") == "complete":
        p, k = c["primary"], c["key_secondary"]
        lines.append(f"% confirmatory: branch={c['result_branch']}; governor-minus-ungoverned "
                     f"{p['effect_pp']:+.2f} pp CI [{p['ci95'][0]:+.2f}, {p['ci95'][1]:+.2f}] pass={p['pass']}; "
                     f"P95 ratio {k['p95_ratio']:.3f} CI [{k['ci95'][0]:.3f}, {k['ci95'][1]:.3f}] pass={k['pass']}; "
                     f"catastrophic_seed={c['catastrophic_seed_regression']}")
    else:
        lines.append(f"% confirmatory status: {c.get('status')} {c.get('reason')}")
    return "\n".join(lines)


def exp_b(result: dict) -> str:
    b = result["B"]
    lines = ["% Experiment B: chaos episodes, 3 seeds x 2 routers",
             r"Router & Episode & SLO viol.\ (\%) & Failed & Cloud failed & Spend (USD) & Wasted (USD) \\",
             r"\midrule"]
    for pol in ("guarded_governor", "tail_aware"):
        for seg in ("outage", "spike", "all"):
            s = b["segments"][pol][seg]
            lines.append(f"{LABEL[pol]} & {seg} & {pct(s['violation_rate'])} & {s['failed']} & "
                         f"{s['cloud_failed']} & {usd(s['spend_usd'])} & {usd(s['wasted_spend_usd'])} \\\\")
    det = [v.get("outage_detect_s") for v in b["breaker"].values() if v.get("outage_detect_s") is not None]
    rec = [v.get("outage_recover_s") for v in b["breaker"].values() if v.get("outage_recover_s") is not None]
    sdet = [v.get("spike_detect_s") for v in b["breaker"].values() if v.get("spike_detect_s") is not None]
    lines.append(f"% breaker outage detect s: {det}; recover s: {rec}; spike detect s: {sdet}")
    return "\n".join(lines)


def exp_c(result: dict) -> str:
    c = result["C"]
    lines = ["% Experiment C: drift phases, 3 seeds x 2 routers",
             r"Phase & Gov.\ viol.\ (\%) & Gov.\ cloud (\%) & Ungov.\ viol.\ (\%) & Ungov.\ cloud (\%) \\",
             r"\midrule"]
    phases = ["low_factual", "short_burst", "reasoning_heavy", "recovery", "cloud_degradation"]
    for ph in phases:
        g = c.get("guarded_governor", {}).get(ph)
        u = c.get("tail_aware", {}).get(ph)
        if g and u:
            lines.append(f"{ph.replace('_', ' ')} & {pct(g['violation_rate'])} & {pct(g['cloud_share'])} & "
                         f"{pct(u['violation_rate'])} & {pct(u['cloud_share'])} \\\\")
    return "\n".join(lines)


def common(result: dict) -> str:
    g = result["gemini_latency"]
    bd = result.get("breach_decomposition", {})
    out = [f"% live Gemini TTFT n={g['n_live']} p50={ms(g['p50_ms'])} p95={ms(g['p95_ms'])} "
           f"p99={ms(g['p99_ms'])} slo_miss={pct(g['slo_miss_share'])}% | calibration p50={ms(g.get('calibration_p50_ms'))} "
           f"p95={ms(g.get('calibration_p95_ms'))} | July n={g.get('july_n')} p50={ms(g.get('july_p50_ms'))} p95={ms(g.get('july_p95_ms'))}"]
    if "breached" in bd:
        b = bd["breached"]
        out.append(f"% local breaches n={b['n']}: mean TTFT {ms(b['mean_ttft_ms'])} ms, queue wait "
                   f"{ms(b['mean_queue_wait_ms'])} ms ({pct(b['queue_share_of_ttft'])}% of TTFT), prefill {ms(b['mean_prefill_ms'])} ms")
    if "met" in bd:
        m = bd["met"]
        out.append(f"% local met n={m['n']}: mean TTFT {ms(m['mean_ttft_ms'])} ms, queue {pct(m['queue_share_of_ttft'])}% of TTFT")
    for pol in ("guarded_governor", "tail_aware"):
        out.append(f"% {pol}: mean GPU Wh/req {mean_telemetry(result, pol, 'wh_per_request')}, "
                   f"Wh per SLO-met local req {mean_telemetry(result, pol, 'wh_per_slo_met_local_request')}, "
                   f"throttled share {mean_telemetry(result, pol, 'throttled_sample_share')}, "
                   f"max temp {mean_telemetry(result, pol, 'max_temp_c')}, "
                   f"spearman temp-TTFT {mean_telemetry(result, pol, 'spearman_temp_vs_local_ttft')}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["A", "B", "C", "dry"], required=True)
    args = ap.parse_args()
    result = json.loads((ANALYSIS / f"live_{args.exp}.json").read_text(encoding="utf-8"))
    body = {"A": exp_a, "B": exp_b, "C": exp_c}.get(args.exp, lambda r: "% dry run")(result)
    text = body + "\n" + common(result) + "\n"
    (ANALYSIS / f"paper_{args.exp}.tex").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
