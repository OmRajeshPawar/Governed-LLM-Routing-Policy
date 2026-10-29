from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EXPECTED_SEEDS = [211, 223, 227, 229, 233, 239, 241, 251]


def load_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def metrics(rows: list[dict], seed: int, policy: str) -> tuple[float, float]:
    vals = [r for r in rows if int(r["seed"]) == seed and r["policy"] == policy]
    if not vals:
        raise ValueError(f"No {policy} rows for seed {seed}")
    sla = np.mean([bool(r["sla_violation"]) for r in vals])
    p95 = np.quantile([float(r["end_to_end_ttft_ms"]) for r in vals], .95)
    return float(sla), float(p95)


def analyze(rows: list[dict], n_boot: int = 10_000, seed: int = 20260713) -> dict:
    present = sorted({int(r["seed"]) for r in rows})
    if present != EXPECTED_SEEDS:
        return {"status": "not_run", "reason": "complete_eight_seed_corpus_missing",
                "expected_seeds": EXPECTED_SEEDS, "present_seeds": present,
                "superiority_claim_allowed": False}
    per_seed = []
    for s in EXPECTED_SEEDS:
        tail_sla, tail_p95 = metrics(rows, s, "tail_aware")
        gov_sla, gov_p95 = metrics(rows, s, "guarded_governor")
        per_seed.append({"seed": s, "tail_sla_rate": tail_sla, "governor_sla_rate": gov_sla,
                         "sla_difference_pp": 100 * (gov_sla - tail_sla),
                         "tail_p95_ttft_ms": tail_p95, "governor_p95_ttft_ms": gov_p95,
                         "p95_ratio": gov_p95 / tail_p95})
    rng = np.random.default_rng(seed)
    primary, secondary = [], []
    for _ in range(n_boot):
        sampled = rng.choice(EXPECTED_SEEDS, len(EXPECTED_SEEDS), replace=True)
        sample_rows = [r for r in rows if int(r["seed"]) in sampled]
        tail_sla, gov_sla, tail_t, gov_t = [], [], [], []
        for s in sampled:
            tr = [r for r in rows if int(r["seed"]) == s and r["policy"] == "tail_aware"]
            gr = [r for r in rows if int(r["seed"]) == s and r["policy"] == "guarded_governor"]
            tail_sla.extend(bool(r["sla_violation"]) for r in tr)
            gov_sla.extend(bool(r["sla_violation"]) for r in gr)
            tail_t.extend(float(r["end_to_end_ttft_ms"]) for r in tr)
            gov_t.extend(float(r["end_to_end_ttft_ms"]) for r in gr)
        primary.append(100 * (np.mean(gov_sla) - np.mean(tail_sla)))
        secondary.append(np.quantile(gov_t, .95) / np.quantile(tail_t, .95))
    all_tail = [r for r in rows if r["policy"] == "tail_aware"]
    all_gov = [r for r in rows if r["policy"] == "guarded_governor"]
    diff = 100 * (np.mean([r["sla_violation"] for r in all_gov]) -
                  np.mean([r["sla_violation"] for r in all_tail]))
    ratio = np.quantile([r["end_to_end_ttft_ms"] for r in all_gov], .95) / np.quantile([r["end_to_end_ttft_ms"] for r in all_tail], .95)
    primary_ci = [float(np.quantile(primary, .025)), float(np.quantile(primary, .975))]
    secondary_ci = [float(np.quantile(secondary, .025)), float(np.quantile(secondary, .975))]
    primary_pass = primary_ci[1] <= 1.0
    secondary_pass = ratio <= .95 and secondary_ci[1] < 1.0
    catastrophic = any(x["sla_difference_pp"] > 3 or x["p95_ratio"] > 1.20 for x in per_seed)
    branch = ("mixed_catastrophic_seed" if catastrophic and primary_pass and secondary_pass else
              "bounded_superiority" if primary_pass and secondary_pass else
              "sla_noninferior_superiority_not_demonstrated" if primary_pass else
              "adaptive_governor_failed_sla_noninferiority")
    return {"status": "complete", "operating_point_total_cloud": .30,
            "primary": {"effect_pp": diff, "ci95": primary_ci, "pass": primary_pass},
            "key_secondary": {"p95_ratio": ratio, "ci95": secondary_ci, "pass": secondary_pass},
            "catastrophic_seed_regression": catastrophic, "result_branch": branch,
            "per_seed": per_seed,
            "limitation": "The confirmatory analysis estimates consistency across eight generated workload seeds in one hardware/software configuration. It does not establish broad population-level generalization."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="analysis/confirmatory_request_results.jsonl")
    ap.add_argument("--output", default="deliverables/APEX_LLM_FINAL/results/confirmatory_statistics.json")
    args = ap.parse_args()
    result = analyze(load_rows(Path(args.input)))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
