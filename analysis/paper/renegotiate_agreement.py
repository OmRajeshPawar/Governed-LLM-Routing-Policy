"""Agreement renegotiation: re-audit the existing live logs against new agreement
versions without re-running anything. CPU only, read-only.

The point of treating the routing policy as a machine-readable agreement is that
a term can be renegotiated and the existing evidence re-judged. Here the quota
term is varied (20/30/40/50% of episode requests) and the counting rule is
varied (every cloud dispatch, or only billable calls: successes that are not
half-open probes), and every completed live run is re-audited.

Each re-audit answers "does this recorded episode comply with the renegotiated
term". How the runtime itself adapts to a different quota is what the live
A/B/C/v4 runs measure.

Run with:  python renegotiate_agreement.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "renegotiation.json"
sys.path.insert(0, str(REPO))

from soc4ai.audit import audit_run, load_agreement  # noqa: E402

BASE = REPO / "governance" / "agreement.yaml"
LEDGER = REPO / "soc4ai_out" / "spend_ledger.json"
BLOCKS = ["A", "B", "C", "X", "V4N", "V4F", "V4D", "B_ext", "C_ext"]
QUOTAS = [0.20, 0.30, 0.40, 0.50]
COUNTS = ["all_dispatches", "billable"]
ARM = {"guarded_governor": "v2 governed", "paced_governor": "v3 governed",
       "v4_governor": "v4 governed", "tail_aware": "ungoverned"}


def variant(agreement: dict, fraction: float, counts: str) -> dict:
    a = copy.deepcopy(agreement)
    q = a["plans"]["local_first"]["quotas"]["cloud_calls"]
    q["max_fraction_of_episode"] = fraction
    q["counts"] = counts
    a["version"] = f"1.1.0-quota{int(fraction*100)}-{counts}"
    return a


def main() -> None:
    base = load_agreement(BASE)
    ledger = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else None
    runs = []
    for block in BLOCKS:
        d = REPO / "soc4ai_out" / "runs" / block
        if not d.exists():
            continue
        for jsonl in sorted(d.glob("*.jsonl")):
            if (jsonl.with_suffix(".manifest.json").exists()
                    and (d / f"{jsonl.stem}.done").exists()):
                runs.append((block, jsonl))
    print(f"{len(runs)} completed runs found")

    result = {"base_agreement": str(BASE.relative_to(REPO)).replace("\\", "/"),
              "n_runs": len(runs), "quotas": QUOTAS, "counts_rules": COUNTS, "cells": {}}
    for fraction in QUOTAS:
        for counts in COUNTS:
            a = variant(base, fraction, counts)
            cell: dict[str, dict] = {}
            for block, jsonl in runs:
                rep = audit_run(jsonl, a, ledger)
                arm = ARM.get(rep["policy"], rep["policy"])
                key = f"{block}:{arm}"
                c = cell.setdefault(key, {"runs": 0, "quota_compliant": 0, "slo_met": 0,
                                          "max_cloud_calls": 0, "limit": 0})
                q = rep["terms"]["cloud_call_quota"]
                c["runs"] += 1
                c["quota_compliant"] += int(bool(q["compliant"]))
                c["slo_met"] += int(bool(rep["terms"]["ttft_slo"]["met"]))
                c["max_cloud_calls"] = max(c["max_cloud_calls"], int(q["observed"]))
                c["limit"] = int(q["limit"])
            result["cells"][f"{a['version']}"] = cell
            summary = ", ".join(f"{k} {v['quota_compliant']}/{v['runs']}" for k, v in sorted(cell.items()))
            print(f"quota {fraction:.0%} counting {counts:<14} compliant: {summary}")
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
