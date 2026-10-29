"""Audit every run's event log against the routing agreement.

The agreement (governance/agreement.yaml) is modelled on SLA4OAI's structure:
context, metrics, a plan with guarantees (the TTFT SLO), quotas (cloud budget,
spend) and a governance block (artifact hash, breaker rule, promotion bar).
Enforced terms (hash, quota, breaker) are checked as compliant/violated;
guarantees are measured and reported, since an SLO can be missed without the
runtime misbehaving.

Blind rule: run only on experiments whose full corpus has completed.
Usage: python soc4ai/audit.py --runs soc4ai_out/runs/A [--runs ...] --out soc4ai_out/audit.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex.core import ttft_breach  # noqa: E402


def load_agreement(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def audit_run(jsonl: Path, agreement: dict, ledger: dict | None) -> dict:
    manifest = json.loads(jsonl.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    plan = agreement["plans"]["local_first"]
    gov = agreement["governance"]
    sla = float(plan["guarantees"]["ttft_slo"]["threshold_ms"])
    n = len(events)
    governed = manifest.get("policy") in {"guarded_governor", "paced_governor", "v4_governor"}
    terms = {}

    # Governance: artifact integrity (governed runs only)
    if governed:
        recorded = (manifest.get("governor_artifact") or {}).get("sha256")
        terms["artifact_hash"] = {"enforced": True, "expected": gov["artifact"]["sha256"],
                                  "observed": recorded, "compliant": recorded == gov["artifact"]["sha256"]}

    # Quota: cloud calls per episode. "all_dispatches" counts every cloud dispatch;
    # "billable" (v4 agreement) excludes half-open probes and failed calls.
    quota = plan["quotas"]["cloud_calls"]
    quota_frac = float(quota["max_fraction_of_episode"])
    counts = quota.get("counts", "all_dispatches")
    cap = math.floor(quota_frac * n)
    if counts == "billable":
        cloud_calls = sum(1 for e in events
                          if e["decision"]["route"] == "cloud" and bool(e["outcome"].get("ok"))
                          and "probe_exempt" not in (e["decision"].get("reason_code") or ""))
    else:
        cloud_calls = sum(1 for e in events if e["decision"]["route"] == "cloud")
    terms["cloud_call_quota"] = {"enforced": governed, "counts": counts, "limit": cap,
                                 "observed": cloud_calls, "compliant": cloud_calls <= cap}

    # Governance: no dispatch to cloud while the breaker is open
    open_dispatch = sum(1 for e in events
                        if e["decision"]["route"] == "cloud" and e["decision"].get("circuit_state") == "open")
    failed_cloud = sum(1 for e in events
                       if e["outcome"]["executed_on"] == "cloud" and not e["outcome"]["ok"])
    terms["breaker_rule"] = {"enforced": governed, "cloud_dispatch_while_open": open_dispatch,
                             "failed_cloud_calls": failed_cloud,
                             "compliant": open_dispatch == 0 if governed else None}

    # Quota: study-wide spend
    if ledger is not None:
        limit = float(plan["quotas"]["cloud_spend"]["max_usd_study"])
        terms["spend_quota"] = {"enforced": True, "limit_usd": limit,
                                "observed_usd": ledger.get("committed_usd"),
                                "compliant": float(ledger.get("committed_usd", 0.0)) < limit}

    # Guarantee (measured): TTFT SLO over all routes, failures count as misses
    misses = sum(ttft_breach(e["outcome"].get("end_to_end_ttft_ms"), sla, bool(e["outcome"].get("ok")))
                 for e in events)
    target = float(plan["guarantees"]["ttft_slo"]["max_violation_rate"])
    terms["ttft_slo"] = {"enforced": False, "threshold_ms": sla, "max_violation_rate": target,
                         "observed_violation_rate": misses / n if n else None,
                         "met": (misses / n) <= target if n else None}

    enforced = [t["compliant"] for t in terms.values() if t.get("enforced") and t.get("compliant") is not None]
    return {"run_id": manifest.get("run_id"), "policy": manifest.get("policy"),
            "n_requests": n, "terms": terms, "enforced_terms_compliant": all(enforced)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", action="append", required=True)
    ap.add_argument("--agreement", default=str(ROOT / "governance" / "agreement.yaml"))
    ap.add_argument("--ledger", default=str(ROOT / "soc4ai_out" / "spend_ledger.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    agreement = load_agreement(Path(args.agreement))
    ledger_path = Path(args.ledger)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists() else None
    reports = []
    for d in args.runs:
        for jsonl in sorted(Path(d).glob("*.jsonl")):
            if jsonl.with_suffix(".manifest.json").exists():
                reports.append(audit_run(jsonl, agreement, ledger))
    agreement_sha = hashlib.sha256(Path(args.agreement).read_bytes()).hexdigest()
    out = {"agreement_sha256": agreement_sha, "n_runs": len(reports), "reports": reports}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    bad = [r["run_id"] for r in reports if r["policy"] == "guarded_governor" and not r["enforced_terms_compliant"]]
    print(f"audited {len(reports)} runs; governed runs with an enforced-term violation: {bad or 'none'}")


if __name__ == "__main__":
    main()
