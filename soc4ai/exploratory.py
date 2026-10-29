"""Exploratory addendum: live test of candidate v3 (paced quota, fresh override).

NOT preregistered in July. The design follows the diagnosis of the preregistered
experiment A, so these runs are reported as exploratory. The addendum protocol is
still hashed before any v3 run, and the analysis stays blind until all runs finish.

Stages (run from soc4ai_runs with SOC4AI_GCP_KEY set):
  artifact   write ml/artifacts/governor_v3.json and governance/agreement_v3.yaml
  freeze     write soc4ai_out/protocol_x.json + protocol_x.sha256
  batch      run the frozen addendum (paired with experiment A seeds)
  analyze    paired comparison vs A's governed (v2) and ungoverned runs
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soc4ai import study  # noqa: E402  (forces the Dubai credentials on import)
from soc4ai.ledger import BudgetKillSwitch, SpendLedger  # noqa: E402
from soc4ai.paced import allowance_for, make_paced_governor  # noqa: E402

OUT = study.OUT
V3_ARTIFACT = ROOT / "ml" / "artifacts" / "governor_v3.json"
AGREEMENT_V2 = ROOT / "governance" / "agreement.yaml"
AGREEMENT_V3 = ROOT / "governance" / "agreement_v3.yaml"
PROTOCOL_X = OUT / "protocol_x.json"
PROTOCOL_X_HASH = OUT / "protocol_x.sha256"
PAIRS = [(211, "synthetic"), (223, "real_sample"), (227, "reasoning_coding")]


def stage_artifact() -> None:
    v2 = json.loads(study.V2_ARTIFACT.read_text(encoding="utf-8"))
    v3 = dict(v2)
    v3["artifact_id"] = "apex-paced-fresh-exploratory-2026-09-15"
    v3["created_utc"] = study.now_utc()
    v3["promotion_status"] = {
        "status": "exploratory_candidate_not_promoted",
        "reason": "Candidate designed from the diagnosis of preregistered live experiment A; "
                  "exploratory evidence cannot promote it (3 seeds < 5 required by the bar).",
        "changes": {
            "budget": "paced: released = min(cap, floor(0.30 * arrivals) + ceil(0.05 * episode))",
            "hard_override": "rolling-P95 hard breach applies only when the local tier is busy "
                             "(pending >= 1 or all concurrency slots active); queue hard cap unchanged",
        },
        "derived_from_artifact_sha256": study.sha256_file(study.V2_ARTIFACT),
        "confirmatory_data_used": "diagnosis used preregistered experiment A outcomes",
    }
    V3_ARTIFACT.write_text(json.dumps(v3, indent=2), encoding="utf-8")
    v3_sha = study.sha256_file(V3_ARTIFACT)
    text = AGREEMENT_V2.read_text(encoding="utf-8")
    v2_sha = study.sha256_file(study.V2_ARTIFACT)
    if v2_sha not in text:
        raise SystemExit("agreement.yaml does not pin the v2 artifact hash")
    text = text.replace(v2_sha, v3_sha).replace("governor_v2.json", "governor_v3.json")
    text = text.replace("id: apex-llm-routing-agreement", "id: apex-llm-routing-agreement-v3-exploratory")
    AGREEMENT_V3.write_text(text, encoding="utf-8")
    study.log(f"exploratory artifact v3 written, sha256 {v3_sha[:16]}; agreement_v3 pins it")


def stage_freeze() -> None:
    parent = study.PROTOCOL_HASH.read_text(encoding="utf-8").strip()
    main_protocol = json.loads(study.PROTOCOL_PATH.read_text(encoding="utf-8"))
    runs = [{"experiment": "X", "key": f"X_s{seed}_{mix}_paced_governor", "trace": "bursty", "n": 300,
             "seed": seed, "mix": mix, "policy": "paced_governor", "condition": "nominal", "faults": {},
             "paired_with": [f"A_s{seed}_{mix}_guarded_governor", f"A_s{seed}_{mix}_tail_aware"]}
            for seed, mix in PAIRS]
    cfg, _ = study.base_config()
    protocol = {
        "created_utc": study.now_utc(),
        "status": "exploratory addendum, not preregistered",
        "parent_protocol_sha256": parent,
        "motivation": "Preregistered experiment A rejected the v2 governed runtime "
                      "(adaptive_governor_failed_sla_noninferiority); the quota was exhausted by "
                      "request 96-144 through sticky hard-override offloads.",
        "candidate": json.loads(V3_ARTIFACT.read_text(encoding="utf-8"))["promotion_status"]["changes"],
        "allowance": allowance_for(300),
        "governor_artifact": {"path": "ml/artifacts/governor_v3.json", "sha256": study.sha256_file(V3_ARTIFACT)},
        "agreement_sha256": study.sha256_file(AGREEMENT_V3),
        "heuristic_weights_values": main_protocol["heuristic_weights_values"],
        "frozen_cloud_p95_ms": main_protocol["frozen_cloud_p95_ms"],
        "sla_ttft_ms": main_protocol["sla_ttft_ms"],
        "analysis_plan": "per seed: SLO violation, P95 TTFT, cloud share, spend, failures, quota audit; "
                         "paired differences vs A governed (v2) and A ungoverned on identical traces; "
                         "gates reported descriptively only (3 seeds cannot satisfy G1)",
        "blind_rule": "no X outcome inspected before all X runs complete",
        "environment": {**study.ollama_fingerprint(cfg.providers.local.model), "gpu_driver": study.gpu_driver()},
        "code_sha256": {"soc4ai/paced.py": study.sha256_file(ROOT / "soc4ai" / "paced.py"),
                        "soc4ai/exploratory.py": study.sha256_file(ROOT / "soc4ai" / "exploratory.py")},
        "runs": runs,
    }
    text = json.dumps(protocol, indent=1)
    PROTOCOL_X.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    PROTOCOL_X_HASH.write_text(digest + "\n", encoding="utf-8")
    study.log(f"exploratory protocol frozen: {len(runs)} runs, sha256 {digest[:16]}, parent {parent[:16]}")


async def stage_batch() -> int:
    text = PROTOCOL_X.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if PROTOCOL_X_HASH.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit("protocol_x.json does not match its hash")
    protocol = json.loads(text)
    if study.sha256_file(V3_ARTIFACT) != protocol["governor_artifact"]["sha256"]:
        raise SystemExit("governor_v3.json changed after the addendum was frozen")
    if (OUT / "HALTED").exists():
        raise SystemExit("main batch halted; refusing to run the addendum")
    if not study.original_repo_status()["clean"]:
        raise SystemExit("source repository has uncommitted changes (commit first, so the protocol records a clean tree)")
    cfg0, _ = study.base_config()
    env_now = study.ollama_fingerprint(cfg0.providers.local.model)
    frozen_env = protocol["environment"]
    if (env_now["ollama_version"], env_now["model_id"]) != (frozen_env["ollama_version"], frozen_env["model_id"]):
        raise SystemExit(f"environment drift: {env_now}")
    study.make_guarded_governor = make_paced_governor  # run_one builds the governor through this name
    ledger = SpendLedger(study.LEDGER_PATH)
    p95 = float(protocol["frozen_cloud_p95_ms"])
    study.log(f"ADDENDUM START {len(protocol['runs'])} runs, protocol_x {digest[:16]}, ledger ${ledger.committed:.4f}")
    exp_dir = OUT / "runs" / "X"
    for idx, spec in enumerate(protocol["runs"], 1):
        done = exp_dir / f"{spec['key']}.done"
        if done.exists():
            continue
        for attempt in (1, 2):
            waited, temp = 0.0, study.gpu_temp_c()
            while temp is not None and temp >= study.COOLDOWN_C and waited < study.COOLDOWN_MAX_S:
                time.sleep(10)
                waited += 10
                temp = study.gpu_temp_c()
            cfg, _ = study.base_config()
            cfg.governor.artifact_path = "ml/artifacts/governor_v3.json"
            cfg.seed = spec["seed"]
            try:
                m = await study.run_one(cfg, study.build_trace(spec), policy_name="guarded_governor",
                                        weights=protocol["heuristic_weights_values"],
                                        cloud=study.build_cloud(spec, ledger, protocol), run_id=spec["key"],
                                        out_dir=exp_dir, frozen_p95=p95,
                                        extra_manifest={"protocol_x_sha256": digest, "spec": spec,
                                                        "cooldown_waited_s": waited, "gpu_temp_start_c": temp})
                done.write_text(study.now_utc(), encoding="utf-8")
                study.log(f"[X {idx}/{len(protocol['runs'])}] done {spec['key']} wall {m['wall_clock_s']:.0f}s "
                          f"ledger ${ledger.committed:.4f}")
                break
            except BudgetKillSwitch as exc:
                study.log(f"KILL SWITCH during {spec['key']}: {exc}")
                (OUT / "HALTED").write_text(study.now_utc() + " " + str(exc), encoding="utf-8")
                return 3
            except Exception:
                err = OUT / "errors"
                err.mkdir(parents=True, exist_ok=True)
                (err / f"{spec['key']}_attempt{attempt}.txt").write_text(traceback.format_exc(), encoding="utf-8")
                study.log(f"[X {idx}] ERROR {spec['key']} attempt {attempt}")
        else:
            study.log(f"[X {idx}] SKIPPED {spec['key']}")
    study.log(f"ADDENDUM END ledger ${ledger.committed:.4f}, original repo clean={study.original_repo_status()['clean']}")
    return 0


def stage_analyze() -> None:
    from soc4ai import analyze_live as al
    from soc4ai.audit import audit_run, load_agreement
    protocol = json.loads(PROTOCOL_X.read_text(encoding="utf-8"))
    missing = [r["key"] for r in protocol["runs"] if not (OUT / "runs" / "X" / f"{r['key']}.done").exists()]
    if missing:
        raise SystemExit(f"blind rule: addendum incomplete, missing {missing}")
    ledger = json.loads(study.LEDGER_PATH.read_text(encoding="utf-8"))
    agreements = {"X": load_agreement(AGREEMENT_V3), "A": load_agreement(AGREEMENT_V2)}
    per_seed, pooled = {}, {"paced_governor": [], "guarded_governor": [], "tail_aware": []}
    for spec in protocol["runs"]:
        seed, mix = spec["seed"], spec["mix"]
        paths = {"paced_governor": OUT / "runs" / "X" / f"{spec['key']}.jsonl",
                 "guarded_governor": OUT / "runs" / "A" / f"A_s{seed}_{mix}_guarded_governor.jsonl",
                 "tail_aware": OUT / "runs" / "A" / f"A_s{seed}_{mix}_tail_aware.jsonl"}
        entry = {}
        for pol, path in paths.items():
            df, manifest, gpu = al.load_run(path)
            pooled[pol].append(df)
            audit = audit_run(path, agreements["X" if pol == "paced_governor" else "A"], ledger)
            entry[pol] = {**al.run_summary(df), "energy": al.energy_and_thermal(df, gpu),
                          "budget": manifest.get("budget"),
                          "quota_compliant": audit["terms"]["cloud_call_quota"]["compliant"],
                          "artifact_hash_ok": audit["terms"].get("artifact_hash", {}).get("compliant")}
        entry["diff_vs_governed_pp"] = 100 * (entry["paced_governor"]["violation_rate"] - entry["guarded_governor"]["violation_rate"])
        entry["diff_vs_ungoverned_pp"] = 100 * (entry["paced_governor"]["violation_rate"] - entry["tail_aware"]["violation_rate"])
        per_seed[f"{seed}_{mix}"] = entry
    import pandas as pd
    pooled_summary = {pol: al.run_summary(pd.concat(frames, ignore_index=True)) for pol, frames in pooled.items()}
    result = {"protocol_x_sha256": PROTOCOL_X_HASH.read_text(encoding="utf-8").strip(),
              "per_seed": per_seed, "pooled": pooled_summary}
    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "live_X.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    for pol, s in pooled_summary.items():
        print(f"{pol:18s} viol {100*s['violation_rate']:5.1f}%  P95 {s['p95_ttft_ms']:8.0f} ms  "
              f"cloud {100*s['cloud_share']:5.1f}%  failed {s['failed']:3d}  spend ${s['spend_usd']:.3f}")
    for k, e in per_seed.items():
        print(k, "| paced-governed %+.1f pp | paced-ungoverned %+.1f pp | paced quota ok %s | budget %s"
              % (e["diff_vs_governed_pp"], e["diff_vs_ungoverned_pp"], e["paced_governor"]["quota_compliant"],
                 e["paced_governor"]["budget"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["artifact", "freeze", "batch", "analyze"])
    args = ap.parse_args()
    if args.stage == "artifact":
        stage_artifact()
    elif args.stage == "freeze":
        stage_freeze()
    elif args.stage == "batch":
        sys.exit(asyncio.run(stage_batch()))
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
