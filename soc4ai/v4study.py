"""Preregistered confirmatory test of candidate v4, plus 5-seed extensions of
chaos (B) and drift (C).

Stages (run from soc4ai_runs with SOC4AI_GCP_KEY set):
  artifact  write ml/artifacts/governor_v4.json and governance/agreement_v4.yaml
  freeze    write soc4ai_out/protocol_v4.json + protocol_v4.sha256 (before any run)
  dryrun    throwaway n=24 v4 runs (nominal and forced outage) to test the pipeline
  batch     execute the frozen matrix: V4N (nominal) -> V4F (faults) -> B_ext -> C_ext
  analyze   blind analysis once every run in the protocol is done

Primary (preregistered): nominal block, candidate v4 vs incumbent v2 on fresh seeds
257, 263, 269, 271, 277, judged by the five promotion gates with a seed-level
bootstrap. Secondary: v4 vs ungoverned, faults block, agreement audit, and the
5-seed chaos and drift extensions.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soc4ai import study  # noqa: E402  (forces the credentials on import when a key is set)
from apex.policies import make_guarded_governor as make_v2_governor  # noqa: E402
from soc4ai.ledger import BudgetKillSwitch, SpendLedger  # noqa: E402
from soc4ai.v4 import make_v4_governor  # noqa: E402

OUT = study.OUT
V3_ARTIFACT = ROOT / "ml" / "artifacts" / "governor_v3.json"
V4_ARTIFACT = ROOT / "ml" / "artifacts" / "governor_v4.json"
AGREEMENT_V2 = ROOT / "governance" / "agreement.yaml"
AGREEMENT_V3 = ROOT / "governance" / "agreement_v3.yaml"
AGREEMENT_V4 = ROOT / "governance" / "agreement_v4.yaml"
PROTOCOL_V4 = OUT / "protocol_v4.json"
PROTOCOL_V4_HASH = OUT / "protocol_v4.sha256"
V4_SEEDS = [(257, "synthetic"), (263, "real_sample"), (269, "reasoning_coding"),
            (271, "synthetic"), (277, "real_sample")]
# 23 and 101 were offline shadow-corpus seeds (data/events, July); never used in a live run,
# the same status as B/C seeds 7/13/42.
EXT_SEEDS = [(23, "synthetic"), (101, "real_sample")]
DRIFT_SEEDS = [7, 13, 42, 23, 101]
FAULTS = {"outage": [96, 156], "spike": [192, 252]}
POLICY_NAME = {"v4": "v4_governor", "v2": "guarded_governor", "ungoverned": "tail_aware"}
BOOT_B = 10000
BOOT_SEED = 20260916


def stage_artifact() -> None:
    v3 = json.loads(V3_ARTIFACT.read_text(encoding="utf-8"))
    v4 = dict(v3)
    v4["artifact_id"] = "apex-v4-paced-fresh-refund-2026-09-16"
    v4["created_utc"] = study.now_utc()
    changes = dict(v3["promotion_status"]["changes"])
    changes["probe_accounting"] = ("half-open breaker probes are not charged to the quota; "
                                   "failed charged cloud calls are refunded")
    v4["promotion_status"] = {
        "status": "candidate_under_preregistered_confirmatory_test",
        "reason": "Candidate addresses the three mechanisms localised by the agreement audits of "
                  "experiments A and B; promotion requires all five gates on fresh seeds.",
        "changes": changes,
        "derived_from_artifact_sha256": study.sha256_file(V3_ARTIFACT),
        "confirmatory_data_used": False,
    }
    V4_ARTIFACT.write_text(json.dumps(v4, indent=2), encoding="utf-8")
    v4_sha = study.sha256_file(V4_ARTIFACT)
    v3_sha = study.sha256_file(V3_ARTIFACT)
    text = AGREEMENT_V3.read_text(encoding="utf-8")
    if v3_sha not in text:
        raise SystemExit("agreement_v3.yaml does not pin the v3 artifact hash")
    text = text.replace(v3_sha, v4_sha).replace("governor_v3.json", "governor_v4.json")
    text = text.replace("id: apex-llm-routing-agreement-v3-exploratory", "id: apex-llm-routing-agreement-v4")
    old_quota = "cloud_calls: {metric: cloud_calls, max_fraction_of_episode: 0.30, enforcement: hard}"
    if old_quota not in text:
        raise SystemExit("quota line not found in agreement_v3.yaml")
    text = text.replace(old_quota, "cloud_calls: {metric: cloud_calls, max_fraction_of_episode: 0.30, "
                                   "enforcement: hard, counts: billable}")
    AGREEMENT_V4.write_text(text, encoding="utf-8")
    study.log(f"v4 artifact written, sha256 {v4_sha[:16]}; agreement_v4 pins it with billable quota counting")


def _order(i: int) -> list[str]:
    base = ["v4", "v2", "ungoverned"]
    return base[i % 3:] + base[:i % 3]


def build_runs() -> list[dict]:
    runs = []
    for block, faults, exp in (("V4N", {}, "nominal"), ("V4F", FAULTS, "chaos")):
        for i, (seed, mix) in enumerate(V4_SEEDS):
            for arm in _order(i):
                runs.append({"experiment": block, "key": f"{block}_s{seed}_{mix}_{POLICY_NAME[arm]}",
                             "trace": "bursty", "n": 300, "seed": seed, "mix": mix, "arm": arm,
                             "policy": POLICY_NAME[arm], "condition": exp, "faults": faults})
    for seed in DRIFT_SEEDS:
        runs.append({"experiment": "V4D", "key": f"V4D_s{seed}_transition_v4_governor", "trace": "transition",
                     "n": 200, "seed": seed, "mix": "phased", "arm": "v4", "policy": POLICY_NAME["v4"],
                     "condition": "drift", "faults": {"degraded_phase": True}})
    for i, (seed, mix) in enumerate(EXT_SEEDS):
        for arm in (["v2", "ungoverned"] if i % 2 == 0 else ["ungoverned", "v2"]):
            runs.append({"experiment": "B_ext", "key": f"B_s{seed}_{mix}_{POLICY_NAME[arm]}", "trace": "bursty",
                         "n": 300, "seed": seed, "mix": mix, "arm": arm, "policy": POLICY_NAME[arm],
                         "condition": "chaos", "faults": FAULTS})
    for i, (seed, _mix) in enumerate(EXT_SEEDS):
        for arm in (["v2", "ungoverned"] if i % 2 == 0 else ["ungoverned", "v2"]):
            runs.append({"experiment": "C_ext", "key": f"C_s{seed}_transition_{POLICY_NAME[arm]}",
                         "trace": "transition", "n": 200, "seed": seed, "mix": "phased", "arm": arm,
                         "policy": POLICY_NAME[arm], "condition": "drift", "faults": {"degraded_phase": True}})
    return runs


def stage_freeze() -> None:
    main = json.loads(study.PROTOCOL_PATH.read_text(encoding="utf-8"))
    cfg, _ = study.base_config()
    used = {7, 13, 23, 42, 101, 211, 223, 227, 229, 233, 239, 241, 251, 998, 999, 9001, 9002, 9003}
    fresh = [s for s, _ in V4_SEEDS]
    if used & set(fresh):
        raise SystemExit("v4 seeds overlap previously used seeds")
    runs = build_runs()
    code_files = sorted((ROOT / "soc4ai").glob("*.py")) + [
        ROOT / "apex" / "policies.py", ROOT / "apex" / "gateway.py", ROOT / "apex" / "workload.py",
        V4_ARTIFACT, study.V2_ARTIFACT, AGREEMENT_V2, AGREEMENT_V4]
    protocol = {
        "created_utc": study.now_utc(),
        "status": "preregistered confirmatory test of candidate v4 (hashed before any v4 run)",
        "parent_protocols_sha256": {"main": study.PROTOCOL_HASH.read_text(encoding="utf-8").strip(),
                                    "exploratory_x": (OUT / "protocol_x.sha256").read_text(encoding="utf-8").strip()},
        "fresh_seeds": V4_SEEDS,
        "incumbent": {"arm": "v2", "artifact": "ml/artifacts/governor_v2.json",
                      "sha256": study.sha256_file(study.V2_ARTIFACT), "agreement": "governance/agreement.yaml"},
        "candidate": {"arm": "v4", "artifact": "ml/artifacts/governor_v4.json",
                      "sha256": study.sha256_file(V4_ARTIFACT), "agreement": "governance/agreement_v4.yaml",
                      "changes": json.loads(V4_ARTIFACT.read_text(encoding="utf-8"))["promotion_status"]["changes"]},
        "comparator": {"arm": "ungoverned", "policy": "tail_aware", "note": "same expert weights and threshold, no breaker or quota"},
        "primary_analysis": {
            "block": "V4N (nominal, bursty n=300, live Gemini)",
            "comparison": "v4 (candidate) vs v2 (incumbent) on identical traces",
            "gates": {
                "G1_seed_support": "v4 violation <= v2 violation in >= 4 of 5 seeds",
                "G2_slo_bound": "upper 95% CI of pooled violation difference (v4 - v2, points) <= +1.0",
                "G3_tail_gain": "pooled P95 TTFT ratio v4/v2 <= 0.95 and CI upper bound < 1.00",
                "G4_no_regression": "no seed more than 5 points worse",
                "G5_decision_cap": "median decision time v4 - v2 <= 5 ms",
            },
            "bootstrap": {"unit": "seed (resampled with replacement, requests pooled)", "B": BOOT_B, "rng_seed": BOOT_SEED},
            "verdict": "promote only if all five gates pass",
        },
        "secondary_analyses": [
            "V4N: v4 vs ungoverned violation difference and P95 ratio with the same bootstrap; cloud share, spend, "
            "cost per SLO-met request, GPU energy per SLO-met request",
            "agreement audit: v4 under agreement_v4 (billable quota counting), v2 under agreement (all dispatches)",
            "V4F: per-seed and pooled violation, failed requests, wasted spend, breaker detection and recovery, "
            "for v4, v2, ungoverned, with seed bootstrap CIs",
            "B and C extended to 5 seeds (existing seeds 7/13/42 plus 23/101): seed bootstrap CIs for violation "
            "and failed requests, same conditions as the main protocol",
            "V4D: v4 on the drift trace for seeds 7/13/42/23/101, compared with v2 and ungoverned from C and C_ext: "
            "degraded-phase violation with seed bootstrap CIs; for every governed drift run, whether the breaker "
            "opened during the degraded phase and the trace position at which the quota was exhausted (if ever)",
            "exact sign-flip permutation test (2^k sign assignments) on per-seed violation differences, "
            "reported next to the bootstrap CI for v4 vs v2 and v4 vs ungoverned",
            "for every governed run: trace position of quota exhaustion, refunds and exempt probes are "
            "reconstructed from events (budget_remaining, reason codes)",
        ],
        "ext_seed_note": "seeds 23 and 101 were offline shadow-corpus seeds only; never used live",
        "truncation_rule": "blocks run in priority order V4N, V4F, V4D, B_ext, C_ext. If the batch is stopped "
                           "for time (not for any outcome), analysis proceeds once soc4ai_out/V4_STOPPED exists; "
                           "V4N must be complete for the primary verdict; incomplete secondary blocks are "
                           "reported with the seeds that completed, and missing runs are listed.",
        "sla_ttft_ms": main["sla_ttft_ms"],
        "sla_violation_rule": main["sla_violation_rule"],
        "failed_request_ttft_rule": main["confirmatory_ttft_rule"],
        "heuristic_weights_values": main["heuristic_weights_values"],
        "tail_aware_threshold": main["tail_aware_threshold"],
        "frozen_cloud_p95_ms": main["frozen_cloud_p95_ms"],
        "breaker": main["breaker"],
        "fault_model": main["fault_model"],
        "blind_rule": "progress log carries no outcome metrics; no block analysed before all runs in the protocol complete",
        "run_order": "V4N then V4F then V4D then B_ext then C_ext; within a seed the arm order rotates",
        "environment": {**study.ollama_fingerprint(cfg.providers.local.model), "gpu_driver": study.gpu_driver()},
        "original_repo": study.original_repo_status(),
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): study.sha256_file(p) for p in code_files if p.exists()},
        "runs": runs,
    }
    text = json.dumps(protocol, indent=1)
    PROTOCOL_V4.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    PROTOCOL_V4_HASH.write_text(digest + "\n", encoding="utf-8")
    study.log(f"v4 protocol frozen: {len(runs)} runs, sha256 {digest[:16]}")


def _factory_for(arm: str):
    return {"v4": make_v4_governor, "v2": make_v2_governor}.get(arm)


def _config_for(arm: str, seed: int):
    cfg, _ = study.base_config()
    if arm == "v4":
        cfg.governor.artifact_path = "ml/artifacts/governor_v4.json"
    cfg.seed = seed
    return cfg


async def _run_spec(spec: dict, ledger: SpendLedger, protocol: dict, out_dir: Path, extra: dict) -> dict:
    arm = spec["arm"]
    factory = _factory_for(arm)
    original = study.make_guarded_governor
    if factory is not None:
        study.make_guarded_governor = factory
    try:
        cfg = _config_for(arm, spec["seed"])
        return await study.run_one(cfg, study.build_trace(spec),
                                   policy_name="guarded_governor" if factory else "tail_aware",
                                   weights=protocol["heuristic_weights_values"],
                                   cloud=study.build_cloud(spec, ledger, protocol), run_id=spec["key"],
                                   out_dir=out_dir, frozen_p95=float(protocol["frozen_cloud_p95_ms"]),
                                   extra_manifest=extra)
    finally:
        study.make_guarded_governor = original


async def stage_dryrun() -> None:
    main = json.loads(study.PROTOCOL_PATH.read_text(encoding="utf-8"))
    ledger = SpendLedger(study.LEDGER_PATH)
    stub = {"heuristic_weights_values": main["heuristic_weights_values"],
            "frozen_cloud_p95_ms": main["frozen_cloud_p95_ms"], "fault_model": main["fault_model"]}
    out_dir = OUT / "dryrun_v4"
    for name, faults in (("dryrun_v4_outage96", {"outage": [24, 72]}),):
        spec = {"trace": "bursty", "n": 96, "seed": 997, "mix": "reasoning_coding", "arm": "v4", "faults": faults, "key": name}
        m = await _run_spec(spec, ledger, stub, out_dir, {"dryrun": True})
        study.log(f"{name}: offloaded {m['n_offloaded']}, wall {m['wall_clock_s']}s, budget {m.get('budget')}, "
                  f"transitions {len(m.get('breaker_transitions', []))}, ledger ${ledger.committed:.4f}")


async def stage_batch() -> int:
    text = PROTOCOL_V4.read_text(encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    if PROTOCOL_V4_HASH.read_text(encoding="utf-8").strip() != digest:
        raise SystemExit("protocol_v4.json does not match its hash")
    protocol = json.loads(text)
    if study.sha256_file(V4_ARTIFACT) != protocol["candidate"]["sha256"]:
        raise SystemExit("governor_v4.json changed after the protocol was frozen")
    if study.sha256_file(study.V2_ARTIFACT) != protocol["incumbent"]["sha256"]:
        raise SystemExit("governor_v2.json changed after the protocol was frozen")
    if (OUT / "HALTED").exists():
        raise SystemExit("spend kill switch has halted the study")
    if not study.original_repo_status()["clean"]:
        raise SystemExit("source repository has uncommitted changes (commit first, so the protocol records a clean tree)")
    cfg0, _ = study.base_config()
    frozen_env = protocol["environment"]
    ledger = SpendLedger(study.LEDGER_PATH)
    runs = protocol["runs"]
    study.log(f"V4 BATCH START {len(runs)} runs, protocol_v4 {digest[:16]}, ledger ${ledger.committed:.4f}")
    for idx, spec in enumerate(runs, 1):
        out_dir = OUT / "runs" / spec["experiment"]
        done = out_dir / f"{spec['key']}.done"
        if done.exists():
            continue
        for attempt in (1, 2):
            waited, temp = 0.0, study.gpu_temp_c()
            while temp is not None and temp >= study.COOLDOWN_C and waited < study.COOLDOWN_MAX_S:
                time.sleep(10)
                waited += 10
                temp = study.gpu_temp_c()
            env = study.ollama_fingerprint(cfg0.providers.local.model)
            if (env["ollama_version"], env["model_id"]) != (frozen_env["ollama_version"], frozen_env["model_id"]):
                study.log(f"HALT environment drift before {spec['key']}: {env}")
                return 4
            try:
                m = await _run_spec(spec, ledger, protocol, out_dir,
                                    {"protocol_v4_sha256": digest, "spec": spec, "cooldown_waited_s": waited,
                                     "gpu_temp_start_c": temp, "environment": env})
                done.write_text(study.now_utc(), encoding="utf-8")
                study.log(f"[V4 {idx}/{len(runs)}] done {spec['key']} wall {m['wall_clock_s']:.0f}s "
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
                study.log(f"[V4 {idx}] ERROR {spec['key']} attempt {attempt}")
        else:
            study.log(f"[V4 {idx}] SKIPPED {spec['key']}")
    study.log(f"V4 BATCH END ledger ${ledger.committed:.4f}, original repo clean={study.original_repo_status()['clean']}")
    return 0


# ------------------------------------------------------------------ analysis
def _load(path: Path) -> dict:
    import numpy as np
    from apex.core import ttft_breach
    sla = 3839.5
    viol, ttft, dec = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        e = json.loads(line)
        o = e["outcome"]
        ok = bool(o.get("ok"))
        t = o.get("end_to_end_ttft_ms")
        viol.append(ttft_breach(t, sla, ok))
        ttft.append(t if (ok and t is not None) else max(float(o.get("end_to_end_ms") or 0.0), sla))
        if o.get("decision_ms") is not None:
            dec.append(float(o["decision_ms"]))
    return {"viol": np.array(viol, dtype=float), "ttft": np.array(ttft, dtype=float), "dec": np.array(dec, dtype=float)}


def _seed_bootstrap(cand: dict, inc: dict) -> dict:
    import numpy as np
    seeds = sorted(cand)
    rng = np.random.default_rng(BOOT_SEED)

    def pool(sample, d, key):
        return np.concatenate([d[s][key] for s in sample])

    diff = 100 * (pool(seeds, cand, "viol").mean() - pool(seeds, inc, "viol").mean())
    ratio = np.percentile(pool(seeds, cand, "ttft"), 95) / np.percentile(pool(seeds, inc, "ttft"), 95)
    diffs, ratios = np.empty(BOOT_B), np.empty(BOOT_B)
    for b in range(BOOT_B):
        smp = rng.choice(seeds, len(seeds), replace=True)
        diffs[b] = 100 * (pool(smp, cand, "viol").mean() - pool(smp, inc, "viol").mean())
        ratios[b] = np.percentile(pool(smp, cand, "ttft"), 95) / np.percentile(pool(smp, inc, "ttft"), 95)
    per_seed = {int(s): float(100 * (cand[s]["viol"].mean() - inc[s]["viol"].mean())) for s in seeds}
    support = sum(cand[s]["viol"].mean() <= inc[s]["viol"].mean() for s in seeds)
    perm_p = _sign_flip_p(list(per_seed.values()))
    dec = float(np.median(pool(seeds, cand, "dec")) - np.median(pool(seeds, inc, "dec")))
    g = {
        "violation_diff_pp": float(diff), "violation_diff_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "p95_ratio": float(ratio), "p95_ratio_ci95": [float(np.percentile(ratios, 2.5)), float(np.percentile(ratios, 97.5))],
        "per_seed_diff_pp": per_seed, "seed_support": int(support), "decision_ms_added": dec,
        "sign_flip_permutation_p_two_sided": perm_p,
    }
    g["gates"] = {
        "G1_seed_support": bool(support >= 4),
        "G2_slo_bound": bool(g["violation_diff_ci95"][1] <= 1.0),
        "G3_tail_gain": bool(ratio <= 0.95 and g["p95_ratio_ci95"][1] < 1.0),
        "G4_no_regression": bool(max(per_seed.values()) <= 5.0),
        "G5_decision_cap": bool(dec <= 5.0),
    }
    g["promote"] = all(g["gates"].values())
    return g


def governed_mechanism(df, manifest: dict, degraded_start: float | None = None) -> dict:
    """Quota exhaustion, exempt probes and breaker behaviour, reconstructed from events."""
    info = {}
    cloud = df[df["route"] == "cloud"]
    info["cloud_calls"] = int(len(cloud))
    info["exempt_probes"] = int(cloud["reason"].str.contains("probe_exempt", na=False).sum())
    info["failed_cloud_calls"] = int((~cloud["ok"]).sum())
    denied_mask = df["budget_denied"].fillna(False).astype(bool)
    info["budget_denied"] = int(denied_mask.sum())
    info["first_budget_denial_pos"] = int(df.loc[denied_mask, "trace_pos"].min()) if denied_mask.any() else None
    tr = manifest.get("breaker_transitions") or []
    info["breaker_opens"] = sum(1 for t in tr if t["to"] == "open")
    if degraded_start is not None:
        opens = [t for t in tr if t["to"] == "open" and t["unix_s"] >= degraded_start]
        info["breaker_opened_in_degraded_phase"] = bool(opens)
        info["degraded_detect_s"] = round(opens[0]["unix_s"] - degraded_start, 2) if opens else None
    return info


def _sign_flip_p(diffs: list[float]) -> float:
    import numpy as np
    d = np.array(diffs, dtype=float)
    signs = np.array(np.meshgrid(*[[1, -1]] * len(d))).reshape(len(d), -1).T
    return float(np.mean(np.abs((signs * d).mean(axis=1)) >= abs(d.mean()) - 1e-12))


def _pooled_ci(runs_by_seed: dict, metric) -> dict:
    import numpy as np
    seeds = sorted(runs_by_seed)
    rng = np.random.default_rng(BOOT_SEED)
    point = metric([runs_by_seed[s] for s in seeds])
    vals = np.empty(BOOT_B)
    for b in range(BOOT_B):
        smp = rng.choice(seeds, len(seeds), replace=True)
        vals[b] = metric([runs_by_seed[s] for s in smp])
    return {"point": float(point), "ci95": [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]}


def stage_analyze() -> None:
    import numpy as np
    from soc4ai import analyze_live as al
    from soc4ai.audit import audit_run, load_agreement
    protocol = json.loads(PROTOCOL_V4.read_text(encoding="utf-8"))
    def is_done(r):
        return (OUT / "runs" / r["experiment"] / f"{r['key']}.done").exists()
    missing = [r["key"] for r in protocol["runs"] if not is_done(r)]
    if missing and not (OUT / "V4_STOPPED").exists():
        raise SystemExit(f"blind rule: {len(missing)} runs missing and batch not stopped, e.g. {missing[:3]}")
    if any(r["experiment"] == "V4N" and not is_done(r) for r in protocol["runs"]):
        raise SystemExit("truncation rule: primary block V4N incomplete; no verdict")
    ledger = json.loads(study.LEDGER_PATH.read_text(encoding="utf-8"))
    agreements = {"v4": load_agreement(AGREEMENT_V4), "v2": load_agreement(AGREEMENT_V2),
                  "ungoverned": load_agreement(AGREEMENT_V2)}
    result = {"protocol_v4_sha256": PROTOCOL_V4_HASH.read_text(encoding="utf-8").strip(),
              "missing_runs": missing}

    for block in ("V4N", "V4F"):
        data = {arm: {} for arm in POLICY_NAME}
        summaries = {arm: {} for arm in POLICY_NAME}
        breaker = {}
        for spec in [r for r in protocol["runs"] if r["experiment"] == block and is_done(r)]:
            path = OUT / "runs" / block / f"{spec['key']}.jsonl"
            data[spec["arm"]][spec["seed"]] = _load(path)
            df, manifest, gpu = al.load_run(path)
            audit = audit_run(path, agreements[spec["arm"]], ledger)
            s = al.run_summary(df)
            s["energy"] = al.energy_and_thermal(df, gpu)
            s["quota"] = audit["terms"]["cloud_call_quota"]
            s["artifact_hash_ok"] = audit["terms"].get("artifact_hash", {}).get("compliant")
            s["budget"] = manifest.get("budget")
            if spec["arm"] != "ungoverned":
                s["mechanism"] = governed_mechanism(df, manifest)
            if block == "V4F":
                seg = df[(df["trace_pos"] >= 96) & (df["trace_pos"] < 156)]
                spk = df[(df["trace_pos"] >= 192) & (df["trace_pos"] < 252)]
                s["outage"] = al.run_summary(seg)
                s["spike"] = al.run_summary(spk)
                if manifest.get("breaker_transitions") is not None:
                    tr = manifest["breaker_transitions"]
                    t0 = float(df.loc[df["trace_pos"] == 96, "arrival_unix"].iloc[0])
                    opens = [t for t in tr if t["to"] == "open" and t["unix_s"] >= t0]
                    info = {"n_transitions": len(tr)}
                    if opens:
                        info["outage_detect_s"] = round(opens[0]["unix_s"] - t0, 2)
                        closes = [t for t in tr if t["to"] == "closed" and t["unix_s"] > opens[0]["unix_s"]]
                        if closes:
                            info["outage_recover_s"] = round(closes[0]["unix_s"] - opens[0]["unix_s"], 2)
                    info["final_state"] = tr[-1]["to"] if tr else "closed"
                    breaker[spec["key"]] = info
            summaries[spec["arm"]][spec["seed"]] = s
        block_out = {"per_seed": {arm: {str(k): v for k, v in d.items()} for arm, d in summaries.items()}}
        block_out["v4_vs_v2"] = _seed_bootstrap(data["v4"], data["v2"])
        block_out["v4_vs_ungoverned"] = _seed_bootstrap(data["v4"], data["ungoverned"])
        pooled = {}
        for arm in POLICY_NAME:
            viol = np.concatenate([d["viol"] for d in data[arm].values()])
            ttft = np.concatenate([d["ttft"] for d in data[arm].values()])
            spend = sum(s["spend_usd"] for s in summaries[arm].values())
            failed = sum(s["failed"] for s in summaries[arm].values())
            met = len(viol) * (1 - viol.mean())
            wh = sum((s["energy"].get("gpu_energy_wh") or 0.0) for s in summaries[arm].values())
            pooled[arm] = {"violation_rate": float(viol.mean()), "p95_ttft_ms": float(np.percentile(ttft, 95)),
                           "spend_usd": spend, "failed": failed,
                           "usd_per_slo_met_x1e4": spend / met * 1e4 if met else None,
                           "wh_per_slo_met": wh / met if met else None,
                           "cloud_share": float(np.mean([s["cloud_share"] for s in summaries[arm].values()])),
                           "quota_compliant_runs": sum(bool(s["quota"]["compliant"]) for s in summaries[arm].values()),
                           "slo_guarantee_runs": sum(s["violation_rate"] <= 0.10 for s in summaries[arm].values()),
                           "violation_ci95_over_seeds": _pooled_ci(data[arm], lambda ds: float(np.concatenate([d["viol"] for d in ds]).mean()))}
        block_out["pooled"] = pooled
        if breaker:
            block_out["breaker"] = breaker
        result[block] = block_out

    for exp, base_dir, ext_dir, trace in (("B5", "B", "B_ext", "bursty"), ("C5", "C", "C_ext", "transition")):
        out = {}
        for pol in ("guarded_governor", "tail_aware"):
            per_seed = {}
            for d in (OUT / "runs" / base_dir, OUT / "runs" / ext_dir):
                for path in sorted(d.glob(f"*_{pol}.jsonl")):
                    df, manifest, _ = al.load_run(path)
                    per_seed[int(manifest["spec"]["seed"])] = df
            out[pol] = {
                "seeds": sorted(per_seed),
                "violation": _pooled_ci(per_seed, lambda dfs: float(np.concatenate([df["violation"].values for df in dfs]).mean())),
                "failed_per_run": _pooled_ci(per_seed, lambda dfs: float(np.mean([(~df["ok"]).sum() for df in dfs]))),
                "wasted_usd_per_run": _pooled_ci(per_seed, lambda dfs: float(np.mean([
                    df.loc[(df["executed_on"] == "cloud") & (df["violation"] == 1), "cost_usd"].sum() for df in dfs]))),
            }
            if exp == "C5":
                out[pol]["degraded_phase_violation"] = _pooled_ci(per_seed, lambda dfs: float(np.concatenate([
                    df.loc[df["phase"] == "cloud_degradation", "violation"].values for df in dfs]).mean()))
        result[exp] = out

    # V4D: v4 under drift vs v2 and ungoverned from C + C_ext (paired by seed)
    drift = {}
    sources = {"v4": [OUT / "runs" / "V4D"], "v2": [OUT / "runs" / "C", OUT / "runs" / "C_ext"],
               "ungoverned": [OUT / "runs" / "C", OUT / "runs" / "C_ext"]}
    by_arm = {}
    for arm, dirs in sources.items():
        per_seed, mech = {}, {}
        for d in dirs:
            for path in sorted(d.glob(f"*_transition_{POLICY_NAME[arm]}.jsonl")):
                if not path.with_suffix(".manifest.json").exists():
                    continue
                df, manifest, _ = al.load_run(path)
                seed = int(manifest["spec"]["seed"])
                per_seed[seed] = df
                if arm != "ungoverned":
                    deg = df.loc[df["phase"] == "cloud_degradation", "arrival_unix"]
                    mech[seed] = governed_mechanism(df, manifest, float(deg.min()) if len(deg) else None)
        by_arm[arm] = per_seed
        if not per_seed:
            continue
        drift[arm] = {
            "seeds": sorted(per_seed),
            "violation": _pooled_ci(per_seed, lambda dfs: float(np.concatenate([x["violation"].values for x in dfs]).mean())),
            "degraded_phase_violation": _pooled_ci(per_seed, lambda dfs: float(np.concatenate([
                x.loc[x["phase"] == "cloud_degradation", "violation"].values for x in dfs]).mean())),
            "per_seed": {str(s): {"violation": float(x["violation"].mean()),
                                  "degraded_violation": float(x.loc[x["phase"] == "cloud_degradation", "violation"].mean()),
                                  "cloud_share": float((x["route"] == "cloud").mean()),
                                  "failed": int((~x["ok"]).sum())} for s, x in per_seed.items()},
        }
        if mech:
            drift[arm]["mechanism"] = {str(s): m for s, m in mech.items()}
    for other in ("v2", "ungoverned"):
        common = sorted(set(by_arm.get("v4", {})) & set(by_arm.get(other, {})))
        if common:
            diffs = [100 * (by_arm["v4"][s]["violation"].mean() - by_arm[other][s]["violation"].mean()) for s in common]
            drift[f"v4_minus_{other}_pp_per_seed"] = dict(zip(map(str, common), diffs))
            drift[f"v4_minus_{other}_sign_flip_p"] = _sign_flip_p(diffs)
    result["V4D"] = drift

    (OUT / "analysis").mkdir(parents=True, exist_ok=True)
    (OUT / "analysis" / "live_V4.json").write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    n = result["V4N"]["v4_vs_v2"]
    print("PRIMARY v4 vs v2 (nominal):", {k: v for k, v in n.items() if k != "per_seed_diff_pp"})
    print("per-seed diff pp:", n["per_seed_diff_pp"])
    for block in ("V4N", "V4F"):
        for arm, p in result[block]["pooled"].items():
            print(block, arm, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in p.items() if k != "violation_ci95_over_seeds"},
                  "viol CI", p["violation_ci95_over_seeds"]["ci95"])
    print("V4F v4_vs_v2:", {k: result["V4F"]["v4_vs_v2"][k] for k in ("violation_diff_pp", "violation_diff_ci95", "p95_ratio", "p95_ratio_ci95")})
    print("B5:", json.dumps(result["B5"])[:600])
    print("C5:", json.dumps(result["C5"])[:600])
    print("V4D:", json.dumps(result["V4D"], default=str)[:1200])
    if missing:
        print("MISSING RUNS:", missing)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["artifact", "freeze", "dryrun", "batch", "analyze"])
    args = ap.parse_args()
    if args.stage == "artifact":
        stage_artifact()
    elif args.stage == "freeze":
        stage_freeze()
    elif args.stage == "dryrun":
        asyncio.run(stage_dryrun())
    elif args.stage == "batch":
        sys.exit(asyncio.run(stage_batch()))
    else:
        stage_analyze()


if __name__ == "__main__":
    main()
