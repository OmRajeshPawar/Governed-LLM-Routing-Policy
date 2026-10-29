"""Run-level (block) bootstrap for the headline policy comparison.

Reuses analysis/evaluate.py functions read-only; writes nothing into the repo.
"""
import os
import sys
import json

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["APEX_SHADOW_GLOB"] = "shadow_*_bursty_n300_s*.jsonl"
os.chdir(REPO)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "analysis"))

import numpy as np
import pandas as pd
from pathlib import Path

import evaluate as ev
from apex.core import load_config, _repo_root
from apex.events import load_events, resolve_event_files
from ml.risk_model import CalibratedRiskModel

cfg = load_config()
events_dir = _repo_root() / cfg.paths.events_dir
files, src = resolve_event_files(events_dir)
df = pd.concat([load_events(f) for f in files], ignore_index=True)
print("rows", len(df), "files", len(files))
print("run/mix-like columns:", [c for c in df.columns if "run" in c.lower() or "mix" in c.lower() or "seed" in c.lower() or "workload" in c.lower()])

meta = json.loads((_repo_root() / cfg.paths.models_dir / "meta.json").read_text())
fr = meta["frozen_thresholds"]
sla = fr["sla_ttft_ms"]
cfg.targets.sla_ttft_ms = fr["sla_ttft_ms"]
cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]
ml = CalibratedRiskModel.load(_repo_root() / cfg.paths.models_dir)
local_ttft = df["end_to_end_ttft_ms"].values.astype(float)
base, per_tok = ev.meta_calib(cfg, meta)
scores = ev.policy_scores(df, cfg, ml, meta.get("heuristic_weights"))

run_col = "run_id"
runs = df[run_col].astype(str).values
mix_col = next((c for c in df.columns if "mix" in c.lower()), None)
if mix_col is None:
    def mix_of(r):
        for m in ("reasoning_coding", "real_sample", "synthetic"):
            if m in r:
                return m
        return "unknown"
    mixes = np.array([mix_of(r) for r in runs])
else:
    mixes = df[mix_col].astype(str).values
print("mixes:", pd.Series(mixes).value_counts().to_dict())
print("runs:", len(set(runs)))

N = len(df)


def viol_vec(name, budget, cloud_ttft):
    k = int(round(budget * N))
    order = np.argsort(-scores[name])
    off = np.zeros(N, dtype=bool)
    off[order[:k]] = True
    return (np.where(off, cloud_ttft, local_ttft) > sla).astype(int)


run_ids = sorted(set(runs))
run_index = {r: np.where(runs == r)[0] for r in run_ids}
run_mix = {r: mixes[run_index[r][0]] for r in run_ids}
by_mix = {}
for r in run_ids:
    by_mix.setdefault(run_mix[r], []).append(r)


def ci(h, m, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = h - m
    req = [d[rng.integers(0, N, N)].mean() for _ in range(B)]
    blk = []
    for _ in range(B):
        idx = []
        for mix, rs in by_mix.items():
            pick = rng.choice(len(rs), size=len(rs), replace=True)
            idx.extend(run_index[rs[p]] for p in pick)
        idx = np.concatenate(idx)
        blk.append(d[idx].mean())
    return d.mean(), np.percentile(req, [2.5, 97.5]), np.percentile(blk, [2.5, 97.5])


out = {"budgets": {}, "speed_ratio_at_30": {}}
cloud_R1, _ = ev.build_cloud_outcomes(df, cfg, base, per_tok, 1.0)
for b in (0.1, 0.2, 0.3, 0.4, 0.5):
    h = viol_vec("tail_aware", b, cloud_R1)
    m = viol_vec("ml_dual_risk", b, cloud_R1)
    diff, rq, bk = ci(h, m)
    out["budgets"][b] = dict(heur=h.mean(), ml=m.mean(), diff=diff, req_ci=list(rq), block_ci=list(bk))
    print(f"b={b:.1f} heur={h.mean()*100:.2f} ml={m.mean()*100:.2f} diff={diff*100:+.2f} "
          f"reqCI=[{rq[0]*100:+.2f},{rq[1]*100:+.2f}] blockCI=[{bk[0]*100:+.2f},{bk[1]*100:+.2f}]")

for R in (0.5, 1.0, 2.0):
    cl, _ = ev.build_cloud_outcomes(df, cfg, base, per_tok, R)
    h = viol_vec("tail_aware", 0.3, cl)
    m = viol_vec("ml_dual_risk", 0.3, cl)
    diff, rq, bk = ci(h, m)
    out["speed_ratio_at_30"][R] = dict(heur=h.mean(), ml=m.mean(), diff=diff, req_ci=list(rq), block_ci=list(bk))
    print(f"R={R} heur={h.mean()*100:.2f} ml={m.mean()*100:.2f} diff={diff*100:+.2f} "
          f"reqCI=[{rq[0]*100:+.2f},{rq[1]*100:+.2f}] blockCI=[{bk[0]*100:+.2f},{bk[1]*100:+.2f}]")

# ML-TTFT vs heuristic at 30% too
h = viol_vec("tail_aware", 0.3, cloud_R1)
m2 = viol_vec("ml_ttft", 0.3, cloud_R1)
diff, rq, bk = ci(h, m2)
print(f"ml_ttft @30: heur={h.mean()*100:.2f} ml_ttft={m2.mean()*100:.2f} diff={diff*100:+.2f} "
      f"blockCI=[{bk[0]*100:+.2f},{bk[1]*100:+.2f}]")

Path(os.environ.get("BB_OUT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "block_boot.json"))).write_text(json.dumps(out, indent=1, default=float))
