from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], title: str, env: dict | None = None):
    print(f"\n{'='*70}\n>>> {title}\n{'='*70}")
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT,
                       env={**os.environ, **env} if env else None)
    if r.returncode != 0:
        raise SystemExit(f"Step failed: {title}")


def cooldown(threshold_c: int = 58, max_wait_s: int = 150):
    import shutil
    if not shutil.which("nvidia-smi"):
        return
    waited = 0
    while waited < max_wait_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu",
                 "--format=csv,noheader,nounits"], text=True).strip().splitlines()[0]
            temp = int(out)
        except Exception:
            return
        if temp < threshold_c:
            print(f"  [cooldown] GPU {temp}C -> proceeding")
            return
        print(f"  [cooldown] GPU {temp}C >= {threshold_c}C, waiting...")
        time.sleep(12)
        waited += 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller trace for a fast demo")
    args = ap.parse_args()
    n = 32 if args.quick else 64

    run(["-m", "apex.runner", "--mode", "shadow", "--pattern", "bursty",
         "--n", str(n), "--seed", "42"], f"1/4  Shadow data collection (seed 42, n={n})")
    cooldown()
    run(["-m", "apex.runner", "--mode", "shadow", "--pattern", "bursty",
         "--n", str(n), "--seed", "7"], f"1/4  Shadow data collection (seed 7, n={n})")
    corpus = "shadow_*_bursty_n300_s*.jsonl"
    demo_out = "analysis/demo_run"
    run(["ml/train.py"], "2/4  Train + calibrate dual-risk models; tune the heuristic",
        env={"APEX_SHADOW_GLOB": corpus})
    run(["analysis/evaluate.py"], f"3/4  Evaluate: frontier, RQ0 verdict, figures -> {demo_out}/",
        env={"APEX_SHADOW_GLOB": corpus, "APEX_ANALYSIS_OUT": demo_out})
    cooldown()
    run(["replay/paired_audit.py"], "4/4  Live paired occupancy-harm audit")

    print(f"\n{'='*70}\nDONE. Pipeline output: {demo_out}/")
    print(f"Steps 2-3 ran against the frozen corpus ({corpus}), so the table")
    print(f"in {demo_out}/ should reproduce analysis/results_table.md exactly.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
