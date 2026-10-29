from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time


SEEDS = [42, 7, 13, 23, 101]
MIXES = ["synthetic", "real_sample", "reasoning_coding"]
PATTERN = "bursty"


def cooldown(threshold_c: int = 62, max_wait_s: int = 180) -> None:
    if not shutil.which("nvidia-smi"):
        return
    waited = 0
    while waited < max_wait_s:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                text=True,
            ).strip().splitlines()[0]
            temp = int(out)
        except Exception:
            return
        if temp < threshold_c:
            print(f"[cooldown] GPU {temp}C -> proceeding")
            return
        print(f"[cooldown] GPU {temp}C >= {threshold_c}C, waiting...")
        time.sleep(12)
        waited += 12


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-seed", type=int, default=300)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    args = ap.parse_args()

    cmds = []
    for mix in MIXES:
        for seed in SEEDS:
            out_name = f"data/events/shadow_{mix}_{PATTERN}_n{args.n_per_seed}_s{seed}.jsonl"
            cmds.append((out_name, [
                sys.executable, "-m", "apex.runner",
                "--mode", "shadow",
                "--pattern", PATTERN,
                "--mix", mix,
                "--n", str(args.n_per_seed),
                "--seed", str(seed),
            ]))

    print(f"Planned {len(cmds)} shadow runs; total requests = {len(cmds) * args.n_per_seed}.")
    for _, cmd in cmds:
        print(" ".join(cmd))

    if args.dry_run:
        return

    for out_name, cmd in cmds:
        if args.skip_existing and __import__("pathlib").Path(out_name).exists():
            print(f"[skip] {out_name}")
            continue
        cooldown()
        subprocess.run(cmd, check=True)

    subprocess.run([sys.executable, "ml/train.py"], check=True)
    subprocess.run([sys.executable, "analysis/evaluate.py"], check=True)
    subprocess.run([sys.executable, "replay/queue_replay.py"], check=True)
    subprocess.run([sys.executable, "showcase/build_showcase.py"], check=True)


if __name__ == "__main__":
    main()
