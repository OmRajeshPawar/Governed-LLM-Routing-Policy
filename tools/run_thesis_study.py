from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS_ALL = [42, 7, 13, 23, 101]
MIXES = ["synthetic", "real_sample", "reasoning_coding"]
PROFILE_N = {"smoke": 20, "minimum": 100, "target": 300, "stretch": 600}


def run(cmd: list[str], title: str, *, env: dict[str, str] | None = None) -> None:
    print(f"\n{'=' * 78}\n>>> {title}\n{'=' * 78}")
    subprocess.run([sys.executable, *cmd], cwd=ROOT, check=True, env=env)


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


def sanitize_variant(value: str | None) -> str | None:
    if not value:
        return None
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._-")
    return safe or None


def shadow_run_name(mix: str, n: int, seed: int, variant: str | None) -> str:
    prefix = f"shadow_{variant}_" if variant else "shadow_"
    return f"{prefix}{mix}_bursty_n{n}_s{seed}"


def shadow_exists(mix: str, n: int, seed: int, variant: str | None) -> bool:
    path = ROOT / "data" / "events" / f"{shadow_run_name(mix, n, seed, variant)}.jsonl"
    return path.exists() and path.stat().st_size > 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=PROFILE_N, default="minimum")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--with-vertex", action="store_true")
    ap.add_argument("--vertex-n", type=int, default=100)
    ap.add_argument("--max-vertex-cost-usd", type=float, default=10.0)
    ap.add_argument("--variant", default=None,
                    help="Optional event namespace, e.g. main_llama31_8b or accelerated_local.")
    ap.add_argument("--local-model", default=None,
                    help="Override providers.local.model for sensitivity runs.")
    ap.add_argument("--strict-vram", action="store_true",
                    help="Run the fail-closed VRAM residency gate before measured GPU runs.")
    ap.add_argument("--num-ctx", type=int, default=None,
                    help="Override Ollama num_ctx for the VRAM gate and local runner config.")
    args = ap.parse_args()

    n = PROFILE_N[args.profile]
    seeds = [42] if args.profile == "smoke" else SEEDS_ALL
    variant = sanitize_variant(args.variant)
    shadow_glob = (
        f"shadow_{variant}_*_bursty_n{n}_s*.jsonl"
        if variant
        else f"shadow_*_bursty_n{n}_s*.jsonl"
    )
    shadow_files = [
        str(ROOT / "data" / "events" / f"{shadow_run_name(mix, n, seed, variant)}.jsonl")
        for mix in MIXES
        for seed in seeds
    ]
    study_env = dict(os.environ)
    study_env["APEX_SHADOW_GLOB"] = shadow_glob
    study_env["APEX_SHADOW_FILES"] = ";".join(shadow_files)

    run(["tools/secret_hygiene.py"], "0/7 secret hygiene scan")
    run(["-m", "pytest", "-q"], "1/7 regression tests")
    if args.strict_vram:
        check_cmd = ["tools/vram_residency_check.py"]
        if args.local_model:
            check_cmd += ["--model", args.local_model]
        if args.num_ctx is not None:
            check_cmd += ["--num-ctx", str(args.num_ctx)]
        run(check_cmd, "1b/7 VRAM residency gate")

    total = len(MIXES) * len(seeds) * n
    print(f"Planned shadow collection: {len(MIXES)} mixes x {len(seeds)} seeds x {n} = {total} requests")
    print(f"Variant: {variant or '<none>'}")
    print(f"Analysis shadow glob: {shadow_glob}")
    for mix in MIXES:
        for seed in seeds:
            if args.skip_existing and shadow_exists(mix, n, seed, variant):
                print(f"[skip] existing {shadow_run_name(mix, n, seed, variant)}.jsonl")
                continue
            cooldown()
            if args.strict_vram:
                check_cmd = ["tools/vram_residency_check.py"]
                if args.local_model:
                    check_cmd += ["--model", args.local_model]
                if args.num_ctx is not None:
                    check_cmd += ["--num-ctx", str(args.num_ctx)]
                run(check_cmd, f"2/7 VRAM gate before shadow mix={mix} seed={seed}")
            runner_cmd = [
                "-m",
                "apex.runner",
                "--mode",
                "shadow",
                "--pattern",
                "bursty",
                "--mix",
                mix,
                "--n",
                str(n),
                "--seed",
                str(seed),
            ]
            if variant:
                runner_cmd += ["--variant", variant]
            if args.local_model:
                runner_cmd += ["--local-model", args.local_model]
            if args.num_ctx is not None:
                runner_cmd += ["--num-ctx", str(args.num_ctx)]
            run(
                runner_cmd,
                f"2/7 shadow run mix={mix} seed={seed} n={n}",
            )

    run(["ml/train.py"], "3/7 train risk models and tune telemetry baseline", env=study_env)
    run(["analysis/evaluate.py"], "4/7 static frontier, oracle-gap, heatmap, and CDF figures", env=study_env)
    run(["replay/queue_replay.py"], "5/7 counterfactual queue replay and oracle headroom", env=study_env)
    cooldown()
    run(["replay/paired_audit.py"], "6/7 live paired occupancy-harm audit")

    if args.with_vertex:
        run(
            [
                "tools/vertex_smoke.py",
                "--n",
                str(args.vertex_n),
                "--max-estimated-cost-usd",
                str(args.max_vertex_cost_usd),
            ],
            "7/7 bounded Gemini 2.5 Flash validation",
        )
    else:
        print("\n[vertex] skipped. Add --with-vertex after rotating/checking the key if live validation is desired.")

    run(["showcase/build_showcase.py"], "final dashboard rebuild")
    print("\nThesis study pipeline finished.")


if __name__ == "__main__":
    main()
