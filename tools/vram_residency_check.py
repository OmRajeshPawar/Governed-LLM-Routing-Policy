from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from apex.core import load_config
from apex.providers import OllamaProvider
from apex.providers import VramGuardError, assert_ollama_vram_resident, nvidia_snapshot


def mb(value: int | float) -> float:
    return float(value) / (1024.0 * 1024.0)


async def main_async(args: argparse.Namespace) -> None:
    cfg = load_config()
    model = args.model or cfg.providers.local.model
    keep_alive = cfg.providers.local.keep_alive
    num_gpu = cfg.providers.local.num_gpu
    num_ctx = args.num_ctx if args.num_ctx is not None else cfg.providers.local.num_ctx

    provider = OllamaProvider(
        "local",
        model,
        keep_alive=keep_alive,
        num_gpu=num_gpu,
        temperature=cfg.generation.temperature,
        num_ctx=num_ctx,
    )

    before = psutil.virtual_memory().used
    t0 = time.perf_counter()
    load_s = await provider.warmup()
    entry = assert_ollama_vram_resident(
        model,
        min_gpu_percent=args.min_gpu_percent,
        max_loaded_models=args.max_loaded_models,
    )
    snap_after_warm = nvidia_snapshot()

    ram_samples = [before, psutil.virtual_memory().used]
    for _ in range(args.samples):
        res = await provider.generate(args.prompt, args.max_tokens)
        if not res.ok:
            raise SystemExit(f"Local model generation failed during VRAM smoke: {res.error}")
        assert_ollama_vram_resident(
            model,
            min_gpu_percent=args.min_gpu_percent,
            max_loaded_models=args.max_loaded_models,
        )
        ram_samples.append(psutil.virtual_memory().used)

    after = psutil.virtual_memory().used
    ram_delta_mb = mb(after - before)
    persistent_delta_mb = mb(min(ram_samples[-2:]) - before) if len(ram_samples) >= 2 else ram_delta_mb
    if persistent_delta_mb > args.max_persistent_ram_delta_mb:
        raise SystemExit(
            "Persistent host RAM increased too much during local model smoke: "
            f"{persistent_delta_mb:.1f} MB > {args.max_persistent_ram_delta_mb:.1f} MB"
        )

    payload = {
        "ok": True,
        "model": model,
        "num_ctx": num_ctx,
        "ollama_processor": entry.processor,
        "warm_load_s": round(load_s, 2),
        "wall_s": round(time.perf_counter() - t0, 2),
        "ram_delta_mb": round(ram_delta_mb, 1),
        "persistent_ram_delta_mb": round(persistent_delta_mb, 1),
        "nvidia": snap_after_warm.__dict__ if snap_after_warm else None,
    }
    print(json.dumps(payload, indent=2))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--num-ctx", type=int, default=None)
    ap.add_argument("--min-gpu-percent", type=int, default=99)
    ap.add_argument("--max-loaded-models", type=int, default=1)
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--max-persistent-ram-delta-mb", type=float, default=1024.0)
    ap.add_argument("--prompt", default="Return the word ok.")
    return ap.parse_args()


def main() -> None:
    try:
        asyncio.run(main_async(parse_args()))
    except VramGuardError as exc:
        raise SystemExit(f"VRAM residency check failed: {exc}") from exc


if __name__ == "__main__":
    main()
