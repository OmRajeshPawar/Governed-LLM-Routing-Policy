from __future__ import annotations

import asyncio
import csv
import math
import os
import random
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, asdict

from ollama import AsyncClient

from .core import CloudProviderCfg


@dataclass
class ProviderResult:
    role: str
    model: str
    ok: bool
    error: str | None
    t_provider_start: float
    t_first_text: float | None
    t_completed: float | None
    provider_ttft_ms: float | None
    service_ms: float | None
    prompt_tokens: int
    output_tokens: int
    cost_usd: float
    text_len_chars: int

    def to_dict(self) -> dict:
        return asdict(self)


class OllamaProvider:

    def __init__(self, role: str, model: str, keep_alive: int = -1,
                 num_gpu: int | None = None, temperature: float = 0.0,
                 cost_per_1k_tokens: float = 0.0, extra_latency_ms: float = 0.0,
                 num_ctx: int | None = None):
        self.role = role
        self.model = model
        self.keep_alive = keep_alive
        self.num_gpu = num_gpu
        self.num_ctx = num_ctx
        self.temperature = temperature
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.extra_latency_ms = extra_latency_ms
        self._client = AsyncClient()

    def _options(self, max_tokens: int) -> dict:
        opts: dict = {"temperature": self.temperature, "num_predict": max_tokens}
        if self.num_gpu is not None:
            opts["num_gpu"] = self.num_gpu
        if self.num_ctx is not None:
            opts["num_ctx"] = self.num_ctx
        return opts

    async def warmup(self) -> float:
        t0 = time.perf_counter()
        try:
            await self._client.chat(
                model=self.model, messages=[{"role": "user", "content": "ok"}],
                stream=False, think=False, keep_alive=self.keep_alive,
                options=self._options(1))
        except Exception:
            pass
        return time.perf_counter() - t0

    async def generate(self, prompt: str, max_tokens: int) -> ProviderResult:
        if self.extra_latency_ms:
            await asyncio.sleep(self.extra_latency_ms / 1000.0)
        t_start = time.perf_counter()
        t_first: float | None = None
        chars = prompt_tokens = output_tokens = 0
        try:
            stream = await self._client.chat(
                model=self.model, messages=[{"role": "user", "content": prompt}],
                stream=True, think=False, keep_alive=self.keep_alive,
                options=self._options(max_tokens))
            async for chunk in stream:
                piece = chunk.message.content or (getattr(chunk.message, "thinking", None) or "")
                if piece:
                    if t_first is None:
                        t_first = time.perf_counter()
                    chars += len(piece)
                if chunk.done:
                    prompt_tokens = int(chunk.prompt_eval_count or 0)
                    output_tokens = int(chunk.eval_count or 0)
            t_done = time.perf_counter()
        except Exception as exc:
            t_done = time.perf_counter()
            return ProviderResult(self.role, self.model, False, str(exc), t_start,
                                  t_first, t_done, None, (t_done - t_start) * 1000.0,
                                  0, 0, 0.0, chars)
        cost = (prompt_tokens + output_tokens) / 1000.0 * self.cost_per_1k_tokens
        return ProviderResult(
            self.role, self.model, True, None, t_start, t_first, t_done,
            ((t_first - t_start) * 1000.0) if t_first else None,
            (t_done - t_start) * 1000.0, prompt_tokens, output_tokens, cost, chars)


class CalibratedCloudProvider:

    def __init__(self, cfg: CloudProviderCfg, base_ttft_ms: float,
                 per_token_ms: float, seed: int = 42):
        self.role = "cloud"
        self.model = f"calibrated[{cfg.calibration_model}]"
        self.cfg = cfg
        self.base_ttft_ms = base_ttft_ms
        self.per_token_ms = per_token_ms
        self._rng = random.Random(seed)

    async def warmup(self) -> float:
        return 0.0

    async def generate(self, prompt: str, max_tokens: int) -> ProviderResult:
        c = self.cfg
        est_tokens = float(max_tokens)
        jitter = math.exp(self._rng.gauss(0.0, c.jitter_sigma))
        ttft_ms = (c.network_rtt_ms + self.base_ttft_ms * c.speed_ratio) * jitter
        service_ms = ttft_ms + self.per_token_ms * c.speed_ratio * est_tokens

        t_start = time.perf_counter()
        t_first = t_start + ttft_ms / 1000.0
        t_done = t_start + service_ms / 1000.0
        prompt_tokens = max(1, len(prompt) // 4)
        output_tokens = int(est_tokens)
        cost = (prompt_tokens + output_tokens) / 1000.0 * c.cost_per_1k_tokens
        return ProviderResult("cloud", self.model, True, None, t_start, t_first,
                              t_done, ttft_ms, service_ms, prompt_tokens,
                              output_tokens, cost, 0)


async def calibrate_cloud(model: str, num_gpu: int | None = 0,
                          temperature: float = 0.0,
                          num_ctx: int | None = None) -> dict:
    import numpy as np
    prov = OllamaProvider("calib", model, num_gpu=num_gpu, temperature=temperature, num_ctx=num_ctx)
    await prov.warmup()
    sizes = [16, 48, 96, 160, 256]
    ttfts, svc, toks = [], [], []
    for n in sizes:
        for _ in range(2):
            r = await prov.generate("Explain caching briefly.", n)
            if r.ok and r.provider_ttft_ms:
                ttfts.append(r.provider_ttft_ms)
                svc.append(r.service_ms)
                toks.append(max(1, r.output_tokens))
    if len(ttfts) < 2:
        return {"calibration_model": model, "base_ttft_ms": 200.0,
                "per_token_ms": 5.0, "jitter_sigma": 0.15, "n_samples": len(ttfts),
                "fallback": True}
    base_ttft_ms = float(np.median(ttfts))
    decode = np.array(svc) - np.array(ttfts)
    slope = float(np.polyfit(np.array(toks, dtype=float), decode, 1)[0])
    per_token_ms = max(0.0, slope)
    jitter_sigma = float(np.std(np.log(np.array(ttfts) / base_ttft_ms)))
    return {
        "calibration_model": model, "base_ttft_ms": round(base_ttft_ms, 2),
        "per_token_ms": round(per_token_ms, 4), "jitter_sigma": round(jitter_sigma, 4),
        "n_samples": len(ttfts), "fallback": False,
    }


class VramGuardError(RuntimeError):
    pass


@dataclass(frozen=True)
class OllamaPsEntry:
    name: str
    processor: str
    gpu_percent: int | None
    has_cpu_split: bool


@dataclass(frozen=True)
class NvidiaSnapshot:
    name: str
    memory_total_mb: int
    memory_used_mb: int
    utilization_gpu_percent: int
    temperature_c: int


def _run_text(cmd: list[str], timeout_s: int = 15) -> str:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, timeout=timeout_s)
    except FileNotFoundError as exc:
        raise VramGuardError(f"Missing command: {cmd[0]}") from exc
    except subprocess.CalledProcessError as exc:
        raise VramGuardError(f"{' '.join(cmd)} failed: {exc.output.strip()}") from exc
    except subprocess.TimeoutExpired as exc:
        raise VramGuardError(f"{' '.join(cmd)} timed out after {timeout_s}s") from exc


def ollama_exe() -> str:
    explicit = os.getenv("APEX_OLLAMA_EXE")
    if explicit:
        return explicit
    found = shutil.which("ollama")
    if found:
        return found
    candidates = [
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
        os.path.expandvars(r"%ProgramFiles%\Ollama\ollama.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return "ollama"


def parse_ollama_ps(text: str) -> list[OllamaPsEntry]:
    entries: list[OllamaPsEntry] = []
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) <= 1:
        return entries

    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        name = parts[0]
        until_idx = next((i for i, token in enumerate(parts) if token.upper() == "UNTIL"), len(parts))
        processor_tokens = parts[4:until_idx] if len(parts) >= 5 else parts[1:]
        processor = " ".join(processor_tokens)
        gpu_match = re.search(r"(\d+)\s*%\s*GPU", processor, flags=re.IGNORECASE)
        gpu_percent = int(gpu_match.group(1)) if gpu_match else None
        has_cpu_split = bool(re.search(r"\bCPU\b", processor, flags=re.IGNORECASE))
        entries.append(OllamaPsEntry(name, processor, gpu_percent, has_cpu_split))
    return entries


def find_model_entry(entries: list[OllamaPsEntry], model: str) -> OllamaPsEntry | None:
    target = model.lower()
    for entry in entries:
        if entry.name.lower() == target:
            return entry
    for entry in entries:
        if target in entry.name.lower() or entry.name.lower() in target:
            return entry
    return None


def assert_ollama_vram_resident(
    model: str,
    *,
    ps_text: str | None = None,
    min_gpu_percent: int = 99,
    max_loaded_models: int = 1,
) -> OllamaPsEntry:
    text = ps_text if ps_text is not None else _run_text([ollama_exe(), "ps"])
    entries = parse_ollama_ps(text)
    if not entries:
        raise VramGuardError("No loaded Ollama models were reported by `ollama ps`.")
    if len(entries) > max_loaded_models:
        loaded = ", ".join(entry.name for entry in entries)
        raise VramGuardError(f"Expected at most {max_loaded_models} loaded model(s), found: {loaded}")
    entry = find_model_entry(entries, model)
    if entry is None:
        loaded = ", ".join(entry.name for entry in entries)
        raise VramGuardError(f"Model {model!r} is not loaded according to `ollama ps`; loaded: {loaded}")
    if entry.has_cpu_split:
        raise VramGuardError(f"Model {entry.name!r} is split with CPU/system RAM: {entry.processor!r}")
    if entry.gpu_percent is None or entry.gpu_percent < min_gpu_percent:
        raise VramGuardError(
            f"Model {entry.name!r} is not fully GPU-resident: processor={entry.processor!r}"
        )
    return entry


def nvidia_snapshot() -> NvidiaSnapshot | None:
    if not shutil.which("nvidia-smi"):
        return None
    out = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    row = next(csv.reader([out.strip().splitlines()[0]]))
    return NvidiaSnapshot(
        name=row[0].strip(),
        memory_total_mb=int(row[1]),
        memory_used_mb=int(row[2]),
        utilization_gpu_percent=int(row[3]),
        temperature_c=int(row[4]),
    )
