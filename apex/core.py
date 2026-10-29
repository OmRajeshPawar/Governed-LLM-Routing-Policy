from __future__ import annotations

import asyncio
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import psutil
import yaml
from pydantic import BaseModel, Field

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML = True
    _HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _NVML = False
    _HANDLE = None


class ProviderCfg(BaseModel):
    model: str
    keep_alive: int = -1
    num_gpu: int | None = None
    num_ctx: int | None = None
    cost_per_1k_tokens: float = 0.0
    extra_latency_ms: float = 0.0


class CloudProviderCfg(BaseModel):
    mode: str = "calibrated"
    calibration_model: str = "gemma4:e2b"
    base_ttft_ms: float | None = None
    per_token_ms: float | None = None
    network_rtt_ms: float = 60.0
    jitter_sigma: float = 0.15
    speed_ratio: float = 1.0
    cost_per_1k_tokens: float = 0.50
    est_output_tokens_source: str = "ingress_only"
    keep_alive: int = -1
    num_gpu: int | None = 0
    num_ctx: int | None = None


class ProvidersCfg(BaseModel):
    local: ProviderCfg
    cloud: CloudProviderCfg


class GenerationCfg(BaseModel):
    temperature: float = 0.0
    default_max_tokens: int = 256


class TargetsCfg(BaseModel):
    sla_ttft_ms: float = 2000.0
    occupancy_service_ms: float = 9000.0


class RollingCfg(BaseModel):
    window: int = 50
    min_samples: int = 8
    ewma_alpha: float = 0.3


class PoliciesCfg(BaseModel):
    queue_threshold: int = 3
    tail_aware_pressure_threshold: float = 1.0
    ml_ttft_threshold: float = 0.5
    ml_occupancy_threshold: float = 0.5
    semantic_embedding_threshold: float = 1.0
    local_llm_probe_threshold: float = 0.75
    thinking_probe_threshold: float = 0.85


class BudgetCfg(BaseModel):
    mode: str = "evaluation_hard"
    target_fraction: float = 0.30


class CircuitBreakerCfg(BaseModel):
    consecutive_failure_limit: int = 3
    failure_window: int = 20
    failure_window_limit: int = 5
    latency_success_minimum: int = 10
    latency_window: int = 10
    latency_breach_limit: int = 5
    latency_multiplier: float = 2.0
    cooldown_seconds: float = 30.0
    cooldown_arrivals: int = 10
    half_open_successes_to_close: int = 3


class GovernorCfg(BaseModel):
    enabled: bool = False
    artifact_path: str = "ml/artifacts/governor_v1.json"
    hard_local_pending: int = 8
    required_full_telemetry: list[str] = Field(default_factory=lambda: [
        "pending_local", "active_local", "roll_ttft_p95", "roll_service_p95",
        "arr_rate_5s", "burst_ratio", "gpu_util", "vram_used_mb",
    ])
    budget: BudgetCfg = Field(default_factory=BudgetCfg)
    circuit: CircuitBreakerCfg = Field(default_factory=CircuitBreakerCfg)


class PathsCfg(BaseModel):
    data_dir: str = "data"
    events_dir: str = "data/events"
    traces_dir: str = "data/traces"
    models_dir: str = "ml/artifacts"
    figures_dir: str = "analysis/figures"


class Config(BaseModel):
    seed: int = 42
    providers: ProvidersCfg
    local_concurrency: int = 2
    generation: GenerationCfg = Field(default_factory=GenerationCfg)
    targets: TargetsCfg = Field(default_factory=TargetsCfg)
    rolling: RollingCfg = Field(default_factory=RollingCfg)
    policies: PoliciesCfg = Field(default_factory=PoliciesCfg)
    governor: GovernorCfg = Field(default_factory=GovernorCfg)
    paths: PathsCfg = Field(default_factory=PathsCfg)


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_config(path: str | Path | None = None) -> Config:
    cfg_path = Path(path) if path else _repo_root() / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def ensure_dirs(cfg: Config) -> None:
    root = _repo_root()
    for p in (cfg.paths.data_dir, cfg.paths.events_dir, cfg.paths.traces_dir,
              cfg.paths.models_dir, cfg.paths.figures_dir):
        (root / p).mkdir(parents=True, exist_ok=True)


def sample_telemetry() -> dict:
    out = {
        "cpu_util": psutil.cpu_percent(interval=None),
        "ram_util": psutil.virtual_memory().percent,
        "gpu_util": None,
        "vram_used_mb": None,
        "vram_free_mb": None,
        "gpu_temp_c": None,
    }
    if _NVML and _HANDLE is not None:
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(_HANDLE)
            out["gpu_util"] = float(util.gpu)
            out["vram_used_mb"] = mem.used / (1024 ** 2)
            out["vram_free_mb"] = mem.free / (1024 ** 2)
            out["gpu_temp_c"] = float(
                pynvml.nvmlDeviceGetTemperature(_HANDLE, pynvml.NVML_TEMPERATURE_GPU)
            )
        except Exception:
            pass
    return out


class RollingQuantiles:

    def __init__(self, window: int, min_samples: int, ewma_alpha: float):
        self.window = window
        self.min_samples = min_samples
        self.alpha = ewma_alpha
        self.values: deque[float] = deque(maxlen=window)
        self.ewma: float | None = None

    def add(self, x: float) -> None:
        self.values.append(float(x))
        self.ewma = x if self.ewma is None else self.alpha * x + (1 - self.alpha) * self.ewma

    @property
    def count(self) -> int:
        return len(self.values)

    def quantile(self, q: float) -> float | None:
        if self.count < self.min_samples:
            return self.ewma
        return float(np.quantile(self.values, q))

    def p50(self): return self.quantile(0.50)
    def p95(self): return self.quantile(0.95)
    def p99(self): return self.quantile(0.99)


class GatewayState:

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.semaphore = asyncio.Semaphore(cfg.local_concurrency)

        self.pending_local = 0
        self.active_local = 0
        self.active_cloud = 0
        self.recent_timeouts = 0
        self.recent_errors = 0
        self.completed_local = 0

        self.arrivals: deque[float] = deque(maxlen=500)

        r = cfg.rolling
        self.ttft_ms = RollingQuantiles(r.window, r.min_samples, r.ewma_alpha)
        self.service_ms = RollingQuantiles(r.window, r.min_samples, r.ewma_alpha)

        self.cloud_recent_ok = deque(maxlen=50)
        self.cloud_recent_ttft = RollingQuantiles(r.window, r.min_samples, r.ewma_alpha)

    def record_arrival(self, now: float) -> None:
        self.arrivals.append(now)

    def arrival_rate(self, now: float, window_s: float) -> float:
        cutoff = now - window_s
        n = sum(1 for t in self.arrivals if t >= cutoff)
        return n / window_s

    def burst_ratio(self, now: float) -> float:
        short = self.arrival_rate(now, 5.0)
        long = self.arrival_rate(now, 60.0)
        return short / long if long > 0 else 1.0

    def on_local_complete(self, ttft_ms: float | None, service_ms: float | None,
                          ok: bool, timed_out: bool = False) -> None:
        if ok and ttft_ms is not None:
            self.ttft_ms.add(ttft_ms)
        if ok and service_ms is not None:
            self.service_ms.add(service_ms)
        self.completed_local += 1
        if timed_out:
            self.recent_timeouts += 1
        if not ok:
            self.recent_errors += 1

    def on_cloud_complete(self, ttft_ms: float | None, ok: bool) -> None:
        self.cloud_recent_ok.append(1 if ok else 0)
        if ok and ttft_ms is not None:
            self.cloud_recent_ttft.add(ttft_ms)

    def cloud_success_rate(self) -> float:
        return float(np.mean(self.cloud_recent_ok)) if self.cloud_recent_ok else 1.0


@dataclass
class Request:
    request_id: str
    prompt: str
    max_tokens: int
    task_category: str
    t_received: float
    trace_pos: int = 0
    scheduled_arrival_offset_s: float | None = None
    workload_phase: str | None = None
    cloud_degraded: bool = False
    prompt_chars: int = field(init=False)
    prompt_words: int = field(init=False)
    prompt_lines: int = field(init=False)
    prompt_token_est: int = field(init=False)
    context_bucket: int = field(init=False)

    def __post_init__(self):
        self.prompt_chars = len(self.prompt)
        self.prompt_words = len(self.prompt.split())
        self.prompt_lines = self.prompt.count("\n") + 1
        self.prompt_token_est = max(1, self.prompt_chars // 4)
        self.context_bucket = (
            0 if self.prompt_token_est < 512 else
            1 if self.prompt_token_est < 2048 else
            2 if self.prompt_token_est < 8192 else 3
        )


FEATURE_NAMES: list[str] = [
    "prompt_chars", "prompt_words", "prompt_lines", "prompt_token_est",
    "max_output_tokens", "context_bucket", "is_code", "is_json", "is_longform",
    "pending_local", "active_local", "active_cloud", "semaphore_occupancy",
    "recent_timeouts", "recent_errors",
    "arr_rate_1s", "arr_rate_5s", "arr_rate_10s", "arr_rate_60s", "burst_ratio",
    "roll_ttft_p50", "roll_ttft_p95", "roll_ttft_p99",
    "roll_service_p50", "roll_service_p95", "roll_service_p99",
    "cpu_util", "ram_util", "gpu_util", "vram_used_mb", "vram_free_mb", "gpu_temp_c",
    "cloud_success_rate", "cloud_ttft_p95",
]

FEATURE_GROUPS: dict[str, list[str]] = {
    "workload": ["prompt_chars", "prompt_words", "prompt_lines", "prompt_token_est",
                 "max_output_tokens", "context_bucket", "is_code", "is_json", "is_longform"],
    "gateway": ["pending_local", "active_local", "active_cloud", "semaphore_occupancy",
                "recent_timeouts", "recent_errors"],
    "arrival": ["arr_rate_1s", "arr_rate_5s", "arr_rate_10s", "arr_rate_60s", "burst_ratio"],
    "history": ["roll_ttft_p50", "roll_ttft_p95", "roll_ttft_p99",
                "roll_service_p50", "roll_service_p95", "roll_service_p99"],
    "hardware": ["cpu_util", "ram_util", "gpu_util", "vram_used_mb", "vram_free_mb", "gpu_temp_c"],
    "cloud": ["cloud_success_rate", "cloud_ttft_p95"],
}

_JSON_HINT = re.compile(r"\b(json|schema|\{|\})", re.IGNORECASE)
_CODE_HINT = re.compile(r"\b(code|function|python|java|c\+\+|sql|algorithm)\b", re.IGNORECASE)


def build_features(state: GatewayState, req: Request, now: float, telemetry: dict) -> dict:
    nan = float("nan")
    occ = state.cfg.local_concurrency
    feats = {
        "prompt_chars": float(req.prompt_chars),
        "prompt_words": float(req.prompt_words),
        "prompt_lines": float(req.prompt_lines),
        "prompt_token_est": float(req.prompt_token_est),
        "max_output_tokens": float(req.max_tokens),
        "context_bucket": float(req.context_bucket),
        "is_code": 1.0 if req.task_category == "code" or _CODE_HINT.search(req.prompt) else 0.0,
        "is_json": 1.0 if req.task_category == "extract_json" or _JSON_HINT.search(req.prompt) else 0.0,
        "is_longform": 1.0 if req.task_category == "longform" else 0.0,
        "pending_local": float(state.pending_local),
        "active_local": float(state.active_local),
        "active_cloud": float(state.active_cloud),
        "semaphore_occupancy": float(state.active_local) / occ if occ else 0.0,
        "recent_timeouts": float(state.recent_timeouts),
        "recent_errors": float(state.recent_errors),
        "arr_rate_1s": state.arrival_rate(now, 1.0),
        "arr_rate_5s": state.arrival_rate(now, 5.0),
        "arr_rate_10s": state.arrival_rate(now, 10.0),
        "arr_rate_60s": state.arrival_rate(now, 60.0),
        "burst_ratio": state.burst_ratio(now),
        "roll_ttft_p50": _or_nan(state.ttft_ms.p50()),
        "roll_ttft_p95": _or_nan(state.ttft_ms.p95()),
        "roll_ttft_p99": _or_nan(state.ttft_ms.p99()),
        "roll_service_p50": _or_nan(state.service_ms.p50()),
        "roll_service_p95": _or_nan(state.service_ms.p95()),
        "roll_service_p99": _or_nan(state.service_ms.p99()),
        "cpu_util": _num(telemetry.get("cpu_util"), nan),
        "ram_util": _num(telemetry.get("ram_util"), nan),
        "gpu_util": _num(telemetry.get("gpu_util"), nan),
        "vram_used_mb": _num(telemetry.get("vram_used_mb"), nan),
        "vram_free_mb": _num(telemetry.get("vram_free_mb"), nan),
        "gpu_temp_c": _num(telemetry.get("gpu_temp_c"), nan),
        "cloud_success_rate": state.cloud_success_rate(),
        "cloud_ttft_p95": _or_nan(state.cloud_recent_ttft.p95()),
    }
    return {k: feats[k] for k in FEATURE_NAMES}


def _or_nan(v: float | None) -> float:
    return float("nan") if v is None else float(v)


def _num(v, default: float) -> float:
    return default if v is None else float(v)


def ttft_breach(end_to_end_ttft_ms: float | None, sla_ms: float, ok: bool) -> int:
    if not ok or end_to_end_ttft_ms is None:
        return 1
    return 1 if end_to_end_ttft_ms > sla_ms else 0


def occupancy_pressure(service_ms: float | None, occupancy_ms: float) -> int:
    if service_ms is None:
        return 0
    return 1 if service_ms > occupancy_ms else 0


def freeze_thresholds(ttft_ms, service_ms, breach_percentile: float = 0.65,
                      occ_percentile: float = 0.85) -> dict:
    ttft = np.asarray([t for t in ttft_ms if t is not None], dtype=float)
    svc = np.asarray([s for s in service_ms if s is not None], dtype=float)
    return {
        "sla_ttft_ms": round(float(np.quantile(ttft, breach_percentile)), 1),
        "occupancy_service_ms": round(float(np.quantile(svc, occ_percentile)), 1),
        "breach_percentile": breach_percentile, "occ_percentile": occ_percentile,
        "n": int(len(ttft)),
    }
