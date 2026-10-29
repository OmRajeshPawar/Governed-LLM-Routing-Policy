"""Telemetry for the SOC4AI live study.

GpuSampler runs in its own thread (never on the asyncio loop) and writes a 1 Hz
CSV of GPU power, temperature, utilisation, VRAM, pstate and throttle flags,
plus CPU and RAM. TracingBreaker records every circuit-breaker state change.
"""
from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import psutil

from apex.policies import CloudCircuitBreaker

# NVML clocks-event (throttle) reason bits
SW_POWER_CAP = 0x4
HW_SLOWDOWN = 0x8
SW_THERMAL = 0x20
HW_THERMAL = 0x40


def _try(fn, default=None):
    try:
        return fn()
    except Exception:
        return default


def gpu_temp_c() -> float | None:
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        t = float(pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        pynvml.nvmlShutdown()
        return t
    except Exception:
        return None


class GpuSampler:
    FIELDS = ["perf_s", "unix_s", "gpu_util_pct", "mem_util_pct", "vram_used_mb",
              "temp_c", "power_w", "pstate", "throttle_bits", "sw_power_cap",
              "hw_slowdown", "sw_thermal", "hw_thermal", "cpu_pct", "ram_pct"]

    def __init__(self, path: str | Path, interval_s: float = 1.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.interval = float(interval_s)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples = 0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="gpu-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self) -> None:
        nv, pynvml, h = False, None, None
        try:
            import pynvml as _p
            _p.nvmlInit()
            pynvml, h, nv = _p, _p.nvmlDeviceGetHandleByIndex(0), True
        except Exception:
            nv = False
        psutil.cpu_percent(None)
        reasons_fn = None
        if nv:
            reasons_fn = (getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
                          or getattr(pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None))
        with open(self.path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(self.FIELDS)
            while not self._stop.is_set():
                row = {"perf_s": time.perf_counter(), "unix_s": time.time(),
                       "cpu_pct": psutil.cpu_percent(None),
                       "ram_pct": psutil.virtual_memory().percent}
                if nv:
                    util = _try(lambda: pynvml.nvmlDeviceGetUtilizationRates(h))
                    mem = _try(lambda: pynvml.nvmlDeviceGetMemoryInfo(h))
                    bits = _try(lambda: int(reasons_fn(h))) if reasons_fn else None
                    row.update({
                        "gpu_util_pct": getattr(util, "gpu", None),
                        "mem_util_pct": getattr(util, "memory", None),
                        "vram_used_mb": (mem.used / 2 ** 20) if mem else None,
                        "temp_c": _try(lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)),
                        "power_w": _try(lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0),
                        "pstate": _try(lambda: pynvml.nvmlDeviceGetPerformanceState(h)),
                        "throttle_bits": bits,
                        "sw_power_cap": int(bool(bits & SW_POWER_CAP)) if bits is not None else None,
                        "hw_slowdown": int(bool(bits & HW_SLOWDOWN)) if bits is not None else None,
                        "sw_thermal": int(bool(bits & SW_THERMAL)) if bits is not None else None,
                        "hw_thermal": int(bool(bits & HW_THERMAL)) if bits is not None else None,
                    })
                w.writerow([row.get(k) for k in self.FIELDS])
                fh.flush()
                self.samples += 1
                self._stop.wait(self.interval)
        if nv:
            _try(pynvml.nvmlShutdown)


class TracingBreaker(CloudCircuitBreaker):
    """Same logic as CloudCircuitBreaker; appends every state change."""

    def _note(self, before: str, cause: str) -> None:
        if self.state != before:
            self.transitions.append({
                "perf_s": time.perf_counter(), "unix_s": time.time(),
                "from": before, "to": self.state,
                "reason": self.reason_code, "cause": cause,
            })

    def on_arrival(self, now: float | None = None) -> None:
        before = self.state
        super().on_arrival(now)
        self._note(before, "arrival")

    def record(self, ok: bool, ttft_ms: float | None, now: float | None = None) -> None:
        before = self.state
        super().record(ok, ttft_ms, now)
        self._note(before, "outcome")


def attach_tracing_breaker(governor) -> None:
    governor.circuit.__class__ = TracingBreaker
    governor.circuit.transitions = []
