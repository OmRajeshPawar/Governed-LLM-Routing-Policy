from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pandas as pd

from . import SCHEMA_VERSION, FEATURE_VERSION
from .core import FEATURE_NAMES

OUTCOME_FIELDS = [
    "executed_on", "queue_wait_ms", "provider_ttft_ms", "end_to_end_ttft_ms",
    "decision_ms", "embedding_ms", "probe_ms", "probe_tokens", "router_cost_usd",
    "realized_ttft_with_decision_ms", "service_ms", "end_to_end_ms",
    "prompt_tokens", "output_tokens", "cost_usd", "ok", "error",
    "arrival_offset_ms", "dispatch_offset_ms", "provider_start_offset_ms",
    "first_token_offset_ms", "completion_offset_ms",
]
LABEL_FIELDS = ["local_ttft_breach", "local_occupancy_pressure"]


def make_event(*, run_id, trace_id, policy, seed, request_id, trace_pos,
               task_category, decision, features, outcome, labels,
               workload_phase=None, cloud_degraded=False) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "feature_version": FEATURE_VERSION,
        "run_id": run_id, "trace_id": trace_id, "policy": policy, "seed": seed,
        "request_id": request_id, "trace_pos": trace_pos, "task_category": task_category,
        "workload_phase": workload_phase, "cloud_degraded": bool(cloud_degraded),
        "decision": decision,
        "features": features,
        "outcome": outcome,
        "labels": labels,
    }


class EventLog:

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._q: asyncio.Queue = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._fh = None

    async def start(self):
        self._fh = open(self.path, "w", encoding="utf-8")
        self._task = asyncio.create_task(self._drain())

    def write(self, record: dict):
        self._q.put_nowait(record)

    async def _drain(self):
        try:
            while True:
                rec = await self._q.get()
                if rec is None:
                    break
                self._fh.write(json.dumps(rec) + "\n")
        finally:
            if self._fh:
                self._fh.flush()

    async def aclose(self):
        await self._q.put(None)
        if self._task:
            await self._task
        if self._fh:
            self._fh.close()


def write_manifest(path: str | Path, cfg_dump: dict, extra: dict) -> None:
    manifest = {"schema_version": SCHEMA_VERSION, "feature_version": FEATURE_VERSION,
                "config": cfg_dump, **extra}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_events(jsonl_path: str | Path) -> pd.DataFrame:
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            row = {
                "run_id": e["run_id"], "trace_id": e["trace_id"], "policy": e["policy"],
                "request_id": e["request_id"], "trace_pos": e["trace_pos"],
                "task_category": e["task_category"],
                "route": e["decision"]["route"], "reason_code": e["decision"]["reason_code"],
                "decision_score": e["decision"]["score"],
                "signal_tier": e["decision"].get("signal_tier"),
                "expert_name": e["decision"].get("expert_name"),
                "regime": e["decision"].get("regime"),
                "telemetry_tier": e["decision"].get("telemetry_tier"),
                "discretionary": e["decision"].get("discretionary"),
                "safety_override": e["decision"].get("safety_override"),
                "budget_used": e["decision"].get("budget_used"),
                "budget_cap": e["decision"].get("budget_cap"),
                "budget_remaining": e["decision"].get("budget_remaining"),
                "budget_denied": e["decision"].get("budget_denied"),
                "circuit_state": e["decision"].get("circuit_state"),
                "matched_budget": e["decision"].get("matched_budget"),
                "workload_phase": e.get("workload_phase"),
                "cloud_degraded": e.get("cloud_degraded", False),
            }
            for k in FEATURE_NAMES:
                row[k] = e["features"].get(k)
            for k in OUTCOME_FIELDS:
                row[k] = e["outcome"].get(k)
            for k in LABEL_FIELDS:
                row[k] = e["labels"].get(k)
            rows.append(row)
    return pd.DataFrame(rows)


def resolve_event_files(events_dir: str | Path, *, default_glob: str = "shadow_*.jsonl") -> tuple[list[Path], str]:
    base = Path(events_dir)
    exact = os.getenv("APEX_SHADOW_FILES")
    if exact:
        files = []
        missing = []
        for raw in exact.split(";"):
            raw = raw.strip()
            if not raw:
                continue
            path = Path(raw)
            if not path.is_absolute():
                path = base / path
            if path.exists():
                files.append(path)
            else:
                missing.append(path)
        if missing:
            detail = ", ".join(str(path) for path in missing[:5])
            extra = "" if len(missing) <= 5 else f" (+{len(missing) - 5} more)"
            raise FileNotFoundError(f"APEX_SHADOW_FILES listed missing event files: {detail}{extra}")
        return sorted(files), f"APEX_SHADOW_FILES[{len(files)}]"
    glob = os.getenv("APEX_SHADOW_GLOB", default_glob)
    return sorted(base.glob(glob)), f"glob={glob}"


def events_to_parquet(jsonl_path: str | Path, parquet_path: str | Path) -> int:
    df = load_events(jsonl_path)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, index=False)
    return len(df)
