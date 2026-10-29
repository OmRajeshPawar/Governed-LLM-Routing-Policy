from __future__ import annotations

import hashlib
import heapq
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from apex.core import Config
from apex.core import Request, build_features
from apex.policies import GuardedGovernor
from apex.policies import Decision, Policy, RiskModel
from apex.core import GatewayState
from apex.workload import make_trace


@dataclass
class SimulatedRequest:
    request_id: str
    trace_pos: int
    arrival_s: float
    route: str
    reason_code: str
    regime: str | None
    expert_name: str | None
    queue_wait_ms: float
    provider_ttft_ms: float | None
    end_to_end_ttft_ms: float | None
    service_ms: float
    completion_s: float
    ok: bool
    cost_usd: float
    budget_used: int
    budget_cap: int
    matched_budget: bool


def load_local_execution_trace(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda x: int(x["trace_pos"]))
    return rows


def regenerate_arrivals(rows: list[dict]) -> list[float]:
    if all(r.get("outcome", {}).get("arrival_offset_ms") is not None for r in rows):
        return [float(r["outcome"]["arrival_offset_ms"]) / 1000.0 for r in rows]
    first = rows[0]
    seed = int(first["seed"])
    run_id = first["run_id"]
    mix = next((m for m in ("reasoning_coding", "real_sample", "synthetic")
                if m in run_id), "synthetic")
    pattern = next((p for p in ("saturation", "bursty", "steady") if p in run_id), "bursty")
    return [x["arrival_offset"] for x in make_trace(len(rows), pattern, seed, mix)]


class ClosedLoopSimulator:
    def __init__(self, cfg: Config, rows: list[dict], policy: Policy, *,
                 threshold: float, ml: RiskModel | None = None,
                 cloud_base_ttft_ms: float = 325.0,
                 cloud_per_token_ms: float = 36.3977,
                 seed: int = 0, route_by_pos: dict[int, str] | None = None):
        self.cfg = cfg
        self.rows = rows
        self.policy = policy
        self.threshold = threshold
        self.ml = ml
        self.cloud_base_ttft_ms = cloud_base_ttft_ms
        self.cloud_per_token_ms = cloud_per_token_ms
        self.seed = seed
        self.route_by_pos = route_by_pos
        self.state = GatewayState(cfg)
        self.worker_free = [0.0] * cfg.local_concurrency
        self.completions: list[tuple] = []
        self.results: list[SimulatedRequest] = []

    def run(self) -> list[SimulatedRequest]:
        arrivals = regenerate_arrivals(self.rows)
        for row, now in zip(self.rows, arrivals):
            self._process_completions(now)
            self._refresh_queue_counts(now)
            self.state.record_arrival(now)
            req = self._request(row, now)
            telemetry = {k: row["features"].get(k) for k in
                         ("cpu_util", "ram_util", "gpu_util", "vram_used_mb",
                          "vram_free_mb", "gpu_temp_c")}
            feat = build_features(self.state, req, now, telemetry)
            if self.route_by_pos is not None:
                route = self.route_by_pos[int(row["trace_pos"])]
                decision = Decision(route, f"frozen_schedule_{route}", 0.0)
            else:
                decision = self.policy.decide(feat, self.threshold, self.ml)
            if decision.route == "cloud":
                result = self._schedule_cloud(row, req, now, decision)
            elif decision.route == "local":
                result = self._schedule_local(row, req, now, decision)
            else:
                result = SimulatedRequest(req.request_id, req.trace_pos, now, "reject",
                                          decision.reason_code, decision.regime,
                                          decision.expert_name, 0.0, None, None, 0.0,
                                          now, False, 0.0, decision.budget_used,
                                          decision.budget_cap, decision.matched_budget)
            self.results.append(result)
        self._process_completions(float("inf"))
        return self.results

    def _request(self, row: dict, now: float) -> Request:
        f = row["features"]
        chars = max(1, int(f.get("prompt_chars", 4)))
        prompt = ("code " if f.get("is_code") else "request ")
        prompt = (prompt * (chars // len(prompt) + 1))[:chars]
        req = Request(row["request_id"], prompt, int(f["max_output_tokens"]),
                      row["task_category"], now)
        req.trace_pos = int(row["trace_pos"])
        req.workload_phase = row.get("workload_phase")
        req.cloud_degraded = bool(row.get("cloud_degraded", False))
        return req

    def _schedule_local(self, row: dict, req: Request, now: float, decision) -> SimulatedRequest:
        worker = min(range(len(self.worker_free)), key=self.worker_free.__getitem__)
        start = max(now, self.worker_free[worker])
        service = float(row["outcome"]["service_ms"] or 0.0)
        provider_ttft = row["outcome"].get("provider_ttft_ms")
        completion = start + service / 1000.0
        self.worker_free[worker] = completion
        queue_wait = (start - now) * 1000.0
        e2e_ttft = (queue_wait + float(provider_ttft)) if provider_ttft is not None else None
        ok = bool(row["outcome"].get("ok", True))
        heapq.heappush(self.completions,
                       (completion, req.trace_pos, "local", e2e_ttft, service, ok,
                        decision, provider_ttft))
        return SimulatedRequest(req.request_id, req.trace_pos, now, "local",
                                decision.reason_code, decision.regime, decision.expert_name,
                                queue_wait, provider_ttft, e2e_ttft, service, completion,
                                ok, 0.0, decision.budget_used, decision.budget_cap,
                                decision.matched_budget)

    def _schedule_cloud(self, row: dict, req: Request, now: float, decision) -> SimulatedRequest:
        rng = random.Random(self._request_seed(req.request_id))
        jitter = math.exp(rng.gauss(0.0, self.cfg.providers.cloud.jitter_sigma))
        ttft = self.cloud_base_ttft_ms * jitter
        if req.cloud_degraded:
            ttft *= 3.0
        service = ttft + self.cloud_per_token_ms * req.max_tokens
        completion = now + service / 1000.0
        cost = ((req.prompt_token_est + req.max_tokens) / 1000.0 *
                self.cfg.providers.cloud.cost_per_1k_tokens)
        ok = not (req.cloud_degraded and rng.random() < 0.20)
        heapq.heappush(self.completions,
                       (completion, req.trace_pos, "cloud", ttft, service, ok,
                        decision, ttft))
        return SimulatedRequest(req.request_id, req.trace_pos, now, "cloud",
                                decision.reason_code, decision.regime, decision.expert_name,
                                0.0, ttft, ttft, service, completion, ok, cost,
                                decision.budget_used, decision.budget_cap,
                                decision.matched_budget)

    def _process_completions(self, now: float) -> None:
        while self.completions and self.completions[0][0] <= now:
            _, _, route, ttft, service, ok, decision, provider_ttft = heapq.heappop(self.completions)
            if route == "local":
                self.state.on_local_complete(ttft, service, ok)
            else:
                self.state.on_cloud_complete(ttft, ok)
                if isinstance(self.policy, GuardedGovernor):
                    self.policy.record_outcome(decision, ok, provider_ttft)

    def _refresh_queue_counts(self, now: float) -> None:
        active = sum(1 for completion in self.worker_free if completion > now)
        queued = sum(1 for completion, _, route, *_ in self.completions
                     if route == "local" and completion > now) - active
        self.state.active_local = min(self.cfg.local_concurrency, max(0, active))
        self.state.pending_local = max(0, queued)
        self.state.active_cloud = sum(1 for completion, _, route, *_ in self.completions
                                      if route == "cloud" and completion > now)

    def _request_seed(self, request_id: str) -> int:
        digest = hashlib.sha256(f"{self.seed}:{request_id}".encode()).digest()
        return int.from_bytes(digest[:8], "big")


def stage_a_metrics(rows: list[dict], simulated: list[SimulatedRequest]) -> dict:
    observed_q = np.array([float(r["outcome"]["queue_wait_ms"]) for r in rows])
    observed_t = np.array([float(r["outcome"]["end_to_end_ttft_ms"]) for r in rows])
    sim_q = np.array([r.queue_wait_ms for r in simulated])
    sim_t = np.array([r.end_to_end_ttft_ms for r in simulated], dtype=float)
    mean_q_denom = max(1.0, float(np.mean(observed_q)))
    p95_q_denom = max(1.0, float(np.quantile(observed_q, .95)))
    p95_t_denom = max(1.0, float(np.quantile(observed_t, .95)))
    metrics = {
        "queue_wait_mae_ms": float(np.mean(np.abs(sim_q - observed_q))),
        "mean_queue_wait_relative_error": abs(float(np.mean(sim_q) - np.mean(observed_q))) / mean_q_denom,
        "p95_queue_wait_relative_error": abs(float(np.quantile(sim_q, .95) - np.quantile(observed_q, .95))) / p95_q_denom,
        "p95_ttft_relative_error": abs(float(np.quantile(sim_t, .95) - np.quantile(observed_t, .95))) / p95_t_denom,
    }
    gates = {
        "queue_wait_mae_ms": metrics["queue_wait_mae_ms"] <= 100.0,
        "mean_queue_wait_relative_error": metrics["mean_queue_wait_relative_error"] <= .05,
        "p95_queue_wait_relative_error": metrics["p95_queue_wait_relative_error"] <= .05,
        "p95_ttft_relative_error": metrics["p95_ttft_relative_error"] <= .05,
    }
    return {
        "stage": "A_event_engine_fidelity",
        "status": "pass" if all(gates.values()) else "fail",
        "metrics": metrics,
        "gates": gates,
        "interpretation": (
            "Stage A validates discrete-event implementation fidelity. Because it "
            "reconstructs an all-local run using service times observed in that run, "
            "it does not independently validate the service-time counterfactual."
        ),
    }


def write_simulation(path: str | Path, results: list[SimulatedRequest]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps([asdict(x) for x in results], indent=2), encoding="utf-8")
