from __future__ import annotations

import time

from .core import (Config, GatewayState, Request, build_features,
                   sample_telemetry, ttft_breach, occupancy_pressure)
from .events import EventLog, make_event
from .policies import Policy, RiskModel, GuardedGovernor
from .providers import ProviderResult


class Gateway:
    def __init__(self, cfg: Config, local_provider, cloud_provider,
                 state: GatewayState, event_log: EventLog,
                 run_id: str, trace_id: str):
        self.cfg = cfg
        self.local = local_provider
        self.cloud = cloud_provider
        self.state = state
        self.log = event_log
        self.run_id = run_id
        self.trace_id = trace_id

    async def handle(self, req: Request, policy: Policy, *,
                     threshold: float, ml: RiskModel | None = None,
                     route_override: str | None = None) -> dict:
        st = self.state
        st.record_arrival(req.t_received)

        tel = sample_telemetry()
        feats = build_features(st, req, time.perf_counter(), tel)

        t_decision0 = time.perf_counter()
        if route_override is not None:
            from .policies import Decision
            decision = Decision(route_override, f"forced_{route_override}", 0.0)
        else:
            decision = policy.decide(feats, threshold, ml)
        decision_ms = (time.perf_counter() - t_decision0) * 1000.0
        decision_meta = {
            "signal_tier": getattr(decision, "signal_tier", getattr(policy, "signal_tier", 0)),
            "decision_ms": decision_ms,
            "embedding_ms": float(getattr(decision, "embedding_ms", 0.0)),
            "probe_ms": float(getattr(decision, "probe_ms", 0.0)),
            "probe_tokens": int(getattr(decision, "probe_tokens", 0)),
            "router_cost_usd": float(getattr(decision, "router_cost_usd", 0.0)),
        }

        dispatch_t = time.perf_counter()
        if decision.route == "local":
            res = await self._run_local(req)
        elif decision.route == "cloud":
            res = await self._run_cloud(req)
        else:
            now = time.perf_counter()
            res = ProviderResult("rejected", "none", False,
                                 "no_executable_route", now, None, now,
                                 None, 0.0, 0, 0, 0.0, 0)

        outcome = self._outcome(req, res, decision_meta, dispatch_t)
        labels = self._labels(res, outcome)
        if isinstance(policy, GuardedGovernor):
            policy.record_outcome(decision, res.ok, res.provider_ttft_ms)
        decision_payload = {
            "route": decision.route, "reason_code": decision.reason_code,
            "score": _finite(decision.score),
            "signal_tier": decision_meta["signal_tier"],
        }
        for field in ("expert_name", "regime", "discretionary", "safety_override",
                      "telemetry_tier", "budget_used", "budget_cap",
                      "budget_remaining", "budget_denied", "circuit_state",
                      "matched_budget"):
            decision_payload[field] = getattr(decision, field)
        self.log.write(make_event(
            run_id=self.run_id, trace_id=self.trace_id, policy=policy.name,
            seed=self.cfg.seed, request_id=req.request_id, trace_pos=req.trace_pos,
            task_category=req.task_category,
            decision=decision_payload,
            features=feats, outcome=outcome, labels=labels,
            workload_phase=req.workload_phase, cloud_degraded=req.cloud_degraded,
        ))
        return {"decision": decision, "outcome": outcome, "labels": labels}

    async def _run_local(self, req: Request) -> ProviderResult:
        st = self.state
        st.pending_local += 1
        async with st.semaphore:
            st.pending_local -= 1
            st.active_local += 1
            try:
                res = await self.local.generate(req.prompt, req.max_tokens)
            finally:
                st.active_local -= 1
        e2e_ttft = _e2e_ttft_ms(req, res)
        st.on_local_complete(e2e_ttft, res.service_ms, res.ok)
        return res

    async def _run_cloud(self, req: Request) -> ProviderResult:
        st = self.state
        st.active_cloud += 1
        try:
            res = await self.cloud.generate(req.prompt, req.max_tokens)
        finally:
            st.active_cloud -= 1
        st.on_cloud_complete(_e2e_ttft_ms(req, res), res.ok)
        return res

    def _outcome(self, req: Request, res: ProviderResult, decision_meta: dict,
                 dispatch_t: float) -> dict:
        qwait = ((res.t_provider_start - req.t_received) * 1000.0)
        e2e = ((res.t_completed - req.t_received) * 1000.0) if res.t_completed else None
        e2e_ttft = _e2e_ttft_ms(req, res)
        return {
            "executed_on": res.role,
            "queue_wait_ms": qwait,
            "provider_ttft_ms": res.provider_ttft_ms,
            "end_to_end_ttft_ms": e2e_ttft,
            "decision_ms": decision_meta["decision_ms"],
            "embedding_ms": decision_meta["embedding_ms"],
            "probe_ms": decision_meta["probe_ms"],
            "probe_tokens": decision_meta["probe_tokens"],
            "router_cost_usd": decision_meta["router_cost_usd"],
            "realized_ttft_with_decision_ms": e2e_ttft,
            "service_ms": res.service_ms,
            "end_to_end_ms": e2e,
            "prompt_tokens": res.prompt_tokens,
            "output_tokens": res.output_tokens,
            "cost_usd": res.cost_usd,
            "ok": res.ok,
            "error": res.error,
            "arrival_offset_ms": (req.scheduled_arrival_offset_s * 1000.0
                                  if req.scheduled_arrival_offset_s is not None else None),
            "dispatch_offset_ms": ((dispatch_t - req.t_received) * 1000.0),
            "provider_start_offset_ms": ((res.t_provider_start - req.t_received) * 1000.0),
            "first_token_offset_ms": ((res.t_first_text - req.t_received) * 1000.0
                                      if res.t_first_text is not None else None),
            "completion_offset_ms": ((res.t_completed - req.t_received) * 1000.0
                                     if res.t_completed is not None else None),
            "extra": getattr(res, "extra", None) or {},
        }

    def _labels(self, res: ProviderResult, outcome: dict) -> dict:
        if res.role == "local":
            return {
                "local_ttft_breach": ttft_breach(
                    outcome["end_to_end_ttft_ms"], self.cfg.targets.sla_ttft_ms, res.ok),
                "local_occupancy_pressure": occupancy_pressure(
                    res.service_ms, self.cfg.targets.occupancy_service_ms),
            }
        return {"local_ttft_breach": None, "local_occupancy_pressure": None}


def _e2e_ttft_ms(req: Request, res: ProviderResult) -> float | None:
    if res.t_first_text is None:
        return None
    return (res.t_first_text - req.t_received) * 1000.0


def _finite(x: float) -> float:
    if x == float("inf"):
        return 1e9
    if x == float("-inf"):
        return -1e9
    return float(x)
