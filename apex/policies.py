from __future__ import annotations

import hashlib
import json
import math
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from .core import Config, _repo_root


@dataclass
class Decision:
    route: str
    reason_code: str
    score: float
    signal_tier: int = 0
    embedding_ms: float = 0.0
    probe_ms: float = 0.0
    probe_tokens: int = 0
    router_cost_usd: float = 0.0
    expert_name: str | None = None
    regime: str | None = None
    discretionary: bool = False
    safety_override: bool = False
    telemetry_tier: str = "full"
    budget_used: int = 0
    budget_cap: int = 0
    budget_remaining: int = 0
    budget_denied: bool = False
    circuit_state: str = "closed"
    matched_budget: bool = True


class RiskModel(Protocol):
    def p_ttft(self, feat: dict) -> float: ...
    def p_occ(self, feat: dict) -> float: ...


def _nz(x: float, default: float = 0.0) -> float:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return default
    return float(x)


HEURISTIC_WEIGHTS = {
    "queue": 0.9, "active": 0.5, "ttft_tail": 1.2, "service_tail": 0.8,
    "burst": 0.4, "vram": 0.5, "work": 0.6,
}


class Policy:
    name: str = "base"
    signal_tier: int = 0
    estimated_decision_ms: float = 0.05
    estimated_embedding_ms: float = 0.0
    estimated_probe_ms: float = 0.0
    estimated_probe_tokens: int = 0
    estimated_router_cost_usd: float = 0.0

    def score(self, feat: dict, ml: RiskModel | None = None) -> float:
        raise NotImplementedError

    def decide(self, feat: dict, threshold: float, ml: RiskModel | None = None) -> Decision:
        s = self.score(feat, ml)
        route = "cloud" if s >= threshold else "local"
        return self._decision(route, self._reason(route, feat), s)

    def _decision(self, route: str, reason_code: str, score: float) -> Decision:
        return Decision(
            route=route,
            reason_code=reason_code,
            score=score,
            signal_tier=self.signal_tier,
            embedding_ms=self.estimated_embedding_ms,
            probe_ms=self.estimated_probe_ms,
            probe_tokens=self.estimated_probe_tokens,
            router_cost_usd=self.estimated_router_cost_usd,
        )

    def _reason(self, route: str, feat: dict) -> str:
        return f"{self.name}:{route}"


class AlwaysLocal(Policy):
    name = "always_local"
    def score(self, feat, ml=None): return float("-inf")
    def decide(self, feat, threshold=0.0, ml=None):
        return self._decision("local", "always_local", float("-inf"))


class AlwaysCloud(Policy):
    name = "always_cloud"
    def score(self, feat, ml=None): return float("inf")
    def decide(self, feat, threshold=0.0, ml=None):
        return self._decision("cloud", "always_cloud", float("inf"))


class QueueThreshold(Policy):
    name = "queue_threshold"
    def score(self, feat, ml=None): return _nz(feat["pending_local"])
    def _reason(self, route, feat):
        return "queue_high" if route == "cloud" else "queue_ok"


class TailAwareHeuristic(Policy):
    name = "tail_aware"
    signal_tier = 0
    estimated_decision_ms = 0.08

    def __init__(self, cfg: Config, weights: dict | None = None):
        self.cfg = cfg
        self.w = weights or HEURISTIC_WEIGHTS

    def score(self, feat: dict, ml=None) -> float:
        c = self.cfg
        conc = c.local_concurrency
        sla = c.targets.sla_ttft_ms
        occ = c.targets.occupancy_service_ms

        queue = _nz(feat["pending_local"]) / max(1, c.policies.queue_threshold)
        active = _nz(feat["active_local"]) / max(1, conc)
        ttft_tail = _nz(feat["roll_ttft_p95"]) / sla
        service_tail = _nz(feat["roll_service_p95"]) / occ
        burst = max(0.0, _nz(feat["burst_ratio"], 1.0) - 1.0)
        used, free = _nz(feat["vram_used_mb"]), _nz(feat["vram_free_mb"])
        vram = used / (used + free) if (used + free) > 0 else 0.0
        work = (_nz(feat["prompt_token_est"]) + _nz(feat["max_output_tokens"])) / 2048.0

        w = self.w
        return (w["queue"] * queue + w["active"] * active + w["ttft_tail"] * ttft_tail
                + w["service_tail"] * service_tail + w["burst"] * burst
                + w["vram"] * vram + w["work"] * work)

    def decide(self, feat, threshold=None, ml=None) -> Decision:
        thr = self.cfg.policies.tail_aware_pressure_threshold if threshold is None else threshold
        s = self.score(feat)
        if _nz(feat["roll_ttft_p95"]) > self.cfg.targets.sla_ttft_ms:
            return self._decision("cloud", "tail_hard_breach", s)
        route = "cloud" if s >= thr else "local"
        reason = "tail_pressure" if route == "cloud" else "tail_ok"
        return self._decision(route, reason, s)


class MLTTFTOnly(Policy):
    name = "ml_ttft"
    signal_tier = 1
    estimated_decision_ms = 0.85
    def score(self, feat, ml: RiskModel | None = None) -> float:
        return ml.p_ttft(feat) if ml else 0.0
    def _reason(self, route, feat):
        return "ml_ttft_risk" if route == "cloud" else "ml_ttft_ok"


class MLDualRisk(Policy):
    name = "ml_dual_risk"
    signal_tier = 1
    estimated_decision_ms = 1.10
    def score(self, feat, ml: RiskModel | None = None) -> float:
        if not ml:
            return 0.0
        return max(ml.p_ttft(feat), ml.p_occ(feat))
    def decide(self, feat, threshold=0.5, ml: RiskModel | None = None) -> Decision:
        if not ml:
            return self._decision("local", "ml_unavailable", 0.0)
        p_t, p_o = ml.p_ttft(feat), ml.p_occ(feat)
        s = max(p_t, p_o)
        if s < threshold:
            return self._decision("local", "ml_dual_ok", s)
        reason = "ml_ttft_risk" if p_t >= p_o else "ml_occupancy_risk"
        return self._decision("cloud", reason, s)


class SemanticEmbeddingSignal(Policy):
    name = "semantic_embedding"
    signal_tier = 2
    estimated_decision_ms = 10.0
    estimated_embedding_ms = 9.0

    def __init__(self, cfg: Config, heuristic_weights: dict | None = None):
        self.tail = TailAwareHeuristic(cfg, heuristic_weights)

    def score(self, feat, ml: RiskModel | None = None) -> float:
        work = (_nz(feat["prompt_token_est"]) + _nz(feat["max_output_tokens"])) / 2048.0
        semantic = (
            0.40 * _nz(feat["is_longform"]) +
            0.30 * _nz(feat["is_code"]) +
            0.20 * _nz(feat["is_json"]) +
            0.25 * _nz(feat["context_bucket"]) +
            0.35 * work
        )
        return 0.65 * self.tail.score(feat) + 0.35 * semantic

    def _reason(self, route, feat):
        return "semantic_complexity" if route == "cloud" else "semantic_ok"


class LocalLLMProbeSignal(Policy):
    name = "local_llm_probe"
    signal_tier = 3
    estimated_decision_ms = 220.0
    estimated_probe_ms = 215.0
    estimated_probe_tokens = 32

    def __init__(self, cfg: Config, heuristic_weights: dict | None = None):
        self.semantic = SemanticEmbeddingSignal(cfg, heuristic_weights)

    def score(self, feat, ml: RiskModel | None = None) -> float:
        ml_part = max(ml.p_ttft(feat), ml.p_occ(feat)) if ml else 0.0
        return 0.55 * self.semantic.score(feat, ml) + 0.45 * ml_part

    def _reason(self, route, feat):
        return "llm_probe_risk" if route == "cloud" else "llm_probe_ok"


class ThinkingProbeSignal(Policy):
    name = "thinking_probe"
    signal_tier = 4
    estimated_decision_ms = 650.0
    estimated_probe_ms = 640.0
    estimated_probe_tokens = 96

    def __init__(self, cfg: Config, heuristic_weights: dict | None = None):
        self.llm_probe = LocalLLMProbeSignal(cfg, heuristic_weights)

    def score(self, feat, ml: RiskModel | None = None) -> float:
        reasoning_load = _nz(feat["max_output_tokens"]) / 256.0
        return self.llm_probe.score(feat, ml) + 0.10 * reasoning_load

    def _reason(self, route, feat):
        return "thinking_probe_risk" if route == "cloud" else "thinking_probe_ok"


def all_policies(cfg: Config, heuristic_weights: dict | None = None) -> dict[str, Policy]:
    return {
        "always_local": AlwaysLocal(),
        "always_cloud": AlwaysCloud(),
        "queue_threshold": QueueThreshold(),
        "tail_aware": TailAwareHeuristic(cfg, heuristic_weights),
        "ml_ttft": MLTTFTOnly(),
        "ml_dual_risk": MLDualRisk(),
        "semantic_embedding": SemanticEmbeddingSignal(cfg, heuristic_weights),
        "local_llm_probe": LocalLLMProbeSignal(cfg, heuristic_weights),
        "thinking_probe": ThinkingProbeSignal(cfg, heuristic_weights),
    }


def make_guarded_governor(cfg: Config, *, episode_requests: int,
                          frozen_cloud_p95_ms: float,
                          heuristic_weights: dict | None = None,
                          local_executable: Callable[[], bool] | None = None
                          ) -> "GuardedGovernor":
    artifact = GovernorArtifact.load(_repo_root() / cfg.governor.artifact_path)
    experts = all_policies(cfg, heuristic_weights)
    return GuardedGovernor(cfg, artifact, experts,
                           episode_requests=episode_requests,
                           frozen_cloud_p95_ms=frozen_cloud_p95_ms,
                           local_executable=local_executable)


def frozen_cloud_p95_from_calibration(cfg: Config, calib: dict,
                                      est_output_tokens: float) -> float:
    cc = cfg.providers.cloud
    base = calib.get("base_ttft_ms")
    if base is None:
        base = cc.base_ttft_ms if cc.base_ttft_ms is not None else 200.0
    median = cc.network_rtt_ms + float(base) * cc.speed_ratio
    return float(median * math.exp(1.645 * cc.jitter_sigma))


def tune_tail_aware(cfg: Config, feat_rows: list[dict], local_breach,
                    budget: float, n_trials: int = 1500, seed: int = 0) -> dict:
    import numpy as np
    rng = np.random.default_rng(seed)
    N = len(feat_rows)
    k = int(round(budget * N))
    lb = np.asarray(local_breach, dtype=float)
    best_w, best_viol = dict(HEURISTIC_WEIGHTS), np.inf
    keys = list(HEURISTIC_WEIGHTS.keys())
    for _ in range(n_trials):
        w = {key: float(rng.uniform(0.0, 2.0)) for key in keys}
        h = TailAwareHeuristic(cfg, w)
        scores = np.array([h.score(f) for f in feat_rows])
        kept = np.argsort(-scores)[k:]
        viol = lb[kept].sum() / N
        if viol < best_viol:
            best_viol, best_w = viol, w
    return best_w


def _valid(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


@dataclass(frozen=True)
class GovernorArtifact:
    schema_version: str
    artifact_id: str
    created_utc: str
    regime_map: dict[str, str]
    thresholds: dict[str, float]
    promotion_status: dict
    development_seeds: list[int]
    protocol_sha256: str | None = None

    @classmethod
    def load(cls, path: str | Path) -> "GovernorArtifact":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        required = {"schema_version", "artifact_id", "created_utc", "regime_map",
                    "thresholds", "promotion_status", "development_seeds"}
        missing = required.difference(raw)
        if missing:
            raise ValueError(f"Governor artifact is missing: {sorted(missing)}")
        if raw["schema_version"] != "governor-v1":
            raise ValueError("Unsupported governor artifact schema")
        return cls(**{k: raw.get(k) for k in cls.__dataclass_fields__})

    @staticmethod
    def sha256(path: str | Path) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TotalCloudBudget:

    def __init__(self, episode_requests: int, fraction: float = 0.30,
                 mode: str = "evaluation_hard"):
        if mode not in {"evaluation_hard", "operational_soft"}:
            raise ValueError(f"Unsupported budget mode: {mode}")
        if episode_requests < 0 or not 0 <= fraction <= 1:
            raise ValueError("Invalid budget dimensions")
        self.mode = mode
        self.cap = math.floor(fraction * episode_requests)
        self.total_cloud = 0
        self.discretionary_cloud = 0
        self.safety_overrides = 0
        self.probes = 0
        self.denials = 0
        self.unmatched = False

    @property
    def remaining(self) -> int:
        return max(0, self.cap - self.total_cloud)

    def request(self, *, discretionary: bool = False, safety_override: bool = False,
                probe: bool = False) -> tuple[bool, str]:
        if self.total_cloud < self.cap:
            self._consume(discretionary, safety_override, probe)
            return True, "budget_admitted"
        if self.mode == "operational_soft" and safety_override:
            self._consume(discretionary, safety_override, probe)
            self.unmatched = True
            return True, "budget_soft_override"
        self.denials += 1
        return False, "budget_denied"

    def _consume(self, discretionary: bool, safety_override: bool, probe: bool) -> None:
        self.total_cloud += 1
        self.discretionary_cloud += int(discretionary)
        self.safety_overrides += int(safety_override)
        self.probes += int(probe)


class CloudCircuitBreaker:

    def __init__(self, cfg, frozen_cloud_p95_ms: float):
        self.cfg = cfg
        self.frozen_cloud_p95_ms = float(frozen_cloud_p95_ms)
        self.state = "closed"
        self.reason_code: str | None = None
        self.consecutive_failures = 0
        self.outcomes = deque(maxlen=cfg.failure_window)
        self.success_latencies = deque(maxlen=cfg.latency_window)
        self.opened_at = 0.0
        self.arrivals_while_open = 0
        self.half_open_probe_inflight = False
        self.half_open_successes = 0

    def on_arrival(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state != "open":
            return
        self.arrivals_while_open += 1
        if (now - self.opened_at >= self.cfg.cooldown_seconds and
                self.arrivals_while_open >= self.cfg.cooldown_arrivals):
            self.state = "half-open"
            self.half_open_probe_inflight = False
            self.half_open_successes = 0

    def cloud_available(self) -> bool:
        if self.state == "closed":
            return True
        return self.state == "half-open" and not self.half_open_probe_inflight

    def begin_attempt(self) -> bool:
        if self.state == "closed":
            return True
        if self.state == "half-open" and not self.half_open_probe_inflight:
            self.half_open_probe_inflight = True
            return True
        return False

    def record(self, ok: bool, ttft_ms: float | None, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        if self.state == "half-open":
            self.half_open_probe_inflight = False
            if ok and ttft_ms is not None and ttft_ms <= self._latency_limit:
                self.half_open_successes += 1
                if self.half_open_successes >= self.cfg.half_open_successes_to_close:
                    self._close()
            else:
                self._open("circuit_half_open_failure", now)
            return

        self.outcomes.append(bool(ok))
        if ok:
            self.consecutive_failures = 0
            if ttft_ms is not None:
                self.success_latencies.append(float(ttft_ms))
        else:
            self.consecutive_failures += 1

        if self.consecutive_failures >= self.cfg.consecutive_failure_limit:
            self._open("circuit_consecutive_failures", now)
        elif (len(self.outcomes) == self.cfg.failure_window and
              sum(not x for x in self.outcomes) >= self.cfg.failure_window_limit):
            self._open("circuit_failure_window", now)
        elif (len(self.success_latencies) >= self.cfg.latency_success_minimum and
              sum(x > self._latency_limit for x in self.success_latencies)
              >= self.cfg.latency_breach_limit):
            self._open("circuit_latency_window", now)

    @property
    def _latency_limit(self) -> float:
        return self.cfg.latency_multiplier * self.frozen_cloud_p95_ms

    def _open(self, reason: str, now: float) -> None:
        self.state = "open"
        self.reason_code = reason
        self.opened_at = now
        self.arrivals_while_open = 0
        self.half_open_probe_inflight = False
        self.half_open_successes = 0

    def _close(self) -> None:
        self.state = "closed"
        self.reason_code = None
        self.consecutive_failures = 0
        self.outcomes.clear()
        self.success_latencies.clear()
        self.half_open_successes = 0


class RegimeRecognizer:

    def __init__(self):
        self.current = "low"
        self.candidate = "low"
        self.candidate_count = 0

    def recognize(self, feat: dict) -> str:
        pending = float(feat.get("pending_local", 0) or 0)
        occ = float(feat.get("semaphore_occupancy", 0) or 0)
        burst = float(feat.get("burst_ratio", 1) or 1)
        rate = float(feat.get("arr_rate_5s", 0) or 0)
        if burst >= 2.5 or rate >= 1.5:
            raw = "burst"
        elif pending >= 3 or occ >= 1.0:
            raw = "high"
        elif pending >= 1 or occ >= 0.5:
            raw = "medium"
        else:
            raw = "low"
        if raw == self.current:
            self.candidate, self.candidate_count = raw, 0
        elif raw == self.candidate:
            self.candidate_count += 1
            if self.candidate_count >= 2:
                self.current, self.candidate_count = raw, 0
        else:
            self.candidate, self.candidate_count = raw, 1
        return self.current


class GuardedGovernor(Policy):
    name = "guarded_governor"

    def __init__(self, cfg: Config, artifact: GovernorArtifact,
                 experts: dict[str, Policy], episode_requests: int,
                 frozen_cloud_p95_ms: float,
                 local_executable: Callable[[], bool] | None = None):
        self.cfg = cfg
        self.artifact = artifact
        self.experts = experts
        self.tail = experts.get("tail_aware") or TailAwareHeuristic(cfg)
        self.regimes = RegimeRecognizer()
        b = cfg.governor.budget
        self.budget = TotalCloudBudget(episode_requests, b.target_fraction, b.mode)
        self.circuit = CloudCircuitBreaker(cfg.governor.circuit, frozen_cloud_p95_ms)
        self.local_executable = local_executable or (lambda: True)

    def score(self, feat: dict, ml: RiskModel | None = None) -> float:
        return self.tail.score(feat, ml)

    def decide(self, feat: dict, threshold: float = 0.0,
               ml: RiskModel | None = None) -> Decision:
        self.circuit.on_arrival()
        local_ok = bool(self.local_executable())
        cloud_ok = self.circuit.cloud_available()
        local_state = self._local_state(feat, local_ok)
        telemetry_tier = self._telemetry_tier(feat)

        if local_state in {"unavailable", "hard_breach"} and not cloud_ok:
            if local_ok:
                return self._final("local", "dual_degraded_local_only", 0.0,
                                   telemetry_tier, safety=True)
            return self._final("reject", "dual_degraded_no_executable_route", 0.0,
                               telemetry_tier, safety=True)
        if not cloud_ok:
            return self._final("local", "cloud_unavailable_local_fallback", 0.0,
                               telemetry_tier, safety=True)
        if local_state in {"unavailable", "hard_breach"}:
            return self._cloud_or_local("local_hard_breach_cloud_override", 1e9,
                                        telemetry_tier, safety=True)

        regime = self.regimes.recognize(feat)
        if telemetry_tier == "full":
            expert_name = self.artifact.regime_map.get(regime, "tail_aware")
            expert = self.experts.get(expert_name, self.tail)
        elif telemetry_tier == "tail":
            expert_name, expert = "tail_aware", self.tail
        elif telemetry_tier == "minimal":
            if float(feat.get("pending_local", 0) or 0) >= self.cfg.policies.queue_threshold:
                return self._cloud_or_local("minimal_queue_guard", 1.0, telemetry_tier,
                                            regime=regime, expert="minimal_queue_guard")
            return self._final("local", "minimal_queue_ok", 0.0, telemetry_tier,
                               regime=regime, expert="minimal_queue_guard")
        else:
            return self._cloud_or_local("local_observability_unusable", 1.0,
                                        telemetry_tier, safety=True, regime=regime)

        expert_threshold = self.artifact.thresholds.get(expert_name, threshold)
        proposal = expert.decide(feat, expert_threshold, ml)
        if proposal.route == "cloud":
            return self._cloud_or_local(proposal.reason_code, proposal.score,
                                        telemetry_tier, regime=regime,
                                        expert=expert_name, proposal=proposal)
        return self._final("local", proposal.reason_code, proposal.score,
                           telemetry_tier, regime=regime, expert=expert_name,
                           proposal=proposal)

    def _cloud_or_local(self, reason: str, score: float, telemetry_tier: str,
                        *, safety: bool = False, regime: str | None = None,
                        expert: str | None = None,
                        proposal: Decision | None = None) -> Decision:
        if not self.circuit.begin_attempt():
            return self._final("local", f"{reason}:circuit_unavailable", score,
                               telemetry_tier, regime=regime, expert=expert,
                               proposal=proposal, safety=True)
        allowed, budget_reason = self.budget.request(
            discretionary=not safety, safety_override=safety)
        if not allowed:
            if self.circuit.state == "half-open":
                self.circuit.half_open_probe_inflight = False
            return self._final("local", f"{reason}:budget_denied", score,
                               telemetry_tier, regime=regime, expert=expert,
                               proposal=proposal, budget_denied=True)
        return self._final("cloud", f"{reason}:{budget_reason}", score,
                           telemetry_tier, regime=regime, expert=expert,
                           proposal=proposal, safety=safety,
                           discretionary=not safety)

    def _final(self, route: str, reason: str, score: float, telemetry_tier: str,
               *, regime: str | None = None, expert: str | None = None,
               proposal: Decision | None = None, safety: bool = False,
               discretionary: bool = False, budget_denied: bool = False) -> Decision:
        p = proposal or Decision(route, reason, score)
        return Decision(
            route=route, reason_code=reason, score=score,
            signal_tier=p.signal_tier, embedding_ms=p.embedding_ms,
            probe_ms=p.probe_ms, probe_tokens=p.probe_tokens,
            router_cost_usd=p.router_cost_usd, expert_name=expert,
            regime=regime, discretionary=discretionary,
            safety_override=safety, telemetry_tier=telemetry_tier,
            budget_used=self.budget.total_cloud, budget_cap=self.budget.cap,
            budget_remaining=self.budget.remaining,
            budget_denied=budget_denied, circuit_state=self.circuit.state,
            matched_budget=not self.budget.unmatched,
        )

    def record_outcome(self, decision: Decision, ok: bool,
                       ttft_ms: float | None) -> None:
        if decision.route == "cloud":
            self.circuit.record(ok, ttft_ms)

    def _local_state(self, feat: dict, executable: bool) -> str:
        if not executable:
            return "unavailable"
        pending = feat.get("pending_local")
        if _valid(pending) and float(pending) >= self.cfg.governor.hard_local_pending:
            return "hard_breach"
        tail = feat.get("roll_ttft_p95")
        if _valid(tail) and float(tail) > self.cfg.targets.sla_ttft_ms:
            return "hard_breach"
        if not _valid(pending):
            return "unknown"
        return "healthy"

    def _telemetry_tier(self, feat: dict) -> str:
        if all(_valid(feat.get(k)) for k in self.cfg.governor.required_full_telemetry):
            return "full"
        tail_fields = ["pending_local", "active_local", "roll_ttft_p95"]
        if all(_valid(feat.get(k)) for k in tail_fields):
            return "tail"
        if all(_valid(feat.get(k)) for k in ["pending_local", "active_local"]):
            return "minimal"
        return "unusable"
