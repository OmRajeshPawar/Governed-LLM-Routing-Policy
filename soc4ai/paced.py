"""Exploratory candidate v3: paced cloud quota and a freshness-aware hard override.

Diagnosis from the preregistered live test (experiment A): the 30% hard quota
was spent by request ~96-144 of 300, 86-89 of 90 cloud calls being safety
overrides, because the tail-aware hard override (rolling local P95 above the
SLO) stays true while requests are offloaded and the local window never
refreshes. After exhaustion every request queued locally.

Candidate changes (nothing else differs from the v2 governor):
1. PacedCloudBudget releases cloud calls in proportion to arrivals:
   released = min(cap, floor(fraction * arrivals) + allowance).
2. A stale rolling P95 no longer forces the cloud when the local tier has free
   capacity (no pending requests and a free concurrency slot). The queue hard
   cap still forces a hard breach.
This module has no dependency on the study driver, so tests can import it.
"""
from __future__ import annotations

import math

from apex.core import Config, _repo_root
from apex.policies import (GovernorArtifact, GuardedGovernor, TailAwareHeuristic,
                           TotalCloudBudget, _nz, _valid, all_policies)


def local_busy(cfg: Config, feat: dict) -> bool:
    pending, active = feat.get("pending_local"), feat.get("active_local")
    if not _valid(pending) and not _valid(active):
        return True  # cannot observe the local tier: keep the conservative override
    return ((_valid(pending) and float(pending) >= 1)
            or (_valid(active) and float(active) >= cfg.local_concurrency))


class PacedCloudBudget(TotalCloudBudget):

    def __init__(self, episode_requests: int, fraction: float = 0.30,
                 mode: str = "evaluation_hard", allowance: int = 0):
        super().__init__(episode_requests, fraction, mode)
        self.fraction = float(fraction)
        self.allowance = int(allowance)
        self.arrivals = 0

    def on_arrival(self) -> None:
        self.arrivals += 1

    @property
    def released(self) -> int:
        return min(self.cap, math.floor(self.fraction * self.arrivals) + self.allowance)

    def request(self, *, discretionary: bool = False, safety_override: bool = False,
                probe: bool = False) -> tuple[bool, str]:
        if self.total_cloud < self.released:
            self._consume(discretionary, safety_override, probe)
            return True, "budget_admitted"
        self.denials += 1
        return False, "budget_paced_denied"


class FreshTailAware(TailAwareHeuristic):

    def decide(self, feat, threshold=None, ml=None):
        thr = self.cfg.policies.tail_aware_pressure_threshold if threshold is None else threshold
        s = self.score(feat)
        if _nz(feat["roll_ttft_p95"]) > self.cfg.targets.sla_ttft_ms and local_busy(self.cfg, feat):
            return self._decision("cloud", "tail_hard_breach", s)
        route = "cloud" if s >= thr else "local"
        return self._decision(route, "tail_pressure" if route == "cloud" else "tail_ok", s)


class PacedGovernor(GuardedGovernor):
    name = "paced_governor"

    def __init__(self, cfg: Config, artifact: GovernorArtifact, experts: dict, *,
                 episode_requests: int, frozen_cloud_p95_ms: float, allowance: int,
                 local_executable=None):
        super().__init__(cfg, artifact, experts, episode_requests=episode_requests,
                         frozen_cloud_p95_ms=frozen_cloud_p95_ms,
                         local_executable=local_executable)
        b = cfg.governor.budget
        self.budget = PacedCloudBudget(episode_requests, b.target_fraction, b.mode, allowance)

    def decide(self, feat: dict, threshold: float = 0.0, ml=None):
        self.budget.on_arrival()
        return super().decide(feat, threshold, ml)

    def _local_state(self, feat: dict, executable: bool) -> str:
        state = super()._local_state(feat, executable)
        if state != "hard_breach":
            return state
        pending = feat.get("pending_local")
        if _valid(pending) and float(pending) >= self.cfg.governor.hard_local_pending:
            return state
        return state if local_busy(self.cfg, feat) else "healthy"


def allowance_for(episode_requests: int) -> int:
    return math.ceil(0.05 * episode_requests)


def make_paced_governor(cfg: Config, *, episode_requests: int, frozen_cloud_p95_ms: float,
                        heuristic_weights: dict | None = None, local_executable=None) -> PacedGovernor:
    artifact = GovernorArtifact.load(_repo_root() / cfg.governor.artifact_path)
    experts = all_policies(cfg, heuristic_weights)
    experts["tail_aware"] = FreshTailAware(cfg, heuristic_weights)
    return PacedGovernor(cfg, artifact, experts, episode_requests=episode_requests,
                         frozen_cloud_p95_ms=frozen_cloud_p95_ms,
                         allowance=allowance_for(episode_requests),
                         local_executable=local_executable)
