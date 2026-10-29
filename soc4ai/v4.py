"""Candidate v4: v3 (paced quota, fresh override) plus quota exemption for
breaker probes and refunds for failed cloud calls.

Diagnosis from chaos experiment B: failed overrides and half-open probes were
charged against the quota, so an outage drained it and the breaker stayed
half-open with no budget left to probe. v4 changes only that accounting:
- a cloud call admitted while the breaker is half-open is a probe: it is still
  bounded by the breaker (one at a time) but is not charged to the quota;
- a charged cloud call that fails is refunded to the quota.
Spend is still bounded by the breaker and by the study's USD kill switch.
"""
from __future__ import annotations

from apex.core import Config, _repo_root
from apex.policies import GovernorArtifact, all_policies

from soc4ai.paced import FreshTailAware, PacedGovernor, allowance_for

PROBE_TAG = "probe_exempt"


class V4Governor(PacedGovernor):
    name = "v4_governor"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.budget.refunds = 0
        self.budget.exempt_probes = 0

    def _cloud_or_local(self, reason, score, telemetry_tier, *, safety=False, regime=None,
                        expert=None, proposal=None):
        if self.circuit.state == "half-open":
            if not self.circuit.begin_attempt():
                return self._final("local", f"{reason}:circuit_unavailable", score, telemetry_tier,
                                   regime=regime, expert=expert, proposal=proposal, safety=True)
            self.budget.exempt_probes += 1
            return self._final("cloud", f"{reason}:{PROBE_TAG}", score, telemetry_tier,
                               regime=regime, expert=expert, proposal=proposal, safety=safety,
                               discretionary=not safety)
        return super()._cloud_or_local(reason, score, telemetry_tier, safety=safety, regime=regime,
                                       expert=expert, proposal=proposal)

    def record_outcome(self, decision, ok, ttft_ms):
        super().record_outcome(decision, ok, ttft_ms)
        if decision.route == "cloud" and not ok and PROBE_TAG not in (decision.reason_code or ""):
            if self.budget.total_cloud > 0:
                self.budget.total_cloud -= 1
                self.budget.refunds += 1


def make_v4_governor(cfg: Config, *, episode_requests: int, frozen_cloud_p95_ms: float,
                     heuristic_weights: dict | None = None, local_executable=None) -> V4Governor:
    artifact = GovernorArtifact.load(_repo_root() / cfg.governor.artifact_path)
    experts = all_policies(cfg, heuristic_weights)
    experts["tail_aware"] = FreshTailAware(cfg, heuristic_weights)
    return V4Governor(cfg, artifact, experts, episode_requests=episode_requests,
                      frozen_cloud_p95_ms=frozen_cloud_p95_ms,
                      allowance=allowance_for(episode_requests),
                      local_executable=local_executable)
