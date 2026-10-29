import asyncio
import json
import shutil
import sys
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from soc4ai import creds
from soc4ai.ledger import BudgetKillSwitch, SpendLedger, worst_case_cost
from soc4ai.providers import (CURRENT_TRACE_POS, FaultInjectingProvider,
                              LiveGeminiProvider)
from apex.providers import ProviderResult


@pytest.fixture
def workdir():
    d = ROOT / "soc4ai_out" / "test_tmp" / uuid.uuid4().hex
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_kill_switch_blocks_a_call_that_would_cross_the_cap(workdir):
    led = SpendLedger(workdir / "ledger.json", cap_usd=1.0)
    led.add_external(0.99)
    with pytest.raises(BudgetKillSwitch):
        led.reserve(0.02)
    assert led.halted
    reopened = SpendLedger(workdir / "ledger.json", cap_usd=1.0)
    assert reopened.halted
    with pytest.raises(BudgetKillSwitch):
        reopened.reserve(0.0001)


def test_unreconciled_reservation_counts_as_spent_after_crash(workdir):
    p = workdir / "ledger.json"
    p.write_text(json.dumps({"committed_usd": 1.0, "reserved_usd": 0.5, "n_calls": 3, "halted": False}))
    led = SpendLedger(p, cap_usd=20.0)
    assert led.committed == pytest.approx(1.5)


def test_commit_that_reaches_the_cap_halts_the_ledger(workdir):
    led = SpendLedger(workdir / "ledger.json", cap_usd=1.0)
    rid = led.reserve(0.1)
    led.commit(rid, 1.2)  # actual cost exceeded its reservation
    assert led.halted
    with pytest.raises(BudgetKillSwitch):
        led.reserve(0.0001)


class _FakeStream:
    def __init__(self, delay, output_tokens):
        self.delay = delay
        self.output_tokens = output_tokens

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        await asyncio.sleep(self.delay)
        usage = SimpleNamespace(prompt_token_count=60, candidates_token_count=self.output_tokens,
                                thoughts_token_count=0)
        yield SimpleNamespace(text="hello", usage_metadata=usage, candidates=None)


def _fake_client(delay=0.05):
    async def generate_content_stream(model, contents, config):
        # A real model never exceeds max_output_tokens.
        return _FakeStream(delay, int(config.max_output_tokens))
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content_stream=generate_content_stream)))


def test_kill_switch_holds_under_concurrent_calls(workdir, monkeypatch):
    monkeypatch.setattr(creds, "assert_credentials", lambda: None)
    cap = 0.05
    led = SpendLedger(workdir / "ledger.json", cap_usd=cap)
    prov = LiveGeminiProvider(led, client=_fake_client(0.05))
    prompt = "x" * 200

    async def main():
        return await asyncio.gather(*(prov.generate(prompt, 256) for _ in range(200)),
                                    return_exceptions=True)

    results = asyncio.run(main())
    kills = [r for r in results if isinstance(r, BudgetKillSwitch)]
    oks = [r for r in results if isinstance(r, ProviderResult)]
    assert kills, "kill switch never fired"
    assert led.total < cap
    assert led.halted
    assert len(oks) <= int(cap / worst_case_cost(prompt, 256)) + 1


def test_live_provider_refuses_a_key_from_another_project(workdir, monkeypatch):
    fake = workdir / "other_project_account.json"
    fake.write_text(json.dumps({"kind": "fake_account", "project_id": "some-other-project"}))
    monkeypatch.setenv(creds.KEY_ENV, str(fake))
    monkeypatch.setenv(creds.PROJECT_ENV, "expected-project")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(fake))
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "expected-project")
    monkeypatch.setenv("GCLOUD_PROJECT", "expected-project")
    with pytest.raises(creds.CredentialGuardError):
        LiveGeminiProvider(SpendLedger(workdir / "l.json"), client=_fake_client())


class _Inner:
    model = "fake"

    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, max_tokens):
        self.calls += 1
        t = time.perf_counter()
        res = ProviderResult("cloud", "fake", True, None, t, t + 0.1, t + 0.2, 100.0, 200.0, 1, 1, 0.0, 5)
        res.extra = {"cloud_provider": "fake"}
        return res


def test_tracing_breaker_logs_open_half_open_and_close():
    from apex.core import load_config
    from apex.policies import make_guarded_governor
    from soc4ai.telemetry import attach_tracing_breaker
    cfg = load_config()
    gov = make_guarded_governor(cfg, episode_requests=100, frozen_cloud_p95_ms=1000.0)
    attach_tracing_breaker(gov)
    br = gov.circuit
    for i in range(cfg.governor.circuit.consecutive_failure_limit):
        br.record(False, None, now=float(i))
    assert br.state == "open"
    for i in range(cfg.governor.circuit.cooldown_arrivals):
        br.on_arrival(now=1000.0 + i)
    assert br.state == "half-open"
    for _ in range(cfg.governor.circuit.half_open_successes_to_close):
        assert br.begin_attempt()
        br.record(True, 500.0, now=2000.0)
    assert br.state == "closed"
    assert [(t["from"], t["to"]) for t in br.transitions] == [
        ("closed", "open"), ("open", "half-open"), ("half-open", "closed")]
    assert br.transitions[0]["reason"] == "circuit_consecutive_failures"


def test_fault_windows_by_trace_position():
    inner = _Inner()
    fp = FaultInjectingProvider(inner, outage=(10, 20), spike=(30, 40),
                                spike_delay_ms=50.0, outage_latency_ms=1.0)

    async def call(pos):
        CURRENT_TRACE_POS.set(pos)
        return await fp.generate("p", 8)

    out = asyncio.run(call(15))
    assert out.ok is False and out.extra["fault"] == "outage" and inner.calls == 0
    spike = asyncio.run(call(35))
    assert spike.ok and spike.extra["fault"] == "latency_spike" and inner.calls == 1
    assert spike.provider_ttft_ms >= 50.0
    normal = asyncio.run(call(5))
    assert normal.extra.get("fault") is None and inner.calls == 2
