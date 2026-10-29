"""Providers for the SOC4AI live study.

- TimedOllamaProvider: the repo's local provider, plus Ollama's own
  load/prefill/decode timings attached as `extra`.
- LiveGeminiProvider: real Gemini 2.5 Flash on Vertex AI (streaming), spend
  ledger with kill switch, credential guard.
- FaultInjectingProvider: outage and latency faults in request-index windows.
"""
from __future__ import annotations

import asyncio
import contextvars
import time

from apex.providers import OllamaProvider, ProviderResult

from . import creds
from .ledger import BudgetKillSwitch, SpendLedger, estimate_cost, worst_case_cost

CURRENT_TRACE_POS: contextvars.ContextVar = contextvars.ContextVar("trace_pos", default=None)
CURRENT_DEGRADED: contextvars.ContextVar = contextvars.ContextVar("cloud_degraded", default=False)


def _ns_ms(value) -> float | None:
    return float(value) / 1e6 if value else None


class TimedOllamaProvider(OllamaProvider):

    async def generate(self, prompt: str, max_tokens: int) -> ProviderResult:
        if self.extra_latency_ms:
            await asyncio.sleep(self.extra_latency_ms / 1000.0)
        t_start = time.perf_counter()
        t_first: float | None = None
        chars = prompt_tokens = output_tokens = 0
        meta: dict = {}
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
                    meta = {
                        "load_ms": _ns_ms(getattr(chunk, "load_duration", None)),
                        "prompt_eval_ms": _ns_ms(getattr(chunk, "prompt_eval_duration", None)),
                        "eval_ms": _ns_ms(getattr(chunk, "eval_duration", None)),
                        "total_ms": _ns_ms(getattr(chunk, "total_duration", None)),
                        "done_reason": getattr(chunk, "done_reason", None),
                    }
            t_done = time.perf_counter()
        except Exception as exc:
            t_done = time.perf_counter()
            res = ProviderResult(self.role, self.model, False, str(exc), t_start,
                                 t_first, t_done, None, (t_done - t_start) * 1000.0,
                                 0, 0, 0.0, chars)
            res.extra = {"error_class": type(exc).__name__}
            return res
        cost = (prompt_tokens + output_tokens) / 1000.0 * self.cost_per_1k_tokens
        res = ProviderResult(
            self.role, self.model, True, None, t_start, t_first, t_done,
            ((t_first - t_start) * 1000.0) if t_first else None,
            (t_done - t_start) * 1000.0, prompt_tokens, output_tokens, cost, chars)
        pe, ev = meta.get("prompt_eval_ms"), meta.get("eval_ms")
        meta["prefill_tokens_per_s"] = prompt_tokens / (pe / 1000.0) if pe else None
        meta["decode_tokens_per_s"] = output_tokens / (ev / 1000.0) if ev else None
        res.extra = meta
        return res


class LiveGeminiProvider:
    role = "cloud"

    def __init__(self, ledger: SpendLedger, *, request_timeout_s: float = 60.0,
                 client=None):
        creds.assert_credentials()
        from google.genai import types
        self._types = types
        if client is None:
            from google import genai
            client = genai.Client(vertexai=True, project=creds.PROJECT, location=creds.LOCATION)
        self.client = client
        self.ledger = ledger
        self.model = creds.MODEL
        self.timeout = float(request_timeout_s)

    async def warmup(self) -> float:
        return 0.0

    async def generate(self, prompt: str, max_tokens: int) -> ProviderResult:
        creds.assert_credentials()
        worst = worst_case_cost(prompt, max_tokens)
        rid = self.ledger.reserve(worst)  # raises BudgetKillSwitch; no await before this line
        ledger_before = self.ledger.committed
        config = self._types.GenerateContentConfig(
            max_output_tokens=int(max_tokens), temperature=0.0,
            thinking_config=self._types.ThinkingConfig(thinking_budget=0))
        t_start = time.perf_counter()
        state = {"first": None, "prev": None, "last": None, "chars": 0, "gaps": []}
        ok, err, err_class = True, None, None

        async def consume():
            stream = await self.client.aio.models.generate_content_stream(
                model=self.model, contents=prompt, config=config)
            async for chunk in stream:
                now = time.perf_counter()
                state["last"] = chunk
                text = getattr(chunk, "text", "") or ""
                if text:
                    if state["first"] is None:
                        state["first"] = now
                    elif state["prev"] is not None:
                        state["gaps"].append((now - state["prev"]) * 1000.0)
                    state["prev"] = now
                    state["chars"] += len(text)

        try:
            await asyncio.wait_for(consume(), timeout=self.timeout)
        except BudgetKillSwitch:
            raise
        except Exception as exc:
            ok, err, err_class = False, f"{type(exc).__name__}: {str(exc)[:300]}", type(exc).__name__
        t_done = time.perf_counter()
        first, prev, last, gaps = state["first"], state["prev"], state["last"], state["gaps"]
        usage = getattr(last, "usage_metadata", None)
        pt = int(getattr(usage, "prompt_token_count", 0) or 0)
        ot = int(getattr(usage, "candidates_token_count", 0) or 0)
        th = int(getattr(usage, "thoughts_token_count", 0) or 0)
        if usage is not None:
            actual = estimate_cost(pt, ot + th)
        elif first is None and not ok:
            actual = 0.0
        else:
            actual = worst
        self.ledger.commit(rid, actual)
        if ok and first is None:
            ok, err, err_class = False, "empty_stream", "EmptyStream"
        finish = None
        try:
            if last is not None and getattr(last, "candidates", None):
                finish = str(last.candidates[0].finish_reason)
        except Exception:
            finish = None
        res = ProviderResult(
            "cloud", self.model, ok, err, t_start, first, t_done,
            ((first - t_start) * 1000.0) if first else None,
            (t_done - t_start) * 1000.0, pt, ot, actual, state["chars"])
        res.extra = {
            "cloud_provider": "vertex_gemini_live",
            "stream_ttft_ms": ((first - t_start) * 1000.0) if first else None,
            "time_to_last_token_ms": ((prev - t_start) * 1000.0) if prev else None,
            "inter_chunk_gap_ms_mean": (sum(gaps) / len(gaps)) if gaps else None,
            "inter_chunk_gap_ms_max": max(gaps) if gaps else None,
            "n_text_chunks": len(gaps) + (1 if first else 0),
            "thought_tokens": th, "finish_reason": finish, "error_class": err_class,
            "worst_case_reserved_usd": worst, "estimated_cost_usd": actual,
            "ledger_committed_before_usd": ledger_before,
            "ledger_committed_after_usd": self.ledger.committed,
        }
        return res


class FaultInjectingProvider:
    """Injects faults by trace position (read from a contextvar set per request).

    outage: [start, end) -> immediate failure after `outage_latency_ms`, no call
    spike:  [start, end) -> fixed additive delay before a real call
    degraded_delay_ms: additive delay for requests flagged cloud_degraded
    """
    role = "cloud"

    def __init__(self, inner, *, outage=None, spike=None, spike_delay_ms: float = 0.0,
                 degraded_delay_ms: float = 0.0, outage_latency_ms: float = 250.0):
        self.inner = inner
        self.outage = tuple(outage) if outage else None
        self.spike = tuple(spike) if spike else None
        self.spike_delay_ms = float(spike_delay_ms)
        self.degraded_delay_ms = float(degraded_delay_ms)
        self.outage_latency_ms = float(outage_latency_ms)
        self.model = getattr(inner, "model", "cloud")
        self.injected = {"outage": 0, "latency_spike": 0, "degraded_phase": 0}

    @staticmethod
    def _inside(window, pos) -> bool:
        return window is not None and pos is not None and window[0] <= pos < window[1]

    async def warmup(self) -> float:
        return 0.0

    async def generate(self, prompt: str, max_tokens: int) -> ProviderResult:
        pos = CURRENT_TRACE_POS.get()
        if self._inside(self.outage, pos):
            t0 = time.perf_counter()
            await asyncio.sleep(self.outage_latency_ms / 1000.0)
            t1 = time.perf_counter()
            self.injected["outage"] += 1
            res = ProviderResult("cloud", self.model, False,
                                 "injected_outage: 503 service unavailable",
                                 t0, None, t1, None, (t1 - t0) * 1000.0, 0, 0, 0.0, 0)
            res.extra = {"fault": "outage", "error_class": "InjectedOutage"}
            return res
        delay, fault = 0.0, None
        if self._inside(self.spike, pos):
            delay, fault = self.spike_delay_ms, "latency_spike"
        elif CURRENT_DEGRADED.get() and self.degraded_delay_ms > 0:
            delay, fault = self.degraded_delay_ms, "degraded_phase"
        if delay <= 0:
            return await self.inner.generate(prompt, max_tokens)
        t0 = time.perf_counter()
        await asyncio.sleep(delay / 1000.0)
        res = await self.inner.generate(prompt, max_tokens)
        self.injected[fault] += 1
        res.t_provider_start = t0
        if res.t_first_text is not None:
            res.provider_ttft_ms = (res.t_first_text - t0) * 1000.0
        if res.t_completed is not None:
            res.service_ms = (res.t_completed - t0) * 1000.0
        res.extra = {**(getattr(res, "extra", None) or {}), "fault": fault, "injected_delay_ms": delay}
        return res
