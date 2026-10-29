from __future__ import annotations

import random

TEMPLATES = {
    "qa": "Answer concisely: what causes {topic}?",
    "summarize": "Summarize the following text in 3 bullet points:\n{filler}",
    "extract_json": "Extract the entities from this text as JSON with keys name, date, amount:\n{filler}",
    "code": "Write a Python function that {task}. Include a short docstring.",
    "longform": "Write a detailed multi-paragraph explanation of {topic}, with examples.",
}
TOPICS = ["queue latency", "GPU memory pressure", "tail latency", "burst traffic",
          "cache eviction", "head-of-line blocking", "throughput collapse"]
TASKS = ["sorts a list of integers", "reverses a linked list", "parses a CSV row",
         "computes a rolling median", "validates a JSON schema"]
_FILLER = ("Service requests arrive in bursts and contend for limited capacity. " * 40)
REAL_SAMPLE_PROMPTS = [
    ("qa", "A customer asks why their local AI assistant is slow only during team meetings. Explain the likely queueing cause."),
    ("summarize", "Summarize this incident note for an operations lead:\n" + _FILLER[:900]),
    ("extract_json", "Extract name, date, amount, and service tier as JSON from: Om Pawar renewed the Vertex AI quota on 2026-06-28 for 45.20 USD."),
    ("code", "Write a Python function that groups request latencies into P50, P95, and P99 buckets. Include input validation."),
    ("longform", "Write a practical explanation of when a local-first LLM gateway should overflow to cloud, with cost and privacy tradeoffs."),
]

SIZES = {
    "short": (120, 32),
    "medium": (700, 96),
    "long": (1800, 192),
}
SIZE_MIX = {
    "qa": ("short", "short", "medium"),
    "summarize": ("medium", "long", "medium"),
    "extract_json": ("medium", "short", "medium"),
    "code": ("short", "medium", "medium"),
    "longform": ("long", "long", "medium"),
}
CATEGORY_MIX = {
    "synthetic": ["qa", "summarize", "extract_json", "code", "longform"],
    "real_sample": ["qa", "summarize", "extract_json", "code", "longform"],
    "reasoning_coding": ["code", "code", "longform", "longform", "qa"],
}
SIZE_MIX_BY_WORKLOAD = {
    "synthetic": SIZE_MIX,
    "real_sample": SIZE_MIX,
    "reasoning_coding": {
        "qa": ("medium", "long", "medium"),
        "summarize": ("long", "medium", "long"),
        "extract_json": ("medium", "medium", "long"),
        "code": ("medium", "long", "long"),
        "longform": ("long", "long", "medium"),
    },
}


def _make_prompt(rng: random.Random, category: str, size: str) -> str:
    target_chars, _ = SIZES[size]
    base = TEMPLATES[category].format(
        topic=rng.choice(TOPICS), task=rng.choice(TASKS),
        filler=_FILLER[: max(0, target_chars)],
    )
    if len(base) < target_chars:
        base = base + "\nContext: " + _FILLER[: target_chars - len(base)]
    return base


def make_request_spec(rng: random.Random, idx: int, mix: str = "synthetic") -> dict:
    category = rng.choice(CATEGORY_MIX.get(mix, CATEGORY_MIX["synthetic"]))
    size = rng.choice(SIZE_MIX_BY_WORKLOAD.get(mix, SIZE_MIX_BY_WORKLOAD["synthetic"])[category])
    _, out_cap = SIZES[size]
    if mix == "real_sample":
        category, prompt = REAL_SAMPLE_PROMPTS[idx % len(REAL_SAMPLE_PROMPTS)]
        if len(prompt) < SIZES[size][0]:
            prompt = prompt + "\nContext: " + _FILLER[: SIZES[size][0] - len(prompt)]
    else:
        prompt = _make_prompt(rng, category, size)
    return {
        "request_id": f"req-{idx:05d}",
        "prompt": prompt,
        "max_tokens": out_cap,
        "task_category": category,
        "size": size,
        "mix": mix,
    }


def _arrival_offsets(rng: random.Random, n: int, pattern: str) -> list[float]:
    offsets, t = [], 0.0
    for i in range(n):
        if pattern == "steady":
            rate = 0.5
        elif pattern == "saturation":
            rate = 1.2
        elif pattern == "bursty":
            rate = 1.5 if (i % 24) < 6 else 0.3
        else:
            rate = 0.6
        t += rng.expovariate(rate)
        offsets.append(t)
    return offsets


def make_trace(n: int = 80, pattern: str = "bursty", seed: int = 42,
               mix: str = "synthetic") -> list[dict]:
    rng = random.Random(seed)
    offsets = _arrival_offsets(rng, n, pattern)
    trace = []
    for i in range(n):
        spec = make_request_spec(rng, i, mix=mix)
        spec["arrival_offset"] = offsets[i]
        spec["trace_pos"] = i
        trace.append(spec)
    return trace


def make_transition_trace(seed: int, n: int = 200) -> list[dict]:
    if n != 200:
        raise ValueError("The preregistered transition trace contains exactly 200 requests")
    rng = random.Random(seed)
    phases = [
        ("low_factual", 40, 0.3, "synthetic", False),
        ("short_burst", 40, 1.8, "synthetic", False),
        ("reasoning_heavy", 60, 0.9, "reasoning_coding", False),
        ("recovery", 20, 0.3, "synthetic", False),
        ("cloud_degradation", 40, 0.6, "real_sample", True),
    ]
    trace, t, i = [], 0.0, 0
    for phase, count, rate, mix, degraded in phases:
        for _ in range(count):
            t += rng.expovariate(rate)
            spec = make_request_spec(rng, i, mix=mix)
            if phase in {"low_factual", "short_burst", "recovery"}:
                spec["task_category"] = "qa"
                spec["prompt"] = _make_prompt(rng, "qa", "short")
                spec["max_tokens"] = SIZES["short"][1]
                spec["size"] = "short"
            spec.update({
                "arrival_offset": t,
                "trace_pos": i,
                "workload_phase": phase,
                "cloud_degraded": degraded,
                "transition_seed": seed,
            })
            trace.append(spec)
            i += 1
    return trace
