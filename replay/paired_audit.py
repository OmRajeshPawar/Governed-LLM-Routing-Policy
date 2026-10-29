from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import load_config, ensure_dirs, _repo_root
from apex.events import EventLog
from apex.core import Request
from apex.gateway import Gateway
from apex.policies import AlwaysLocal
from apex.runner import build_providers
from apex.core import GatewayState
from apex.workload import _make_prompt
import random

HEAVY_ID = "HEAVY"


def audit_trace() -> list[dict]:
    rng = random.Random(123)
    trace = [
        {"request_id": "filler-0", "task_category": "summarize", "size": "medium",
         "max_tokens": 96, "arrival_offset": 0.0},
        {"request_id": HEAVY_ID, "task_category": "longform", "size": "long",
         "max_tokens": 256, "arrival_offset": 0.05},
    ]
    for i in range(6):
        trace.append({"request_id": f"follow-{i}", "task_category": "qa",
                      "size": "short", "max_tokens": 32,
                      "arrival_offset": 0.30 + i * 0.15})
    for spec in trace:
        spec["prompt"] = _make_prompt(rng, spec["task_category"], spec["size"])
        spec["trace_pos"] = 0
    return trace


async def run_condition(cfg, trace, local, cloud, offload_heavy: bool, tag: str):
    state = GatewayState(cfg)
    out = _repo_root() / cfg.paths.events_dir / f"audit_{tag}.jsonl"
    log = EventLog(out); await log.start()
    gw = Gateway(cfg, local, cloud, state, log, run_id=f"audit_{tag}", trace_id="audit")
    pol = AlwaysLocal()

    t0 = time.perf_counter()
    tasks, follower_results = [], {}
    for spec in trace:
        delay = (t0 + spec["arrival_offset"]) - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        req = Request(spec["request_id"], spec["prompt"], spec["max_tokens"],
                      spec["task_category"], time.perf_counter())
        route = "cloud" if (offload_heavy and spec["request_id"] == HEAVY_ID) else "local"

        async def _go(r=req, ro=route):
            res = await gw.handle(r, pol, threshold=0.0, route_override=ro)
            return r.request_id, res
        tasks.append(asyncio.create_task(_go()))
    for coro in asyncio.as_completed(tasks):
        rid, res = await coro
        if rid.startswith("follow-"):
            follower_results[rid] = res["outcome"]
    await log.aclose()
    return follower_results


def summarize(followers: dict, sla: float) -> dict:
    qwaits = [f["queue_wait_ms"] for f in followers.values()]
    ttfts = [f["end_to_end_ttft_ms"] for f in followers.values() if f["end_to_end_ttft_ms"]]
    breaches = sum(1 for t in ttfts if t > sla)
    return {"n": len(followers), "sum_queue_wait_ms": sum(qwaits),
            "mean_ttft_ms": sum(ttfts) / len(ttfts) if ttfts else 0.0,
            "breaches": breaches}


async def main():
    cfg = load_config()
    ensure_dirs(cfg)
    sla = cfg.targets.sla_ttft_ms
    trace = audit_trace()
    local, cloud, calib = await build_providers(cfg)
    await local.warmup()

    print("Paired occupancy-harm audit (same scenario, heavy request admitted vs offloaded)\n")
    admit = summarize(await run_condition(cfg, trace, local, cloud, False, "admit"), sla)
    await asyncio.sleep(1.0)
    offload = summarize(await run_condition(cfg, trace, local, cloud, True, "offload"), sla)

    print(f"{'downstream metric (6 followers)':36} {'ADMIT heavy':>12} {'OFFLOAD heavy':>14} {'delta':>10}")
    for key, label in [("sum_queue_wait_ms", "cumulative queue wait (ms)"),
                       ("mean_ttft_ms", "mean follower TTFT (ms)"),
                       ("breaches", "follower SLA breaches")]:
        a, o = admit[key], offload[key]
        print(f"{label:36} {a:>12.0f} {o:>14.0f} {a-o:>+10.0f}")
    print(f"\n=> Admitting the heavy request locally added {admit['sum_queue_wait_ms']-offload['sum_queue_wait_ms']:.0f} ms "
          f"of cumulative downstream queue delay and {admit['breaches']-offload['breaches']:+d} follower breaches.")
    print("   This is the occupancy harm the dual-risk model exists to anticipate (proposal N3).")


if __name__ == "__main__":
    asyncio.run(main())
