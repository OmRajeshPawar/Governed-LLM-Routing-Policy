# Live co-located probe interference audit

This is a bounded live A/B audit, not the full 4,500-request study. It replays identical local-only traces twice: once with no routing probe and once where each request first runs a short local LLM probe on the same Ollama model.

- Seeds: `7, 13, 42`
- Requests per condition per seed: `6`
- User max tokens: `24`
- Probe tokens: `8`
- Arrival spacing: `0.25s`
- Runtime: `21.7s`

## Aggregate results

| condition | user requests | probes | SLA viol % | mean TTFT ms | P95 TTFT ms | P99 TTFT ms | mean queue wait ms | P95 queue wait ms | mean probe wall ms | P95 probe wall ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | 18 | 0 | 0.0 | 543 | 899 | 927 | 166 | 446 | 0 | 0 |
| co_located_probe | 18 | 18 | 0.0 | 892 | 1352 | 1377 | 553 | 932 | 473 | 847 |

## Mean paired delta (co-located probe minus baseline)

| metric | delta |
|---|---:|
| SLA violation | 0.0 points |
| mean TTFT | 349 ms |
| P95 TTFT | 453 ms |
| P99 TTFT | 450 ms |
| mean queue wait | 386 ms |
| P95 queue wait | 487 ms |
| cumulative queue wait per trace | 2317 ms |
| mean probe wall latency | 473 ms |
| P95 probe wall latency | 847 ms |

## Interpretation

A positive delta means the co-located local probe made user-facing local service worse on the same trace. It measures a mechanism behind the signal-cost frontier: a probe can add direct decision latency and contend with the constrained local LLM resource.

## Scope

A controlled live A/B on the testbed GPU that isolates the probe-contention mechanism behind the signal-cost frontier. The policy comparison itself comes from the 4,500-request replay corpus.
