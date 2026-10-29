# Deep-dive insights from the 4,500-request target run

These are derived metrics from existing data. A later bounded live
probe-interference audit is reported separately in
`analysis/probe_interference_summary.md`.

## Main additional findings

1. Telemetry is the best deployable policy in queue replay at 30% offload.
   Tail-aware telemetry reaches 10.1% SLA violations, against ML dual-risk at
   10.4%, semantic/embedding at 10.2%, and local probe at 10.5%.
2. The oracle gap is real but mostly uncaptured by the richer signals. The
   service-time oracle reaches 6.3% at 30% offload. ML dual-risk captures
   -8.8% of the T0-to-oracle headroom and semantic/embedding captures -1.8%.
   Negative values mean worse than T0.
3. Dual-risk is not better than TTFT-only here. Static ML-TTFT is 6.6%
   adjusted SLA violation, ML dual-risk is 6.9%. The occupancy model detects
   a real phenomenon, but its current score is not improving routing.
4. T3 and T4 lose for an explainable systems reason. Static local-probe raw
   SLA is 6.9% but adjusted SLA is 8.2%; thinking-probe adjusted SLA rises to
   11.2%. The probe tax cancels the apparent signal value.
5. Live probe interference adds mechanism evidence. The bounded A/B audit
   shows a co-located local probe adding 349 ms mean TTFT, 453 ms P95 TTFT,
   and 386 ms mean queue wait on identical traces. This supports the
   signal-cost mechanism but remains a small controlled audit.

## New figures

- `analysis/figures/queue_replay_budget_curves.png`
- `analysis/figures/headroom_captured_by_budget.png`
- `analysis/figures/workload_breakdown_30pct.png`
- `analysis/figures/signal_tax_scatter.png`
- `analysis/figures/probe_interference_ab.png`
