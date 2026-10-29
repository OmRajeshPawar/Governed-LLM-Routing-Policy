# Signal-cost policy comparison @ matched ~30% offload budget

SLA = 3840 ms. Always-Local/Cloud shown at their fixed operating points.
Adjusted metrics add decision/probe overhead and router cost. T2-T4 remain candidate/proxy policy tiers; a separate bounded live probe-interference audit now measures the T3 mechanism penalty, but T3/T4 are not full-scale live policy winners.

| policy | tier | offload % | raw SLA viol % | adj SLA viol % | adj P99 TTFT ms | all-in cost $ | $/adj SLA-ok req |
|---|---:|---:|---:|---:|---:|---:|---:|
| always_local | 0 | 0 | 35.0 | 35.0 | 29741 | 0.000 | 0.0000 |
| always_cloud | 0 | 100 | 0.0 | 0.0 | 378 | 820.604 | 0.1824 |
| queue_threshold | 0 | 30 | 7.4 | 7.4 | 6659 | 281.923 | 0.0677 |
| tail_aware | 0 | 30 | 6.5 | 6.5 | 5812 | 302.948 | 0.0720 |
| ml_ttft | 1 | 30 | 6.6 | 6.6 | 5812 | 303.375 | 0.0722 |
| ml_dual_risk | 1 | 30 | 6.9 | 6.9 | 6182 | 314.830 | 0.0752 |
| semantic_embedding | 2 | 30 | 6.7 | 6.7 | 5999 | 307.214 | 0.0732 |
| local_llm_probe | 3 | 30 | 6.9 | 8.2 | 6328 | 311.512 | 0.0754 |
| thinking_probe | 4 | 30 | 7.0 | 11.2 | 6868 | 315.769 | 0.0790 |
