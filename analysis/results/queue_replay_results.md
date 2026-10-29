# Queue replay policy comparison

Static queue replay on identical arrival traces. Metrics are averaged across shadow runs.

| policy | budget | SLA viol % | P95 TTFT ms | P99 TTFT ms | mean queue wait ms | all-in cost $ |
|---|---:|---:|---:|---:|---:|---:|
| always_cloud | 10% | 0.0 | 336 | 372 | 0 | 54.707 |
| always_local | 10% | 35.0 | 12417 | 15829 | 3096 | 0.000 |
| local_llm_probe | 10% | 23.3 | 8211 | 10375 | 1613 | 5.694 |
| ml_dual_risk | 10% | 22.1 | 8407 | 10968 | 1633 | 5.539 |
| ml_ttft | 10% | 22.0 | 8180 | 10645 | 1618 | 5.470 |
| queue_service_oracle | 10% | 18.5 | 7359 | 10305 | 1434 | 8.106 |
| queue_threshold | 10% | 22.6 | 8078 | 10228 | 1652 | 5.285 |
| semantic_embedding | 10% | 22.4 | 8002 | 10139 | 1616 | 5.827 |
| tail_aware | 10% | 22.5 | 8037 | 10142 | 1632 | 5.665 |
| thinking_probe | 10% | 27.5 | 8685 | 10809 | 1606 | 5.857 |
| always_cloud | 20% | 0.0 | 336 | 372 | 0 | 54.707 |
| always_local | 20% | 35.0 | 12417 | 15829 | 3096 | 0.000 |
| local_llm_probe | 20% | 16.2 | 6443 | 8608 | 1009 | 11.534 |
| ml_dual_risk | 20% | 15.1 | 6080 | 8083 | 985 | 11.354 |
| ml_ttft | 20% | 15.3 | 6254 | 8121 | 1021 | 10.970 |
| queue_service_oracle | 20% | 10.9 | 4834 | 6990 | 657 | 16.276 |
| queue_threshold | 20% | 15.8 | 6277 | 8108 | 1039 | 10.760 |
| semantic_embedding | 20% | 15.5 | 6235 | 8400 | 1011 | 11.527 |
| tail_aware | 20% | 15.4 | 6183 | 8274 | 1011 | 11.410 |
| thinking_probe | 20% | 19.7 | 6896 | 9042 | 1011 | 11.709 |
| always_cloud | 30% | 0.0 | 336 | 372 | 0 | 54.707 |
| always_local | 30% | 35.0 | 12417 | 15829 | 3096 | 0.000 |
| local_llm_probe | 30% | 10.5 | 5260 | 6909 | 579 | 17.179 |
| ml_dual_risk | 30% | 10.4 | 5002 | 7129 | 625 | 17.001 |
| ml_ttft | 30% | 10.4 | 5002 | 6870 | 619 | 16.372 |
| queue_service_oracle | 30% | 6.3 | 3514 | 5316 | 368 | 22.487 |
| queue_threshold | 30% | 11.0 | 5187 | 6865 | 632 | 16.059 |
| semantic_embedding | 30% | 10.2 | 5007 | 6757 | 580 | 17.170 |
| tail_aware | 30% | 10.1 | 4984 | 6620 | 576 | 17.011 |
| thinking_probe | 30% | 13.6 | 5595 | 7372 | 562 | 17.454 |
| always_cloud | 40% | 0.0 | 336 | 372 | 0 | 54.707 |
| always_local | 40% | 35.0 | 12417 | 15829 | 3096 | 0.000 |
| local_llm_probe | 40% | 7.2 | 3909 | 5534 | 319 | 23.138 |
| ml_dual_risk | 40% | 7.2 | 3897 | 5549 | 362 | 23.021 |
| ml_ttft | 40% | 7.2 | 3867 | 5580 | 357 | 21.701 |
| queue_service_oracle | 40% | 3.2 | 2734 | 3897 | 190 | 27.539 |
| queue_threshold | 40% | 7.6 | 4213 | 6030 | 379 | 21.693 |
| semantic_embedding | 40% | 6.8 | 3670 | 5344 | 316 | 23.110 |
| tail_aware | 40% | 6.9 | 3658 | 5340 | 327 | 22.870 |
| thinking_probe | 40% | 9.7 | 4386 | 6029 | 322 | 23.350 |
| always_cloud | 50% | 0.0 | 336 | 372 | 0 | 54.707 |
| always_local | 50% | 35.0 | 12417 | 15829 | 3096 | 0.000 |
| local_llm_probe | 50% | 4.5 | 2984 | 4471 | 172 | 28.717 |
| ml_dual_risk | 50% | 4.5 | 3030 | 4624 | 195 | 28.871 |
| ml_ttft | 50% | 4.5 | 3089 | 4756 | 194 | 27.053 |
| queue_service_oracle | 50% | 1.4 | 1904 | 2698 | 81 | 32.501 |
| queue_threshold | 50% | 5.1 | 3550 | 5227 | 232 | 27.101 |
| semantic_embedding | 50% | 4.1 | 2693 | 4190 | 166 | 28.742 |
| tail_aware | 50% | 4.3 | 2718 | 4307 | 175 | 28.547 |
| thinking_probe | 50% | 6.8 | 3436 | 4939 | 171 | 29.163 |
