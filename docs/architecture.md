# Architecture

One gateway process decides per request: local GPU or cloud overflow. The
design goal is to hold a latency SLO inside a cloud budget, with every routing
decision logged, sealed, and auditable against a machine-readable agreement.

## Request path

```
arrival -> telemetry sample -> feature snapshot (pre-decision only)
       -> policy.score() -> guard layer -> local (FIFO queue) or cloud
       -> measure timings -> event log (JSONL -> Parquet + run manifest)
```

Everything left of the decision boundary is available before the request runs;
everything right of it is outcome.

Code map:

- apex/core.py: config models, gateway state (rolling P50/P95/P99 windows),
  telemetry sampling, the 34-feature builder, the label functions, and the
  frozen thresholds
- apex/providers.py: live Ollama local provider, the cloud model, the cloud
  calibration routine, the VRAM residency guard
- apex/policies.py: all nine routing policies plus the guard layer
- apex/events.py: JSONL event log, Parquet flattener, run manifests
- apex/gateway.py: the request orchestrator, one class
- apex/workload.py: seeded trace generator for three workload mixes with
  bursty arrivals (steady and saturation modes exist but are unused here)
- apex/runner.py: replays a trace under one policy, writes run artifacts

## The two tiers

Local is a real model on the real GPU: `llama3.1:8b`, two requests at a time,
FIFO. Its queueing is genuine, so under bursts requests actually wait and the
tail degrades. This contention is the object of study.

Cloud is deliberately not a second live model. That was tried first, and the
co-located model's latency inflated roughly eight times whenever the local
model was under load, which is GPU contention rather than cloud behaviour, and
it corrupted the cloud-derived features every machine-learning policy read.
Instead the cloud tier is replayed from a fixed cost-and-latency model:

```
TTFT_cloud = rtt + base_ttft * R + per_token * R * est_tokens,   times log-normal jitter
```

`rtt` is the network round trip, `base_ttft` the fixed per-call latency before
the first token, `per_token` the decode time per token, and `est_tokens` the
request's predicted output length. The jitter term is multiplicative
log-normal, which reproduces the run-to-run spread seen in the measurements
better than additive Gaussian or exponential noise. `R` is a speed-ratio
multiplier held at 1 for every headline number and swept to 0.5 and 2 only as
a sensitivity check.

The constants are fitted from real Gemini 2.5 Flash calls (Vertex AI)
measured once in isolation. Because the repository cannot ship GCP
credentials, the default calibration path re-fits the same constants from a
local `gemma4:e2b` run, which is what the checked-in run manifests record
under `calibration.calibration_model`. The model is never tuned against the
run it is used in.

## Features and leakage

One function, `build_features` in core.py, produces the features for logging,
training, and online inference alike, so there is no train/serve skew. The
rule: nothing known only after the decision goes in. Rolling quantiles only
include requests that completed before now. A test asserts that no
post-decision field ever appears among the features.

34 features in 6 groups: workload shape (9), gateway state (6), arrival
dynamics (5), rolling local history (6), hardware telemetry (6), and cloud
health (2). Workload and arrival features describe the request and the
arrival process, so they transfer to other hardware. The absolute VRAM figures
and absolute rolling-latency percentiles are specific to this GPU and Ollama
setup and would need recalibration elsewhere. Where the tuned heuristic uses a
configuration-specific group, it uses the ratio to the SLA rather than the raw
value, so the scoring function does not silently assume this hardware.

## Policies

Every policy implements one method, `score(features) -> float`, so the
evaluator can sweep an offload threshold and compare all policies at matched
budgets (10-50%) rather than at whatever operating point each prefers.

Baselines: `always_local`, `always_cloud`, `queue_threshold`, and the tuned
heuristic, called tail-aware here because it watches the slow-request tail.
The tail-aware rule is a weighted sum of seven normalised pressure signals
(queue depth, active local jobs, rolling P95 TTFT against the SLA, rolling P95
service time against the occupancy target, burst ratio, VRAM pressure, and
request work), with the weights tuned by random search on the same data and
objective the machine-learning model sees. It carries one hard override: if
the rolling P95 TTFT already exceeds the SLA, it routes to cloud regardless of
score. It is a tuned scoring function compared head-to-head against a
calibrated classifier under a matched-budget test, not a chain of if-else
branches.

Machine-learning tiers: TTFT-only and dual-risk calibrated classifiers, built
with scikit-learn's `CalibratedClassifierCV`, best Brier score selected.
Signal tiers T2 (semantic embedding), T3 (local model probe), and T4 (thinking
probe) each carry an explicit estimate of their decision and probe cost
(`decision_ms`, `probe_ms`, `probe_tokens`, `router_cost_usd`), which the
evaluator adds back into their measured TTFT before scoring rather than
crediting them with a zero-cost score. T3's real cost was also measured live
in the probe-interference audit, which is where the +349 ms mean TTFT penalty
comes from.

## The governed slot: the deliverable

The deliverable is not any one scoring function but the slot they plug into: a
governed, swappable policy slot that decides which policy is active for a
given traffic condition, and only lets that choice change through an
acceptance rule fixed in advance.

Around that slot sits a deterministic guard chain that enforces safety and
ordering before any policy is consulted. In order, `GuardedGovernor.decide()`
checks: both tiers degraded at once (local unavailable *and* cloud breaker
open), cloud breaker availability, a local hard-breach override, the telemetry
tier (full, tail, minimal, or unusable), and the condition lookup against the
frozen file, before handing the routing choice to the active policy's
`score()`. The order is load-bearing: an earlier version checked breaker
availability before the both-degraded case and gave the wrong fallback under
one failure combination, so the sequence is now pinned by a dedicated test
rather than left as an informal convention.

The cloud circuit breaker opens on three consecutive failures or when cloud
latency exceeds twice its measured baseline, waits a 30-second cooldown, then
admits one probe at a time; three clean probes close it and any failure
re-opens it. Each probe is charged to the cloud budget like any other call.
The breaker needs one reference latency for a whole episode; that value is
derived from the run's own cloud calibration and written into the run
manifest, so the threshold a run was judged against stays auditable rather
than living as a constant in the code.

The total-cloud budget ledger completes the runtime. Every discretionary
route, safety override, and probe draws on the same budget, so overrides and
probes are never free.

The governor is built by one factory, `make_guarded_governor`, which the
runner, the demo tool, and the tests all go through, so the thing that ships
is the thing that is tested. It is deliberately not registered in
`all_policies()`, since it takes the expert pool as a constructor argument and
would recurse. Run it with `--policy guarded_governor`, or walk every guard
without a GPU using `tools/demo_governor.py`.

## The frozen artifact and the five-gate bar

The policy artifact (`ml/artifacts/governor_v4.json` for the promoted policy;
`governor_v1.json` to `governor_v3.json` are its predecessors) maps each of four
traffic conditions (low, medium, high, burst) to an expert policy and a
threshold, and is not updated during a run. It stores a SHA-256 hash of itself and of the calibration values
it was built from, so a run's manifest can be checked against the exact file
it was judged under.

A candidate replaces the incumbent in a condition only by clearing all five
gates below, in order. The gates and their thresholds were fixed in
`config.yaml` before the comparison was run, so a result favouring either side
could not be read as chosen after the fact.

1. **Seed support.** The candidate must be at least as good as the incumbent
   on the primary metric in at least 4 of the 5 evaluation seeds. This blocks
   a policy that wins only on one lucky seed.
2. **SLA CI bound.** The upper end of the bootstrap 95% CI on the
   candidate-minus-incumbent SLA-violation difference must be at most +1.0
   percentage point. A candidate whose plausible worst case is a meaningful
   regression does not pass.
3. **P95 ratio.** The ratio of candidate to incumbent P95 TTFT must be at most
   0.95, with the interval's upper bound below 1.00. The candidate has to
   actually improve the tail, not just the median.
4. **No catastrophic regression.** No single seed x mix cell may be more than
   5 percentage points worse than the incumbent, even if the pooled average
   improves. This protects the hard regimes.
5. **Decision-time cap.** The candidate must add no more than 5 ms of routing
   time per request over the incumbent, so a policy cannot buy accuracy with
   latency the user pays for.

Mean queue wait and cost cannot rescue a candidate that fails a gate. The
bar did its job in both directions: the replay candidates (trained classifiers,
probe signals) did not beat the tuned tail-aware rule, so it kept the slot; the
live candidate `governor_v4.json`, which adds paced release, a fresh-evidence
override, and recovery-aware accounting, cleared all five gates on fresh seeds
and is the promoted policy. Every state is recorded, versioned, and checkable
from any run's manifest.

The online runtime is read-only: it loads the frozen artifact and applies it,
and nothing in a live run can change which policy holds a condition. All
adaptivity lives in a separate offline promotion cycle that collects a fresh
all-local corpus, trains candidates, scores each against the incumbent at
matched budgets, and runs the five gates. A new artifact, with a new hash, is
written only when a candidate clears all five.

## The all-local test set

Routed logs only show outcomes for the route taken. At 30% offload the local
outcome is missing for exactly the requests that got offloaded, so naive
kept-local statistics looked far better than reality (9.3% against the true
35.0% breach rate for the same requests). The fix: every run forces the
request local and logs the full outcome, features, and label, while the
routing decision each policy would have made is still computed and stored,
just not carried out. That gives an unbiased training set and a per-request
local result.

This is the missing-outcome (selective-label) correction applied to a routed
inference system, and quantifying it turned out to matter as much as the
router comparison itself.

## Replay and live evidence

Replay ranks candidates and sets the quota; live runs decide promotion. Every
performance number behind the promoted policy is measured against the real
Gemini 2.5 Flash API on the real GPU (see the repository README, Sections 9
and 11).

## Reproducibility

Every run writes a manifest with the full frozen config, seed, calibration
values, artifact hash, and wall clock. Traces are seeded. Measured GPU runs go
through the VRAM residency guard, which fails closed rather than continuing
with a model silently split into system RAM, and `run_all.py` waits for the
GPU to cool below 58C between measured runs, because the laptop GPU
thermal-throttles under sustained back-to-back measurement and that quietly
skews timing comparisons.
