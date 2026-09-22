# Governed LLM routing: local-first serving with cloud overflow

Companion repository for the SOC4AI 2026 paper *Promote, Don't Presume:
Preregistered Live Testing of Governed LLM Routing Policies* (Pawar and
Ibrahim, University of Birmingham Dubai). Every live run log, frozen
artifact, agreement, and preregistered protocol behind the paper is in this
repository. The paper keeps to the high-level
story. This README holds the detail the paper leaves out: every constant, every
number, where each one comes from, and how to reproduce it.

## 1. The problem in one paragraph

A local GPU serves LLM requests almost for free but runs only two at a time.
Under bursty traffic requests queue, and time to first token (TTFT) breaks its
SLO. A paid cloud API can take the overflow, but every offloaded request costs
money. The routing policy decides, per request, whether to keep it local or pay
to offload it. This repository treats that policy as a governed service
artifact: sealed at runtime, bound to a machine-readable agreement, protected by
a guard chain, and replaced only when a candidate clears five promotion gates
fixed in advance.

## Names in the code versus the paper

The code keeps its internal names so that every log and hash stays valid. The
paper uses plain names:

| Paper | Code, files, and logs |
|---|---|
| incumbent | `v2`, `guarded_governor`, `governor_v2.json`, `agreement.yaml` |
| candidate (promoted) | `v4`, `v4_governor`, `governor_v4.json`, `agreement_v4.yaml` |
| preliminary variant | `v3`, `agreement_v3.yaml` |
| ungoverned router | `tail_aware` with no breaker and no quota |
| telemetry heuristic | `tail_aware` (TailAwareHeuristic) |
| session 1: nominal / faults / drift / exploratory | blocks `A` / `B` / `C` / `X` |
| session 2: nominal / faults / drift | blocks `V4N` / `V4F` / `V4D` |
| session 2 extensions of session 1 | blocks `B_ext` / `C_ext` |
| replay (offline screening) | shadow corpus, `shadow_*_bursty_n300_s*.jsonl` |

## 2. The agreement

File: `governance/agreement_v4.yaml` (the one the promoted policy runs under).
Earlier versions: `agreement.yaml` (incumbent), `agreement_v3.yaml`
(preliminary variant).

| Term | Value | Why this value |
|---|---|---|
| SLO (per request) | TTFT <= 3839.5 ms | 65th percentile of all-local TTFT in a pilot run, fixed before any policy was compared. An all-local service breaks it on about 35% of requests, so routing has real work to do. |
| SLO guarantee (per run) | at most 10% of requests violate the SLO | A request that exceeds the SLO or fails counts as a violation. |
| Cloud-call quota | at most 30% of a run's requests (90 of 300) | The smallest quota at which telemetry routers reach the 10% target offline (Section 6.1). |
| Quota counting | `counts: billable` | Breaker probes are exempt and failed cloud calls are refunded (recovery-aware accounting). The incumbent's agreement used `all_dispatches`. |
| Study spend cap | USD 20, kill switch | A safety cap for the whole live study, not a service term. Ledger: `soc4ai_out/spend_ledger.json`. Actual spend: USD 2.40. |
| Artifact | SHA-256 `d3b48bfa...dcfb` | The only policy allowed to run. Every run manifest records the hash; the auditor checks it. |
| Breaker | no cloud dispatch while open; opens on 3 consecutive failures or cloud latency > 2x calibrated P95; closes after 3 clean probes | |
| Promotion bar | the five gates in Section 4 | |

Cost is reported as cloud USD per SLO-compliant request, so a policy cannot
look cheap by failing requests. The paper quotes it per 10,000 compliant
requests.

The auditor (`soc4ai/audit.py`) replays each run's JSONL event log against the
agreement and reports every term as met or violated. Because the quota is a
versioned term, a renegotiated agreement can re-judge old logs without a code
change (`analysis/paper/renegotiate_agreement.py`, Section 6.5).

## 3. Runtime

- Local tier: `llama3.1:8b` via Ollama on an RTX 5060 Laptop GPU (8 GB),
  `num_ctx` 2048, keep-alive pinned, concurrency 2 so bursts genuinely queue.
- Cloud tier: Gemini 2.5 Flash on Vertex AI, streaming, thinking disabled. A
  30-call calibration measured median cloud TTFT 577 ms and P95 3,070 ms.
- Feature builder: one builder produces 34 pre-decision features for logging,
  training, and online scoring alike. A test asserts that no post-decision field
  leaks into them.
- Traffic regime: low, medium, high, or burst load, recognised from queue and
  arrival features. The artifact maps each regime to an expert (scoring rule)
  and threshold.

### Guard chain (fixed order, before the expert is consulted)

1. Both tiers degraded.
2. Cloud circuit breaker open.
3. Local hard-breach override: queue past a hard cap, or rolling P95 TTFT above
   the SLO.
4. Telemetry quality check.
5. Regime selection, then the expert scores the request.
6. The expert's proposal passes the quota ledger and the breaker.

A full governed decision takes a median 4.0 us (2.2 us without the guard
chain), so governance costs under 2 us per decision.

## 4. Promotion gates

Fixed in `docs/architecture.md` before any live comparison (last changed
4 Sep 2026; the first live session ran 15 Sep 2026).

| Gate | Criterion | Why | Promoted candidate |
|---|---|---|---|
| G1 Seed support | at least as good on SLO violation in >= 4 of 5 seeds | a gain must hold across traces | 5 of 5 |
| G2 SLO bound | upper 95% CI bound of violation difference <= +1.0 pt | statistical evidence the main term is not worse | -6.0 pt, CI [-14.87, -0.73] |
| G3 Tail gain | P95 TTFT ratio <= 0.95, CI upper < 1.00 | SLOs break in the tail | 0.613, CI [0.471, 0.958] |
| G4 No regression | no seed x workload cell > 5 pt worse | averages hide bad cells | worst seed diff 0.0 pt |
| G5 Decision cap | <= 5 ms added routing time per request | slow signals eat the latency they save | +0.0056 ms |

Per-seed violation differences (candidate minus incumbent, pt): seed 257 -2.3,
263 -3.7, 269 -23.3, 271 0.0, 277 -0.7.

The CI is a seed-level bootstrap. An exact sign-flip permutation test on the
per-seed differences gives p = 0.125 (two-sided). It is reported but not used:
with five seeds the smallest attainable two-sided p is 0.0625. Note that
`analysis/confirmatory_statistics.py` implements a stricter 3 pt / 1.20 rule
for an 8-seed simulated study that was cancelled (Section 7); the live gates
above are the ones used.

## 5. Experiments

### 5.1 Offline screening (simulation)

- Corpus: 4,500 bursty requests = 3 workload mixes (synthetic, real_sample,
  reasoning_coding) x 5 seeds (7, 13, 23, 42, 101) x 300 requests, near GPU
  saturation.
- Shadow mode: every request is served locally and its true TTFT logged; each
  policy's decision is recorded but not executed.
- Why shadow mode: a routed log offloads exactly the requests most likely to
  breach, so it shows a local breach rate of 9.3% where the true rate is 35.0%,
  a 25.7-point selective-labels bias.
- Nine policies, each charged its own decision time: always-local,
  always-cloud, queue threshold, tuned telemetry heuristic (tail-aware), ML TTFT
  classifier, ML dual-risk classifier (breach ROC-AUC 0.958), semantic proxy,
  local-model probe, thinking probe.
- Cloud side: an analytic model `rtt + base_ttft*R + per_token*R*est_tokens`
  with log-normal jitter. Its validity check failed (below), so screening only
  ranks policies. It never supplies a performance claim.

### 5.2 Live experiments

Every run replays a generated trace against the real GPU and real Gemini and
logs every routing decision, TTFT, prefill and decode timing, streaming chunk
times, 1 Hz GPU power and temperature, breaker transitions, and cloud charges.

Conditions:

- Nominal: bursty load, healthy tiers, n = 300.
- Faults: cloud outage on requests 96-155, cloud latency spike of 2.5x the
  calibrated P95 on requests 192-251, n = 300.
- Drift: five-phase trace, final phase slows the cloud by the same delay,
  n = 200.

Run inventory (74 audited runs, 71 preregistered):

| Session | Condition | Seeds | Policies | Runs | Role |
|---|---|---|---|---:|---|
| 1 | nominal | 8, fixed before the study | incumbent, ungoverned | 16 | confirmatory |
| 1 | faults | 3 | incumbent, ungoverned | 6 | preregistered |
| 1 | drift | 3 | incumbent, ungoverned | 6 | preregistered |
| 1 | nominal | 3 of the 8 (211/223/227) | preliminary variant | 3 | exploratory |
| 2 | nominal | 5 fresh (257/263/269/271/277) | candidate, incumbent, ungoverned | 15 | confirmatory (gates) |
| 2 | faults | same 5 | all three | 15 | preregistered |
| 2 | drift | 5 | candidate | 5 | preregistered |
| 2 | faults, drift | 2 more each | incumbent, ungoverned | 8 | extends session 1 |

### 5.3 Preregistration and outcome blinding

- Each session's protocol (seeds, mixes, policy order, fault windows, analysis
  code, decision rule) was frozen and hashed before its first run:
  `soc4ai_out/protocol.json` (sha256 `647b314d...`), `protocol_x.json`
  (`99019c5e...`, exploratory), `protocol_v4.json` (`b61ca6c2e3e288ff...`).
- The progress log recorded only run completion, wall time, and spend.
- The analysis code refuses to read a block until every run in it has a
  `.done` marker.
- Dry runs used separate throwaway seeds and are excluded from all results.
- The candidate's artifact and agreement were hashed before it ran on seeds
  that had never generated a trace.
- Session 2 had a preregistered truncation rule: the nominal block must
  complete; later blocks could be cut for time only. None were cut.

## 6. Results in full

### 6.1 Offline screening at equal cloud budgets

Violation rate at each cloud-call quota (`analysis/paper/frontier_eval.json`, 500 bootstrap
resamples):

| Router | 10% | 20% | 25% | 30% | 40% | 50% | 70% | Min. quota for 10% target | Cost share of always-cloud at that quota |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Telemetry heuristic | 25.0 | 15.1 | 10.6 | 6.5 | 1.9 | 0.7 | 0.0 | 30% | 0.37 |
| Trained dual-risk classifier | 25.0 | 15.1 | 10.5 | 6.9 | 2.7 | 0.6 | 0.0 | 30% | 0.38 |
| Queue threshold | 25.0 | 15.4 | 10.9 | 7.9 | 2.9 | 2.7 | 1.2 | 30% | 0.34 |
| Prompt-only static | 28.8 | 25.4 | 22.1 | 19.3 | 15.8 | 13.2 | 8.8 | 70% | 0.87 |

The prompt-only router scores `z(max_output_tokens) + 0.5 z(prompt_chars) +
is_code` with no telemetry and no labels. It stands in for the signal class used
by RouteLLM and CARROT, which need per-request quality labels this setting does
not have. It is not a reproduction of those systems.

Heuristic minus dual-risk at 30%: -0.38 pt, run-level block-bootstrap 95% CI
[-1.09, +0.31] (a tie). At 40% the heuristic is significantly better: -0.73 pt,
CI [-1.33, -0.18]. Every single-policy candidate fails at least two gates; a
leave-one-seed-out regime-conditioned expert map passes 4 of 5 (fails only G3,
ratio 1.000). The heuristic holds a 2% violation rate at 47% of always-cloud
spend against 67% for the queue threshold.

Static comparison at 30% (cost relative to the heuristic):

| Policy | SLO violation | P99 TTFT | Cost / SLO-ok request (rel.) |
|---|---:|---:|---:|
| always local | 35.0% | 29741 ms | 0.00 |
| always cloud | 0.0% | 378 ms | 2.53 |
| queue threshold | 7.4% | 6659 ms | 0.94 |
| tuned heuristic | 6.5% | 5812 ms | 1.00 |
| ML TTFT | 6.6% | 5812 ms | 1.00 |
| ML dual-risk | 6.9% | 6182 ms | 1.04 |
| semantic proxy | 6.7% | 5999 ms | 1.02 |
| local model probe | 8.2% | 6328 ms | 1.05 |
| thinking probe | 11.2% | 6868 ms | 1.10 |

Probe interference (paired live A/B, 3 seeds, 18 request pairs): a probe
sharing the serving GPU added 349 ms mean TTFT, 453 ms P95 TTFT, and 386 ms
queue wait to the requests it was meant to protect
(`analysis/probe_interference_summary.md`).

### 6.2 Session 1: the incumbent is rejected

- Nominal, 8 seeds: incumbent 18.5% vs ungoverned 4.7% SLO violation,
  difference +13.8 pt, 95% CI [+4.75, +25.58]. P95 13.1 s vs 3.1 s. The
  incumbent was worse in 7 of 8 seeds; excluding reasoning-coding seeds it was
  9.8% vs 4.3%.
- Audit: incumbent 0/8 quota breaches (90/90 calls), artifact hash and breaker
  rule honoured in every run. Ungoverned 8/8 quota breaches, ~270 calls, ~90%
  cloud, ~3x spend.
- Diagnosis: in every incumbent run, 86 to 89 of the 90 cloud calls were
  safety overrides. The rolling-P95 override stayed true while requests were
  offloaded (offloaded requests never refresh the local tail), and the quota ran
  out between requests 96 and 144.
- Telemetry: local breaches come from queueing (73% of breached TTFT; prefill
  median 158 ms). The GPU reached 76 C and throttled in 31% of samples. Live
  Gemini over 2,860 calls: p50 665 ms, p95 2.2 s, p99 10.2 s; 2.9% of cloud
  calls missed the SLO on their own.
- Faults (3 seeds): incumbent 24.0% vs ungoverned 41.6%; failed requests 23 vs
  164; outage detected in 1.6-3.9 s (3/3 runs).
- Drift (3 seeds): cloud-degradation phase incumbent 7.5% vs ungoverned 100%.
  Part of this protection came from quota exhaustion rather than breaker
  detection, which is why drift was re-tested in session 2.
- Preliminary variant (fresh-evidence override + paced release only), 3
  exploratory runs: 14.6% vs incumbent 23.8% vs ungoverned 3.1%; cloud share
  17% vs 30% vs 88%. It beat the incumbent on all 3 seeds.

### 6.3 The three mechanisms of the candidate

1. Fresh-evidence override: the rolling-P95 hard override applies only while
   the local tier is busy.
2. Paced release: after t of n arrivals, at most
   `min(0.3n, floor(0.3t) + ceil(0.05n))` cloud calls may have been made
   (15-call burst allowance at n = 300). Code: `soc4ai/paced.py`.
3. Recovery-aware accounting: breaker probes exempt from the quota, failed
   charged calls refunded. Code: `soc4ai/v4.py`. This reverses an earlier
   logged design decision (D05) that probes should count.

### 6.4 Session 2: the candidate is promoted

Nominal, five fresh seeds (1,500 requests per policy):

| Policy | Viol. % | P95 (s) | Cloud % | USD (5 runs) | USD per 10k compliant | Quota met | SLO met |
|---|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 6.5 | 4.2 | 12.7 | 0.058 | 0.41 | 5/5 | 5/5 |
| Incumbent | 12.5 | 6.9 | 30.0 | 0.120 | 0.91 | 5/5 | 4/5 |
| Ungoverned | 3.7 | 3.0 | 82.2 | 0.330 | 2.29 | 0/5 | 5/5 |

- Ungoverned minus candidate: -2.8 pt violation, CI [1.8, 3.8] in the
  ungoverned router's favour, at 199 to 291 cloud calls per run against a limit
  of 90 and 5.6x the cost per compliant request.
- Cumulative cloud calls (Fig. 3b): incumbent hits 89 calls around request 185;
  candidate ends near 38; ungoverned near 247.
- GPU energy: 0.041 Wh per compliant request for the candidate against 0.020
  for the ungoverned router, the price of serving locally.

Faults, five fresh seeds:

| Policy | Viol. % | P95 (s) | Cloud % | USD | USD per 10k compliant | Failed requests | Quota met | SLO met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 12.2 | 5.9 | 8.6 | 0.031 | 0.24 | 32 | 5/5 | 1/5 |
| Incumbent | 17.3 | 8.1 | 29.5 | 0.108 | 0.87 | 39 | 5/5 | 0/5 |
| Ungoverned | 42.6 | 9.5 | 82.2 | 0.255 | 2.96 | 301 | 0/5 | 0/5 |

- Candidate minus incumbent: -5.13 pt, CI [-8.13, -2.80]; P95 ratio 0.725,
  CI [0.594, 0.912].
- 5 to 9 breaker probes per run were exempt from the quota; no request was
  denied for budget.
- Peak 24-request rolling violation: candidate 25.8%, incumbent 35.8%,
  ungoverned 100%.
- Incumbent vs ungoverned under faults over 5 seeds (session 1 plus the
  extension): 21.1% [16.1, 29.5] vs 42.3% [39.8, 44.1]; failed 7.6 vs 57.6 per
  run; incumbent better on every seed.

Drift: degraded phase ungoverned 100%, incumbent 13%, candidate 15% (CI [7, 21],
sign-flip p = 1.0). The candidate had no budget denial in any of its 5 drift
runs, so the protection comes from routing and the breaker, not an exhausted
quota.

### 6.5 Renegotiated agreements

`analysis/paper/renegotiate_agreement.py` re-audits every completed log under quotas of
20/30/40/50% and both counting rules (`renegotiation.json`):

- At a tightened 20% quota the candidate complies in 13 of 15 runs (4/5
  nominal, 5/5 faults, 4/5 drift); the incumbent complies in 0/5 and 0/8.
- The ungoverned router complies in only 2 of its 28 runs even at a 50% quota.

## 7. Simulator validity checks (why simulation only ranks)

Pass marks were fixed before the checks ran.

| Check | Question | Result | Pass mark | Verdict |
|---|---|---|---|---|
| Stage A | Is the queue-replay event arithmetic correct? | queue-wait MAE 5.63 ms | build check | pass |
| Stage B | Do stored local service times survive a change in contention? | P90 per-request error 41.5%, Kendall tau 0.03, n = 140 | 25%, 0.95 | fail |
| Cloud proxy | Does the analytic cloud model match live Gemini? | P50/P95 TTFT error 51.8%/94.0%, 0% prediction-interval coverage, 60 calls | 20%/30% | fail |

Because Stage B and the cloud proxy failed, a planned 3,200-request simulated
confirmatory study was cancelled (decision D15) and every performance claim
moved to live runs.

## 8. Layout

- `apex/`: gateway package. `core.py` (config, gateway state with rolling
  P50/P95/P99, telemetry, the 34-feature builder, labels), `policies.py` (nine
  policies behind one `score(features)` method plus the guarded governor),
  `providers.py` (Ollama provider, cloud model, calibration, VRAM guard),
  `gateway.py`, `events.py` (JSONL log, Parquet, manifests), `workload.py`
  (seeded traces), `runner.py`.
- `soc4ai/`: live study. `study.py` (session 1 driver), `v4study.py` (session 2
  driver), `v4.py` (candidate governor), `paced.py` (paced release and the
  fresh-evidence override), `audit.py` (agreement auditor), `ledger.py` (spend
  kill switch), `creds.py` (credential guard), `telemetry.py` (1 Hz GPU
  sampling), `providers.py` (Vertex AI streaming, fault injection),
  `analyze_live.py`, `exploratory.py`, `paper_numbers.py`.
- `governance/`: the three agreement files.
- `ml/`: risk-model training and frozen artifacts (`governor_v1.json`,
  `governor_v2.json` threshold 4.9865, `governor_v4.json` sha256 `d3b48bfa...`).
- `replay/`: queue-replay simulator, paired audits, probe-interference audit.
- `analysis/`: evaluation scripts, validity checks, result tables, figures.
  `analysis/paper/` holds the scripts behind the paper: `frontier_eval.py`
  (Table 2, Fig. 3a), `renegotiate_agreement.py` (20% quota re-audit),
  `block_boot.py` (heuristic vs classifier CI), `make_figures.py` (Fig. 3),
  `pipeline.tex` (Fig. 1), and their JSON outputs.
- `data/`: frozen manifests, the 4,500-request replay corpus, validity runs.
- `soc4ai_out/`: protocols and their hashes, spend ledger, calibration, every
  live run log (`runs/<block>/*.jsonl`), and the analysis JSON
  (`analysis/live_V4.json` and others).
- `docs/`: scope, architecture (the five gates), results, decision log.
- `deliverables/`: the simulator validity reports.
- `tests/`: 43 tests. Run `python -m pytest -q` from the repository root.

## 9. Reproducing

Tested on Python 3.13 on Windows 11. Install the dependencies first:

```
python -m pip install -r requirements.txt
python -m pytest -q
```

### 9.1 Reproduce the paper's numbers (no GPU, no cloud key, a few minutes)

Every live run log ships in `soc4ai_out/runs/`, so all paper numbers can be
recomputed on a laptop CPU:

```
python -m soc4ai.v4study analyze               # session 2: Table 3, gates (Table 1)
python soc4ai/analyze_live.py --exp A          # session 1 nominal (also B, C)
python analysis/paper/frontier_eval.py         # replay: Table 2, Fig. 3a
python analysis/paper/renegotiate_agreement.py # re-audit at 20-50% quotas
python analysis/paper/make_figures.py          # Fig. 3 -> analysis/paper/figures/
```

`frontier_eval.json` and `renegotiation.json` are regenerated byte-for-byte
from the shipped data.

### 9.2 Run the gateway locally (GPU, no cloud key)

Needs Ollama running with `llama3.1:8b` pulled (the default cloud-model
calibration path also uses `gemma4:e2b`).

```
python tools/preflight.py        # checks setup, prints READY or what is broken
python tools/demo_governor.py    # walks every guard, no GPU or network
python run_all.py --quick        # 3-4 minute end-to-end demo
python analysis/evaluate.py      # replay comparison of the nine policies
```

### 9.3 Re-run the live study (GPU and a Google Cloud project)

Set these environment variables; never commit a key (the `.gitignore` and
`tests/test_secret_hygiene.py` guard against it):

| Variable | Meaning |
|---|---|
| `SOC4AI_GCP_KEY` | path to a service-account JSON key with Vertex AI access (required) |
| `SOC4AI_GCP_PROJECT` | expected project ID (optional; defaults to the key's `project_id`) |
| `SOC4AI_GCP_LOCATION` | Vertex AI region (optional; default `us-central1`) |
| `SOC4AI_KEY_SHA256_PREFIX` | optional pin: first 16 hex chars of the key file's SHA-256 |

PowerShell: `$env:SOC4AI_GCP_KEY = "C:\path\to\key.json"`.
Bash: `export SOC4AI_GCP_KEY=/path/to/key.json`.

```
python -m soc4ai.study calibrate    # stages: recalibrate, calibrate, dryrun, freeze, batch
python -m soc4ai.study dryrun
python -m soc4ai.study freeze       # writes protocol.json + protocol.sha256
python -m soc4ai.study batch
python -m soc4ai.v4study artifact   # stages: artifact, freeze, dryrun, batch, analyze
python -m soc4ai.v4study freeze
python -m soc4ai.v4study batch
python -m soc4ai.v4study analyze
```

`freeze` refuses to run on a dirty git tree, so the protocol records exactly
which code produced the results; commit first. The spend ledger stops a batch
at USD 20. The full live study cost USD 2.40. A new run writes new protocol
files, so move the shipped `soc4ai_out/` aside first if you want to keep it.

The shipped `soc4ai_out/protocol.json` records the GCP project used for the
original runs; it is left unedited because its SHA-256 is the preregistration
record.

## 10. Scope

One 8 GB laptop GPU that thermally throttles, one Gemini region, generated
workloads from seeded mixes rather than production traces, five seeds per
policy in session 2. Answer quality is out of scope: local-first overflow
assumes the local model meets the task's acceptance bar. Prompt text in the
synthetic traces was produced with LLM assistance; every measurement is the
real model's response.

## Citation

```
@inproceedings{pawar2026promote,
  title     = {Promote, Don't Presume: Preregistered Live Testing of Governed LLM Routing Policies},
  author    = {Pawar, Om Rajesh and Ibrahim, Ahmad},
  booktitle = {Service-Oriented Computing -- ICSOC 2026 Workshops (SOC4AI)},
  year      = {2026},
  note      = {Under review}
}
```
