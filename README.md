# Governed LLM Routing Policy

Code, data, frozen artifacts, agreements, preregistered protocols, and every
live run log behind the SOC4AI 2026 paper:

> Om Rajesh Pawar and Ahmad Ibrahim. *Promote, Don't Presume: Preregistered
> Live Testing of Governed LLM Routing Policies.* Accepted at SOC4AI 2026
> (ICSOC 2026 Workshops, Springer LNCS). School of Computer Science,
> University of Birmingham Dubai, UAE.
> PDF: [`paper/Promote_Dont_Presume_SOC4AI_2026.pdf`](paper/Promote_Dont_Presume_SOC4AI_2026.pdf)

The paper keeps to the high-level story. This README holds everything the paper
leaves out: every constant, every number, which file it comes from, and the
command that regenerates it. All paper numbers can be recomputed on a laptop
CPU in a few minutes, with no GPU and no cloud key (Section 10.1).

---

## Contents

1. [What this repository shows](#1-what-this-repository-shows)
2. [Quick start](#2-quick-start)
   - [Using the governed router](#using-the-governed-router)
3. [Paper to evidence map](#3-paper-to-evidence-map)
4. [Names in the code versus the paper](#4-names-in-the-code-versus-the-paper)
5. [The agreement](#5-the-agreement)
6. [Runtime: artifact, guard chain, breaker](#6-runtime-artifact-guard-chain-breaker)
7. [The routing policies](#7-the-routing-policies)
8. [Promotion gates](#8-promotion-gates)
9. [Experiments](#9-experiments)
10. [Reproducing](#10-reproducing)
11. [Results in full](#11-results-in-full)
12. [Repository layout](#12-repository-layout)
13. [Testbed hardware and software](#13-testbed-hardware-and-software)
14. [Deploying on other hardware or providers](#14-deploying-on-other-hardware-or-providers)
15. [Troubleshooting](#15-troubleshooting)
16. [Citation](#16-citation)

---

## 1. What this repository shows

A local GPU serves LLM requests at near-zero marginal cost but runs only two at
a time. Under bursty traffic requests queue and time to first token (TTFT)
breaks its SLO. A paid cloud API can take the overflow, but every offloaded
request costs money. The **routing policy** decides, per request, whether to
keep it local or pay to offload it.

This repository treats that policy as a **governed service artifact**:

- it is **sealed**: a versioned, SHA-256-hashed JSON file the runtime reads and
  never writes;
- it is **bound to a machine-readable agreement** (SLA4OAI-style YAML) that
  fixes an SLO guarantee, a cloud-call quota, and governance terms;
- it sits **behind a guard chain** of hard constraints it cannot override
  (circuit breaker, quota ledger, hard-breach override);
- it is **replaced only when a candidate clears five promotion gates** on
  preregistered, held-out live evidence;
- every run is **audited** against the agreement from its event log.

Headline result (session 2, five fresh seeds, live Gemini 2.5 Flash): under an
agreement of at most 10% SLO violations and at most 30% cloud calls, the
promoted policy met the SLO guarantee and the quota in 5 of 5 nominal runs while
sending 12.7% of requests to the cloud. The ungoverned router met the SLO only
by sending 82.2% of requests to the cloud, breaking the quota in every run, at
5.6 times the cloud cost per SLO-compliant request (2.29 vs 0.41 USD per 10,000
compliant requests).

---

## 2. Quick start

```
git clone https://github.com/OmRajeshPawar/Governed-LLM-Routing-Policy.git
cd Governed-LLM-Routing-Policy
python -m pip install -r requirements.txt
python -m pytest -q                            # 43 tests
python -m soc4ai.v4study analyze               # session 2 numbers (Table 1, Table 3)
python analysis/paper/frontier_eval.py         # replay numbers (Table 2, Fig. 3a)
python analysis/paper/make_figures.py          # Fig. 3 -> analysis/paper/figures/
```

None of these commands needs a GPU, Ollama, or a cloud key.

### Using the governed router

The promoted policy is a component you can load and call.
`examples/route_with_governor.py` verifies the sealed artifact against the
agreement, builds the governor, and routes four requests:

```python
from apex.core import load_config
from apex.policies import GovernorArtifact
from soc4ai.v4 import make_v4_governor

cfg = load_config()
cfg.governor.artifact_path = "ml/artifacts/governor_v4.json"      # the promoted policy
assert GovernorArtifact.sha256(cfg.governor.artifact_path) == agreement_sha256

gov = make_v4_governor(cfg, episode_requests=300,
                       frozen_cloud_p95_ms=3069.6,               # calibrated cloud P95
                       heuristic_weights=tuned_weights)          # ml/artifacts/meta.json
decision = gov.decide(features, gov.artifact.thresholds["tail_aware"])
# ... serve on decision.route ("local" or "cloud"), then report the outcome:
gov.record_outcome(decision, ok=True, ttft_ms=812.0)
```

```
$ python examples/route_with_governor.py
artifact sealed and verified: d3b48bfa0f66dcfb...
idle local tier        -> local  score=  1.60  reason=tail_ok
busy, short queue      -> local  score=  4.21  reason=tail_ok
busy, queue building   -> cloud  score=  5.74  reason=tail_pressure:budget_admitted
burst, deep queue      -> cloud  score= guard  reason=local_hard_breach_cloud_override:budget_admitted
cloud calls charged so far: 2 of 90
```

`features` is the 34-field pre-decision snapshot built by
`apex.core.build_features`. Every decision carries a reason code, so a run log
shows which layer routed each request: the scoring rule (`tail_*`), the guard
chain (`local_hard_breach_*`, `circuit_*`), or the quota ledger
(`budget_admitted`, `budget_denied`). The full gateway, which wires the
governor to Ollama and Vertex AI, is `apex/gateway.py` with the live providers
in `soc4ai/providers.py`.

---

## 3. Paper to evidence map

Every quantitative statement in the paper, where it lives, and how to
regenerate it.

| Paper item | Value(s) | Evidence file | Regenerate with |
|---|---|---|---|
| Abstract, Table 3: candidate nominal | 6.5% viol., P95 4.2 s, 12.7% cloud, 0.41 USD/10k met, quota 5/5, SLO 5/5 | `soc4ai_out/analysis/live_V4.json` -> `V4N.pooled.v4` | `python -m soc4ai.v4study analyze` |
| Table 3: incumbent nominal | 12.5%, 6.9 s, 30.0%, 0.91, 5/5, 4/5 | `live_V4.json` -> `V4N.pooled.v2` | same |
| Table 3: ungoverned nominal | 3.7%, 3.0 s, 82.2%, 2.29, 0/5, 5/5 | `live_V4.json` -> `V4N.pooled.ungoverned` | same |
| 5.6x cost ratio | 2.29 / 0.41 | `V4N.pooled.*.usd_per_slo_met_x1e4` | same |
| Ungoverned 2.8 pt lower violation | CI [1.8, 3.8] | `V4N.v4_vs_ungoverned` | same |
| Table 3: faults rows | 12.2/17.3/42.6% viol.; failed 32/39/301 | `live_V4.json` -> `V4F.pooled` | same |
| Table 1: gate results | G1 5/5; G2 -6.0 pt, CI [-14.87, -0.73]; G3 0.613, CI [0.471, 0.958]; G4 none worse; G5 +0.0056 ms | `live_V4.json` -> `V4N.v4_vs_v2` | same |
| Session 1: incumbent misses SLO | 18.5% vs 4.7%, diff +13.8 pt, CI [+4.75, +25.58] | `soc4ai_out/analysis/live_A.json` -> `A.confirmatory_statistics` | `python soc4ai/analyze_live.py --exp A` |
| Session 1: audit (quota, hash, breaker) | incumbent compliant 8/8, ungoverned breaches quota 8/8 | `soc4ai_out/analysis/audit_A.json` | same |
| Cause 1: quota spent by request 96-144 | 86-89 of 90 calls were overrides | `soc4ai_out/runs/A/*_guarded_governor.jsonl` | `python soc4ai/analyze_live.py --exp A` |
| Fig. 3b: incumbent hits quota ~req 185, candidate ~38 calls | 185 / 38.2 / 246.6 | `analysis/paper/figures/make_figures.json` | `python analysis/paper/make_figures.py` |
| Fig. 3c: ungoverned 100% in fault windows | peak 100% | same | same |
| 20% renegotiated quota | candidate 13 of 15 runs, incumbent 0 | `analysis/paper/renegotiation.json` | `python analysis/paper/renegotiate_agreement.py` |
| Drift: degraded phase | ungoverned 100%, incumbent 13%, candidate 15% | `live_V4.json` -> `V4D`, `C5` | `python -m soc4ai.v4study analyze` |
| Table 2: replay | heuristic 6.5%, classifier 6.9%, queue 7.9%, prompt-only 19.3% at 30%; prompt-only needs 70% | `analysis/paper/frontier_eval.json` | `python analysis/paper/frontier_eval.py` |
| 2.3x spend of prompt-only | 0.87 / 0.37 | `frontier_eval.json` -> `feasible_quota` | same |
| Selective-labels bias | routed log 9.3% vs true 35.0% | `analysis/results/summary_metrics.json` -> `selective_label` | `python analysis/evaluate.py` |
| Classifier choice | logistic regression selected (Brier 0.0657 vs 0.0670), ROC-AUC 0.958 | `ml/artifacts/meta.json` -> `selected`, `metrics` | Section 7.2 |
| Heuristic weights | Section 7.1 | `ml/artifacts/meta.json` -> `heuristic_weights`; `soc4ai_out/protocol_v4.json` -> `heuristic_weights_values` | Section 7.1 |
| Heuristic threshold | 4.9865 | `ml/artifacts/governor_v4.json` -> `thresholds.tail_aware` | Section 7.1 |
| Breaker: 3 failures, 2x P95, 3 probes | | `config.yaml` -> `circuit`; `governance/agreement_v4.yaml` | |
| Arrival process | Poisson, 1.5/s for 6 of every 24 requests, else 0.3/s | `apex/workload.py` -> `_arrival_offsets` | |
| Run counts | 74 audited, 71 preregistered | `soc4ai_out/runs/*/` (74 `.jsonl` + `.done`) | Section 9.3 |
| Fig. 1 | pipeline diagram | `analysis/paper/pipeline.tex` | `pdflatex pipeline.tex` |
| Fig. 2 | agreement excerpt | `governance/agreement_v4.yaml` | |

---

## 4. Names in the code versus the paper

The code keeps its internal names so that every log and hash stays valid. The
paper uses plain names.

| Paper | Code, files, and logs |
|---|---|
| incumbent | `v2`, `guarded_governor`, `governor_v2.json`, `governance/agreement.yaml` |
| candidate (promoted) | `v4`, `v4_governor`, `governor_v4.json`, `governance/agreement_v4.yaml` |
| preliminary variant | `v3`, `paced_governor`, `governor_v3.json`, `governance/agreement_v3.yaml` |
| ungoverned router | `tail_aware` with no breaker and no quota |
| telemetry heuristic | `TailAwareHeuristic` (`tail_aware`) in `apex/policies.py` |
| trained classifier | `ml_dual_risk` / `ml_ttft` (`ml/artifacts/*_model.joblib`) |
| prompt-only router | `prompt_static` in `analysis/paper/frontier_eval.py` |
| session 1: nominal / faults / drift / exploratory | blocks `A` / `B` / `C` / `X` |
| session 2: nominal / faults / drift | blocks `V4N` / `V4F` / `V4D` |
| session 2 extensions of session 1 | blocks `B_ext` / `C_ext` |
| replay (offline screening) | shadow corpus `data/events/shadow_*_bursty_n300_s*.jsonl` |
| SLO violation | `ttft_violation` / `violation` |
| SLO guarantee met | `slo_guarantee_runs` |

---

## 5. The agreement

File: `governance/agreement_v4.yaml` (the agreement the promoted policy runs
under). Earlier versions: `governance/agreement.yaml` (incumbent) and
`governance/agreement_v3.yaml` (preliminary variant). Structure follows SLA4OAI
(Gamez-Diaz, Fernandez, Ruiz-Cortes, ICSOC 2019): context, metrics, plans with
guarantees and quotas, plus a governance block.

| Term | Value | Why this value |
|---|---|---|
| SLO (per request) | TTFT <= 3839.5 ms | 65th percentile of all-local TTFT on the 4,500-request replay corpus (`ml/artifacts/meta.json` -> `frozen_thresholds`). An all-local service breaks it on 35.0% of requests. |
| SLO violation | TTFT above 3839.5 ms, or the request failed | A failed request (provider error or timeout) counts as a violation and is also counted on its own (`failed`). |
| SLO guarantee (per run) | at most 10% of a run's requests violate the SLO | |
| Cloud-call quota | at most 30% of a run's requests (90 of 300) | Smallest quota at which telemetry routers reach the 10% target in replay (Table 2). |
| Quota counting | `counts: billable` | Breaker probes are exempt and failed cloud calls are refunded. The incumbent's agreement counts `all_dispatches`. |
| Study spend cap | USD 20, kill switch | Safety cap for the whole live study. Ledger: `soc4ai_out/spend_ledger.json`. Actual spend: USD 2.40. |
| Artifact | SHA-256 `d3b48bfa0f66dcfb...` | The only policy allowed to run. Every run manifest records the hash; the auditor checks it. |
| Breaker | opens on 3 consecutive failures or cloud latency > 2x frozen P95; cooldown 30 s or 10 arrivals; closes after 3 successful half-open probes; no cloud dispatch while open | |
| Promotion bar | the five gates of Section 8 | |

Cost is reported as cloud USD per SLO-compliant request, so a policy cannot
lower its cost by failing requests. The paper quotes it per 10,000 compliant
requests.

**Auditor.** `soc4ai/audit.py` replays each run's JSONL event log against an
agreement and reports every term as met or violated (artifact hash, cloud-call
quota, breaker rule, spend quota, SLO guarantee). Because the quota is a
versioned term, a renegotiated agreement re-judges old logs without any code
change (`analysis/paper/renegotiate_agreement.py`, Section 11.6).

---

## 6. Runtime: artifact, guard chain, breaker

### 6.1 Tiers

- **Local tier:** `llama3.1:8b` via Ollama, `num_ctx` 2048, keep-alive pinned,
  concurrency 2 (`config.yaml` -> `local_concurrency`), so bursts genuinely
  queue.
- **Cloud tier:** Gemini 2.5 Flash on Vertex AI (`us-central1`), streaming,
  thinking budget 0, temperature 0. A 30-call calibration measured median cloud
  TTFT 577 ms and P95 3,069.6 ms (`soc4ai_out/calibration.json`); the P95 is
  frozen into the protocol and used by the breaker and the fault model.

### 6.2 The policy artifact

`ml/artifacts/governor_v4.json`:

```json
{
  "schema_version": "governor-v1",
  "artifact_id": "apex-v4-paced-fresh-refund-2026-09-16",
  "regime_map": {"low": "tail_aware", "medium": "tail_aware",
                 "high": "tail_aware", "burst": "tail_aware"},
  "thresholds": {"tail_aware": 4.986486535529362, "ml_ttft": 0.5, "ml_dual_risk": 0.5},
  "promotion_status": {"changes": {
     "budget": "paced: released = min(cap, floor(0.30 * arrivals) + ceil(0.05 * episode))",
     "hard_override": "rolling-P95 hard breach applies only when the local tier is busy",
     "probe_accounting": "half-open probes not charged; failed charged calls refunded"}},
  "development_seeds": [7, 13, 23, 42, 101]
}
```

The regime map names the scoring rule used at each load level (low, medium,
high, burst, recognised from queue and arrival features). The runtime loads the
file, checks its hash, and never writes it.

### 6.3 Guard chain (fixed order, before the scoring rule is consulted)

1. Both tiers degraded -> fail safe.
2. Cloud circuit breaker open -> keep local.
3. Local hard breach: queue past the hard cap, or rolling P95 TTFT above the
   SLO (in the candidate, only while the local tier is busy) -> offload.
4. Telemetry quality check.
5. Regime recognition, then the scoring rule proposes a route.
6. Any cloud call must have room in the quota ledger and a closed breaker.

A full governed decision takes a median 4.0 us (2.2 us without the guard
chain). G5 measures the candidate's added decision time at +0.0056 ms.

### 6.4 Circuit breaker (`config.yaml` -> `circuit`)

| Parameter | Value |
|---|---|
| `consecutive_failure_limit` | 3 |
| `failure_window` / `failure_window_limit` | 5 failures in the last 20 calls |
| `latency_multiplier` | 2.0 x frozen cloud P95 |
| `latency_window` / `latency_breach_limit` | 5 of the last 10 calls |
| `latency_success_minimum` | 10 calls before latency tripping applies |
| `cooldown_seconds` / `cooldown_arrivals` | 30 s / 10 arrivals |
| `half_open_successes_to_close` | 3 |

---

## 7. The routing policies

### 7.1 Telemetry heuristic (the scoring rule of every governed policy)

`apex/policies.py` -> `TailAwareHeuristic.score`. A weighted sum of seven
normalised pre-decision features:

```
score = w_queue        * pending_local / queue_threshold          (queue_threshold = 3)
      + w_active       * active_local / local_concurrency         (concurrency = 2)
      + w_ttft_tail    * roll_ttft_p95 / SLO                      (SLO = 3839.5 ms)
      + w_service_tail * roll_service_p95 / occupancy_target      (4610.9 ms)
      + w_burst        * max(0, burst_ratio - 1)
      + w_vram         * vram_used / (vram_used + vram_free)
      + w_work         * (prompt_token_est + max_output_tokens) / 2048
route = cloud if score >= threshold, else local
```

Tuned weights used in every live run (`ml/artifacts/meta.json` ->
`heuristic_weights`, recorded in `soc4ai_out/protocol_v4.json`):

| Weight | Value |
|---|---:|
| `queue` | 1.8199 |
| `active` | 1.2334 |
| `ttft_tail` | 0.0230 |
| `service_tail` | 1.5896 |
| `burst` | 0.1017 |
| `vram` | 0.9930 |
| `work` | 1.8853 |
| threshold (`governor_v2.json`, `governor_v4.json`) | 4.9865 |

How they were obtained (`ml/train.py`, `apex/policies.py` -> `tune_tail_aware`):
random search of 1,500 weight vectors, each weight drawn uniformly from [0, 2],
on the first 70% of the replay corpus (chronological split). The vector that
minimises SLO violation when the top 30% of scores are offloaded is kept. The
threshold is the 70th percentile of the tuned score on the corpus, which is the
matched 30% offload point (`python -m soc4ai.study recalibrate`).

### 7.2 Trained classifier

`ml/train.py`. Two breach models (TTFT breach and occupancy harm), each fitted
as a calibrated logistic regression and a calibrated XGBoost (120 trees, depth
3); calibration is isotonic when a split has at least 15 positives, otherwise
sigmoid. The model with the better Brier score on a 70/30 chronological split
is kept. Logistic regression won for both targets:

| Target | Model | ROC-AUC | PR-AUC | Brier | ECE |
|---|---|---:|---:|---:|---:|
| TTFT breach | logistic regression (selected) | 0.958 | 0.810 | 0.0657 | 0.0424 |
| TTFT breach | XGBoost | 0.953 | 0.778 | 0.0670 | 0.0183 |
| occupancy | logistic regression (selected) | 0.884 | 0.174 | 0.0197 | 0.0105 |
| occupancy | XGBoost | 0.917 | 0.177 | 0.0212 | 0.0323 |

Features: the 34 pre-decision features in `apex/core.py` -> `FEATURE_NAMES`.
`tests/test_features_leakage.py` asserts that no post-decision field leaks
into them.

### 7.3 Prompt-only router

`analysis/paper/frontier_eval.py` -> `prompt_static`:
`z(max_output_tokens) + 0.5 z(prompt_chars) + is_code`, no telemetry and no
labels. It represents the prompt-signal class of RouteLLM and CARROT, which
need per-request quality labels an overflow service does not have.

### 7.4 The three candidate mechanisms

| Mechanism | Rule | Code |
|---|---|---|
| Fresh-evidence override | the rolling-P95 hard override applies only while the local tier is busy (`pending >= 1` or all slots active) | `soc4ai/paced.py` -> `FreshTailAware`, `local_busy` |
| Paced release | after t of n arrivals, at most `min(0.3n, floor(0.3t) + ceil(0.05n))` cloud calls (15-call head start at n = 300, which covers the first six-request burst) | `soc4ai/paced.py` -> `PacedCloudBudget`, `allowance_for` |
| Recovery-aware accounting | half-open breaker probes are exempt from the quota; failed charged calls are refunded | `soc4ai/v4.py` |

---

## 8. Promotion gates

Fixed in `docs/architecture.md` and in the agreement's `promotion_bar`.

| Gate | Criterion | Why | Promoted candidate |
|---|---|---|---|
| G1 Seed support | at least as good on SLO violation in >= 4 of 5 seeds | a gain must hold across traces | 5 of 5 |
| G2 SLO bound | upper 95% CI bound of violation difference <= +1.0 pt | the agreement's main term must not get worse | -6.0 pt, CI [-14.87, -0.73] |
| G3 Tail gain | P95 TTFT ratio <= 0.95, CI upper < 1.00 | SLOs break in the tail | 0.613, CI [0.471, 0.958] |
| G4 No regression | no seed x workload cell > 5 pt worse | averages hide bad cells | worst seed diff 0.0 pt |
| G5 Decision cap | <= 5 ms added routing time per request | slow signals erase the latency they save | +0.0056 ms |

Per-seed violation differences, candidate minus incumbent (pt): seed 257 -2.3,
263 -3.7, 269 -23.3, 271 0.0, 277 -0.7.

G2 and G3 use a seed-level bootstrap over the five fresh seeds.

---

## 9. Experiments

### 9.1 Replay (offline screening)

- **Corpus:** 4,500 bursty requests = 3 workload mixes (synthetic,
  real_sample, reasoning_coding) x 5 seeds (7, 13, 23, 42, 101) x 300
  requests, near GPU saturation (`data/events/`).
- **All-local ground truth:** every request is served locally and its true
  TTFT is logged; each policy's decision is recorded without being executed
  (shadow mode). A routed log hides the breaches its router prevented: for the
  same requests it reads a local breach rate of 9.3% against the true 35.0%.
- **Policies:** always-local, always-cloud, queue threshold, telemetry
  heuristic, ML TTFT classifier, ML dual-risk classifier, prompt-only router,
  semantic proxy, local-model probe, thinking probe. Each is charged its own
  decision time and compared at equal cloud budgets.
- **Role:** replay selects the scoring rule and the quota; live runs against
  the real Gemini API decide promotion and supply every performance number.

### 9.2 Live experiments

Every run replays a seeded trace against the real GPU and the real Gemini API.

**Arrivals** (`apex/workload.py` -> `_arrival_offsets`, pattern `bursty`):
Poisson, i.e. exponential inter-arrival times. The rate is 1.5 requests per
second for requests 0-5 of every 24-request cycle and 0.3 requests per second
for the other 18.

**Workload mixes** (`apex/workload.py`): `synthetic`, `real_sample`,
`reasoning_coding`; drift runs use a five-phase `transition` trace.

**Conditions:**

| Condition | Requests | What happens |
|---|---:|---|
| Nominal | 300 | bursty load, healthy tiers |
| Faults | 300 | cloud outage on requests 96-155 (every cloud call fails); cloud latency spike of 2.5 x frozen P95 on requests 192-251 |
| Drift | 200 | five-phase workload change; the last phase slows the cloud by the same 2.5 x P95 delay |

**Logged per run:** every routing decision and reason code, TTFT, prefill and
decode timing, streaming chunk times, breaker transitions, cloud charge, and
1 Hz GPU power, temperature, and utilisation.

### 9.3 Run inventory (74 audited runs, 71 preregistered)

| Session | Block | Condition | Seeds (mix) | Policies | Runs | Role |
|---|---|---|---|---|---:|---|
| 1 | `A` | nominal | 211, 223, 227, 229, 233, 239, 241, 251 | incumbent, ungoverned | 16 | preregistered, confirmatory |
| 1 | `B` | faults | 7, 13, 42 | incumbent, ungoverned | 6 | preregistered |
| 1 | `C` | drift | 7, 13, 42 | incumbent, ungoverned | 6 | preregistered |
| 1 | `X` | nominal | 211, 223, 227 | preliminary variant | 3 | exploratory (the only runs outside a preregistered protocol) |
| 2 | `V4N` | nominal | 257, 263, 269, 271, 277 (fresh) | candidate, incumbent, ungoverned | 15 | preregistered, decides the gates |
| 2 | `V4F` | faults | 257, 263, 269, 271, 277 | all three | 15 | preregistered |
| 2 | `B_ext` | faults | 23, 101 | incumbent, ungoverned | 4 | preregistered, extends `B` to five seeds |
| 2 | `C_ext` | drift | 23, 101 | incumbent, ungoverned | 4 | preregistered, extends `C` to five seeds |
| 2 | `V4D` | drift | 7, 13, 23, 42, 101 | candidate | 5 | preregistered, paired with `C` + `C_ext` |

Total: 31 runs in session 1 and 43 in session 2. Drift for the incumbent and
the ungoverned router comes from `C` + `C_ext`, and the candidate's drift runs
use those same five seeds, so drift is compared seed by seed. Every comparison
is made on matched seeds. Each run file is `soc4ai_out/runs/<block>/<block>_s<seed>_<mix>_<policy>.jsonl`
with a `.done` marker.

### 9.4 Preregistration and outcome blinding

- Each session's protocol (seeds, mixes, policy order, fault windows, analysis
  code, decision rule) was frozen and hashed before its first run:
  `soc4ai_out/protocol.json` + `protocol.sha256`, `protocol_x.json` +
  `protocol_x.sha256` (exploratory), `protocol_v4.json` + `protocol_v4.sha256`
  (`b61ca6c2e3e288ff...`).
- The candidate's artifact and agreement were hashed before it ran on seeds
  that had never generated a trace.
- The progress log (`soc4ai_out/progress.log`) recorded only run completion,
  wall time, and spend.
- The analysis code refuses to read a block until every run in it has a
  `.done` marker.
- Dry runs used separate throwaway seeds (`soc4ai_out/dryrun*/`) and are
  excluded from all results.
- Session 2 had a preregistered truncation rule (the nominal block must
  complete; later blocks could be cut for time only). None were cut.
- Live calls ran on a dedicated Google Cloud project. Its ID is withheld from
  the published `protocol.json` (`cloud_tier.project`); every other field is as
  frozen, and `protocol.sha256` is the digest recorded at freeze time.

---

## 10. Reproducing

Tested on Python 3.13, Windows 11. The code is plain Python and should run on
Linux and macOS as well.

```
python -m pip install -r requirements.txt
python -m pytest -q
```

### 10.1 Recompute every paper number (CPU only, a few minutes)

All live run logs ship in `soc4ai_out/runs/`.

```
python -m soc4ai.v4study analyze               # session 2: Table 1 gate results, Table 3, drift
python soc4ai/analyze_live.py --exp A          # session 1 nominal (also --exp B, --exp C)
python analysis/paper/frontier_eval.py         # replay: Table 2, Fig. 3a
python analysis/paper/renegotiate_agreement.py # re-audit at 20/30/40/50% quotas
python analysis/paper/block_boot.py            # heuristic vs classifier CI
python analysis/paper/make_figures.py          # Fig. 3 -> analysis/paper/figures/
python analysis/evaluate.py                    # nine-policy replay table, selective-label numbers
```

`live_V4.json`, `frontier_eval.json`, and `renegotiation.json` regenerate
byte for byte from the shipped data.

### 10.2 Run the gateway locally (GPU, no cloud key)

Needs Ollama running with `llama3.1:8b` pulled. The default cloud-model
calibration path also uses `gemma4:e2b`.

```
ollama pull llama3.1:8b
python tools/preflight.py        # checks setup, prints READY or what is broken, writes nothing
python tools/demo_governor.py    # walks every guard, no GPU or network
python run_all.py --quick        # 3-4 minute end-to-end demo
python -m apex.runner --mode replay --policy guarded_governor --pattern bursty --n 24
```

### 10.3 Re-run the live study (GPU and a Google Cloud project)

Environment variables (never commit a key; `.gitignore` and
`tests/test_secret_hygiene.py` guard against it):

| Variable | Meaning |
|---|---|
| `SOC4AI_GCP_KEY` | path to a service-account JSON key with Vertex AI access (required for live calls) |
| `SOC4AI_GCP_PROJECT` | expected project ID (optional; defaults to the key's `project_id`) |
| `SOC4AI_GCP_LOCATION` | Vertex AI region (optional; default `us-central1`) |
| `SOC4AI_KEY_SHA256_PREFIX` | optional pin: first 16 hex chars of the key file's SHA-256 |

PowerShell: `$env:SOC4AI_GCP_KEY = "C:\path\to\key.json"`.
Bash: `export SOC4AI_GCP_KEY=/path/to/key.json`.

```
python -m soc4ai.study recalibrate  # threshold from the replay corpus (no cloud calls)
python -m soc4ai.study calibrate    # 30 live Gemini calls -> frozen cloud P95
python -m soc4ai.study dryrun       # throwaway seeds, never analysed
python -m soc4ai.study freeze       # writes protocol.json + protocol.sha256
python -m soc4ai.study batch        # session 1 blocks A, B, C
python -m soc4ai.v4study artifact   # builds and hashes the candidate artifact
python -m soc4ai.v4study freeze     # writes protocol_v4.json + protocol_v4.sha256
python -m soc4ai.v4study dryrun
python -m soc4ai.v4study batch      # session 2 blocks
python -m soc4ai.v4study analyze
```

- `freeze` refuses to run on a dirty git tree, so the protocol records exactly
  which code produced the results. Commit first.
- The spend ledger stops a batch at USD 20. The full live study cost USD 2.40.
- A new run writes new protocol files. Move the shipped `soc4ai_out/` aside
  first if you want to keep it.
- Batches cool the GPU below 58 C between runs (`COOLDOWN_C` in
  `soc4ai/study.py`).

---

## 11. Results in full

### 11.1 Replay at equal cloud budgets (Table 2, Fig. 3a)

SLO violation (%) at each cloud-call quota (`analysis/paper/frontier_eval.json`,
500 bootstrap resamples):

| Router | 10% | 20% | 25% | 30% | 40% | 50% | 70% | Min. quota for 10% target | Cost share of always-cloud at that quota |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Telemetry heuristic | 25.0 | 15.1 | 10.6 | 6.5 | 1.9 | 0.7 | 0.0 | 30% | 0.37 |
| Trained classifier | 25.0 | 15.1 | 10.5 | 6.9 | 2.7 | 0.6 | 0.0 | 30% | 0.38 |
| Queue threshold | 25.0 | 15.4 | 10.9 | 7.9 | 2.9 | 2.7 | 1.2 | 30% | 0.34 |
| Prompt-only | 28.8 | 25.4 | 22.1 | 19.3 | 15.8 | 13.2 | 8.8 | 70% | 0.87 |

- Heuristic minus classifier at 30%: -0.38 pt, run-level block-bootstrap 95% CI
  [-1.09, +0.31] (a tie). At 40% the heuristic is better: -0.73 pt, CI
  [-1.33, -0.18].
- Probe interference (paired live A/B, 3 seeds, 18 request pairs): a probe
  sharing the serving GPU added 349 ms mean TTFT, 453 ms P95 TTFT, and 386 ms
  queue wait to the requests it was meant to protect
  (`analysis/results/probe_interference_summary.md`), which is why probe-based
  rules fail G5.

Nine-policy comparison at 30% (cost relative to the heuristic,
`analysis/results/results_table.md`):

| Policy | SLO violation | P99 TTFT | Cost per SLO-met request (rel.) |
|---|---:|---:|---:|
| always local | 35.0% | 29741 ms | 0.00 |
| always cloud | 0.0% | 378 ms | 2.53 |
| queue threshold | 7.4% | 6659 ms | 0.94 |
| telemetry heuristic | 6.5% | 5812 ms | 1.00 |
| ML TTFT | 6.6% | 5812 ms | 1.00 |
| ML dual-risk | 6.9% | 6182 ms | 1.04 |
| semantic proxy | 6.7% | 5999 ms | 1.02 |
| local model probe | 8.2% | 6328 ms | 1.05 |
| thinking probe | 11.2% | 6868 ms | 1.10 |

### 11.2 Session 1: the incumbent misses the SLO

- Nominal, 8 seeds: incumbent 18.5% vs ungoverned 4.7% SLO violation,
  difference +13.8 pt, 95% CI [+4.75, +25.58]. The incumbent was worse in 7 of
  8 seeds; excluding the reasoning-coding seeds it was 9.8% vs 4.3%.
- Audit: the incumbent honoured its quota (90/90 calls), artifact hash, and
  breaker rule in all 8 runs. The ungoverned router breached the quota in 8/8
  runs (~270 calls, ~90% cloud).
- Cause 1 (stale evidence): in every incumbent run, 86 to 89 of the 90 cloud
  calls were hard-breach overrides. Offloaded requests never refresh the local
  rolling P95, so the override stayed true and the quota ran out between
  requests 96 and 144; every later request queued locally.
- Cause 2 (early bursts): nothing paced the budget, so the first bursts took it.
- Cause 3 (outage accounting, from block `B`): failed calls and breaker probes
  counted against the quota, so an outage used up the calls the breaker needed
  to confirm recovery.
- Telemetry: local breaches come from queueing (73% of breached TTFT; median
  prefill 158 ms). The GPU reached 76 C and throttled in 31% of samples. Live
  Gemini over 2,860 calls: p50 665 ms, p95 2.2 s, p99 10.2 s; 2.9% of cloud
  calls missed the SLO on their own.
- Faults (block `B`, 3 seeds): incumbent 24.0% vs ungoverned 41.6%; failed
  requests 23 vs 164; outage detected in 1.6-3.9 s.
- Preliminary variant (block `X`, fresh-evidence override + paced release
  only): 14.6% vs incumbent 23.8% vs ungoverned 3.1% on the same 3 seeds; it
  beat the incumbent on all 3.

### 11.3 Session 2: the candidate is promoted (Table 1, Table 3, Fig. 3b)

Nominal, five fresh seeds (1,500 requests per policy):

| Policy | Viol. % | P95 TTFT (s) | Cloud % | USD (5 runs) | USD per 10k compliant | Failed | Quota met | SLO met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 6.5 | 4.2 | 12.7 | 0.058 | 0.41 | 3 | 5/5 | 5/5 |
| Incumbent | 12.5 | 6.9 | 30.0 | 0.120 | 0.91 | 2 | 5/5 | 4/5 |
| Ungoverned | 3.7 | 3.0 | 82.2 | 0.330 | 2.29 | 12 | 0/5 | 5/5 |

- Candidate vs incumbent: see Section 8 (all five gates pass).
- Ungoverned minus candidate: -2.8 pt violation, CI [-3.8, -1.8], with 199 to
  291 cloud calls per run against a limit of 90, at 5.6 x the cost per
  compliant request.
- Cumulative cloud calls (Fig. 3b): incumbent reaches 89 calls around request
  185; candidate ends near 38; ungoverned near 247.
- GPU energy: 0.041 Wh per compliant request for the candidate.

### 11.4 Faults (Table 3, Fig. 3c)

| Policy | Viol. % | P95 TTFT (s) | Cloud % | USD | USD per 10k compliant | Failed | Quota met | SLO met |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Candidate | 12.2 | 5.9 | 8.6 | 0.031 | 0.24 | 32 | 5/5 | 1/5 |
| Incumbent | 17.3 | 8.1 | 29.5 | 0.108 | 0.87 | 39 | 5/5 | 0/5 |
| Ungoverned | 42.6 | 9.5 | 82.2 | 0.255 | 2.96 | 301 | 0/5 | 0/5 |

- Candidate minus incumbent: -5.13 pt, CI [-8.13, -2.80]; P95 ratio 0.725,
  CI [0.594, 0.912].
- 5 to 9 breaker probes per candidate run were exempt from the quota; no
  request was denied for budget.
- Peak 24-request rolling violation: candidate 25.8%, incumbent 35.8%,
  ungoverned 100%.
- Incumbent vs ungoverned under faults over five seeds (`B` + `B_ext`): 21.1%
  [16.1, 29.5] vs 42.3% [39.8, 44.1]; failed 7.6 vs 57.6 per run; incumbent
  better on every seed.

### 11.5 Drift

Five paired seeds (7, 13, 23, 42, 101), degraded (final) phase:

| Policy | Whole-run viol. | Degraded-phase viol. |
|---|---:|---:|
| Candidate (`V4D`) | 12.0% [6.3, 21.1] | 15% [7, 21] |
| Incumbent (`C` + `C_ext`) | 12.0% [5.8, 22.3] | 13% [5.5, 20] |
| Ungoverned (`C` + `C_ext`) | 25.2% [24.6, 26.1] | 100% |

Candidate minus incumbent: sign-flip p = 1.0 (a tie). The candidate had no
budget denial in any drift run, so its protection comes from routing and the
breaker, not from an exhausted quota.

### 11.6 Renegotiated agreements

`analysis/paper/renegotiate_agreement.py` re-audits every completed log under
quotas of 20/30/40/50% and both counting rules (`renegotiation.json`):

- At a tightened 20% quota (billable counting) the candidate complies in 13 of
  15 runs (4/5 nominal, 5/5 faults, 4/5 drift); the incumbent complies in none.
- The ungoverned router complies in only 2 of its 28 runs even at a 50% quota.

---


## 12. Repository layout

```
apex/                 gateway package
  core.py             config, gateway state (rolling P50/P95/P99), telemetry,
                      the 34-feature builder, labels
  policies.py         the routing policies behind one score(features) method,
                      TailAwareHeuristic, the guarded governor, circuit breaker,
                      tune_tail_aware
  providers.py        Ollama provider, cloud model, calibration, VRAM guard
  gateway.py          request orchestrator
  events.py           JSONL event log, Parquet flattener, run manifests
  workload.py         seeded trace generator (arrival process, mixes, drift)
  runner.py           replays a trace under one policy
soc4ai/               live study
  study.py            session 1 driver (recalibrate, calibrate, dryrun, freeze, batch)
  v4study.py          session 2 driver (artifact, freeze, dryrun, batch, analyze)
  v4.py               candidate governor (recovery-aware accounting)
  paced.py            paced release and fresh-evidence override
  audit.py            agreement auditor
  ledger.py           spend ledger and kill switch
  creds.py            credential guard (environment variables only)
  providers.py        Vertex AI streaming provider, fault injection
  telemetry.py        1 Hz GPU sampling, breaker tracing
  analyze_live.py     per-block analysis
  exploratory.py      block X driver
  paper_numbers.py    helper that prints paper numbers
governance/           agreement.yaml, agreement_v3.yaml, agreement_v4.yaml
ml/                   train.py, risk_model.py, artifacts/ (governor_v1..v4.json,
                      meta.json, *_model.joblib)
replay/               queue-replay simulator, paired audits, probe-interference audit
analysis/             evaluate.py, validity_gates.py, confirmatory_statistics.py,
                      results/ (tables and JSON), figures/
analysis/paper/       scripts behind the paper: frontier_eval.py,
                      renegotiate_agreement.py, block_boot.py, make_figures.py,
                      pipeline.tex, and their JSON outputs
data/                 frozen manifests and the 4,500-request replay corpus
soc4ai_out/           protocols and hashes, spend ledger, calibration,
                      runs/<block>/*.jsonl (every live run), analysis/*.json
deliverables/         simulator validity reports
docs/                 architecture.md (runtime, artifact, the five gates)
examples/             route_with_governor.py (use the promoted policy as a component)
paper/                the SOC4AI 2026 paper (PDF)
tools/                preflight, demo_governor, secret_hygiene, study helpers
tests/                43 tests (python -m pytest -q)
config.yaml           every fixed constant (SLO, concurrency, breaker, thresholds)
```

---

## 13. Testbed hardware and software

| Component | Value |
|---|---|
| Machine | laptop, Windows 11 Home |
| CPU | Intel Core i7-13650HX |
| RAM | 32 GB |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8 GB VRAM, driver 592.15 |
| Local model | `llama3.1:8b` via Ollama, concurrency 2, `num_ctx` 2048 |
| Cloud model | Gemini 2.5 Flash, Vertex AI `us-central1`, streaming, thinking budget 0 |
| Python | 3.13 |

The Ollama version, model digest, and GPU driver are recorded in each
protocol's `environment` block.

---

## 14. Deploying on other hardware or providers

The policy, agreement, and gates are hardware-independent: only the
calibrated thresholds depend on the machine. To deploy on new hardware or a new
provider:

1. Collect a new all-local replay corpus
   (`python -m apex.runner --mode shadow --pattern bursty --n 300`, several seeds).
2. Refreeze the SLO and occupancy targets and retrain/retune
   (`python ml/train.py`), which rewrites `ml/artifacts/meta.json`.
3. Recalibrate the threshold (`python -m soc4ai.study recalibrate`) and the
   cloud P95 (`python -m soc4ai.study calibrate`).
4. Edit the agreement's quota and SLO if the service promises change.
5. Freeze a new protocol and run a promotion cycle; promote only if all five
   gates pass.

To use a different cloud provider, implement a provider with the same
`generate()` interface as `soc4ai/providers.py` -> `LiveGeminiProvider` and keep
the spend ledger in front of it.

---

## 15. Troubleshooting

| Symptom | Fix |
|---|---|
| `SOC4AI_GCP_KEY is not set` | Only live stages need a key. Analysis commands in Section 10.1 run without it. |
| `source repository has uncommitted changes` on `freeze` | Commit, then freeze, so the protocol records a clean tree. |
| pandas fails to import | Reinstall with `python -m pip install -r requirements.txt` in a fresh virtual environment. |
| Slow first run of `evaluate.py` or `make_figures.py` | matplotlib is building its font cache; later runs are fast. |
| `preflight.py` reports a missing model | `ollama pull llama3.1:8b` (and `gemma4:e2b` for the default calibration path). |
| Live batch halts early | The spend ledger reached its cap, or the GPU stayed above the cooldown temperature; check `soc4ai_out/progress.log`. |

---


## 16. Citation

```
@inproceedings{pawar2026promote,
  title     = {Promote, Don't Presume: Preregistered Live Testing of Governed LLM Routing Policies},
  author    = {Pawar, Om Rajesh and Ibrahim, Ahmad},
  booktitle = {Service-Oriented Computing -- ICSOC 2026 Workshops (SOC4AI)},
  publisher = {Springer},
  series    = {Lecture Notes in Computer Science},
  year      = {2026},
  note      = {Accepted}
}
```
