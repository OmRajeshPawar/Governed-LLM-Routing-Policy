# Where each number lives

Two directories, two different jobs. Worth knowing which is which before
quoting a figure.

## `analysis/` - regenerated

`analysis/evaluate.py` overwrites these every time it runs:

- `results_table.md` - the main static all-policy table
- `signal_cost_results_table.md`
- `summary_metrics.json` - includes `shadow_source` and the exact list of
  shadow files the run consumed, so a table can always be traced back to
  the corpus that produced it
- `figures/*.png`

Reproduce them from the checked-in corpus:

```
set APEX_SHADOW_GLOB=shadow_*_bursty_n300_s*.jsonl
python analysis/evaluate.py
```

## `analysis/results/` - curated, not regenerated

Frozen artifacts from runs that are not part of the main evaluate path, plus
a dated snapshot of the three files above. Nothing here is overwritten by
`evaluate.py`:

- `queue_replay_results.{csv,md}` - fixed-service-time scenario analysis
- `probe_interference_{summary,deltas,rows}.csv` - the live T3 probe A/B
- `deep_{signal_tax,headroom_captured}.csv`, `oracle_headroom_captured.csv`
- `vertex_validation_summary.json`, `cloud_streaming_validity_summary.json`
  - the bounded live-provider samples behind the cloud-proxy accuracy check,
  collected once and never re-run
- `deep_dive_insights.md`

## If the two copies disagree

They currently do not: every numeric row in the duplicated tables is
identical, and `summary_metrics.json` differs only in that the regenerated
copy carries the extra `shadow_source` / `shadow_files` provenance.

**Quote `analysis/`.** It is the copy the code produces and the one
`README.md` points at. The snapshot under `results/` is kept because the
live-validation files next to it are not reproducible without spending
provider budget again, and splitting them up would be worse.
