from __future__ import annotations

from replay.queue_replay import parse_trace_meta


def test_queue_replay_parses_variant_shadow_run_id() -> None:
    meta = parse_trace_meta("shadow_main_llama31_8b_synthetic_bursty_n600_s42")
    assert meta.mix == "synthetic"
    assert meta.pattern == "bursty"
    assert meta.n == 600
    assert meta.seed == 42


def test_queue_replay_parses_legacy_shadow_run_id() -> None:
    meta = parse_trace_meta("shadow_bursty_n48_s7")
    assert meta.mix == "synthetic"
    assert meta.pattern == "bursty"
    assert meta.n == 48
    assert meta.seed == 7
