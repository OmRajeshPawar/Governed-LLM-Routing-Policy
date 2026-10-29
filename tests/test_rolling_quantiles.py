from apex.core import RollingQuantiles


def test_ewma_fallback_below_min_samples():
    rq = RollingQuantiles(window=50, min_samples=8, ewma_alpha=0.3)
    assert rq.p95() is None
    rq.add(100)
    assert rq.p95() is not None
    assert rq.count == 1


def test_empirical_quantile_after_min_samples():
    rq = RollingQuantiles(window=50, min_samples=8, ewma_alpha=0.3)
    for v in range(1, 21):
        rq.add(v)
    p50, p95, p99 = rq.p50(), rq.p95(), rq.p99()
    assert p50 < p95 <= p99
    assert 9 <= p50 <= 12


def test_window_is_bounded():
    rq = RollingQuantiles(window=10, min_samples=3, ewma_alpha=0.3)
    for v in range(100):
        rq.add(v)
    assert rq.count == 10
    assert rq.p50() >= 90
