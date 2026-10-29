from apex.core import ttft_breach, occupancy_pressure


def test_ttft_breach_basic():
    assert ttft_breach(2500, sla_ms=2000, ok=True) == 1
    assert ttft_breach(1500, sla_ms=2000, ok=True) == 0
    assert ttft_breach(1500, sla_ms=2000, ok=False) == 1
    assert ttft_breach(None, sla_ms=2000, ok=True) == 1


def test_occupancy_pressure():
    assert occupancy_pressure(9500, occupancy_ms=9000) == 1
    assert occupancy_pressure(8000, occupancy_ms=9000) == 0
    assert occupancy_pressure(None, occupancy_ms=9000) == 0
