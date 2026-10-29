from apex.core import FEATURE_NAMES
from apex.events import OUTCOME_FIELDS, LABEL_FIELDS


def test_features_disjoint_from_outcomes():
    leaked = set(FEATURE_NAMES) & set(OUTCOME_FIELDS + LABEL_FIELDS)
    assert not leaked, f"post-decision fields leaked into features: {leaked}"


def test_no_obvious_outcome_terms_in_feature_names():
    forbidden = ["end_to_end", "service_ms", "provider_ttft",
                 "cost_usd", "_breach", "occupancy_pressure", "executed_on"]
    for name in FEATURE_NAMES:
        for term in forbidden:
            assert term not in name, f"feature '{name}' looks post-decision ('{term}')"


def test_feature_names_unique_and_ordered():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_build_features_returns_canonical_keys():
    import time
    from apex.core import load_config
    from apex.core import GatewayState
    from apex.core import Request, build_features
    cfg = load_config()
    st = GatewayState(cfg)
    now = time.perf_counter()
    req = Request("r", "hello world", 64, "qa", now)
    feats = build_features(st, req, now, telemetry={})
    assert list(feats.keys()) == FEATURE_NAMES
