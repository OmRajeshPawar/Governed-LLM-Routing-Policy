"""Use the promoted governed routing policy as a component.

Loads the sealed candidate artifact (ml/artifacts/governor_v4.json), verifies
its hash against the agreement, builds the governor with the tuned heuristic
weights, and routes a few requests. No GPU, Ollama, or cloud key is needed.

    python examples/route_with_governor.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from apex.core import load_config  # noqa: E402
from apex.policies import GovernorArtifact  # noqa: E402
from soc4ai.v4 import make_v4_governor  # noqa: E402

ARTIFACT = "ml/artifacts/governor_v4.json"
AGREEMENT = ROOT / "governance" / "agreement_v4.yaml"
META = ROOT / "ml" / "artifacts" / "meta.json"


def main() -> None:
    agreement = yaml.safe_load(AGREEMENT.read_text(encoding="utf-8"))
    sealed = agreement["governance"]["artifact"]["sha256"]
    actual = GovernorArtifact.sha256(ROOT / ARTIFACT)
    if actual != sealed:
        raise SystemExit(f"artifact hash {actual[:12]} does not match the agreement {sealed[:12]}")
    print(f"artifact sealed and verified: {actual[:16]}...")

    cfg = load_config()
    cfg.governor.artifact_path = ARTIFACT
    weights = json.loads(META.read_text(encoding="utf-8"))["heuristic_weights"]
    calibration = json.loads((ROOT / "soc4ai_out" / "calibration.json").read_text(encoding="utf-8"))
    frozen_p95 = float(calibration["ttft_p95_ms"])
    gov = make_v4_governor(cfg, episode_requests=300, frozen_cloud_p95_ms=frozen_p95,
                           heuristic_weights=weights)

    base = {name: 0.0 for name in cfg_feature_names()}
    base.update({"prompt_token_est": 180, "max_output_tokens": 96, "burst_ratio": 1.0,
                 "vram_used_mb": 5900, "vram_free_mb": 2200, "gpu_util": 60,
                 "arr_rate_5s": 0.5, "cloud_success_rate": 1.0})
    scenarios = {
        "idle local tier": {"pending_local": 0, "active_local": 0,
                            "roll_ttft_p95": 900, "roll_service_p95": 1800},
        "busy, short queue": {"pending_local": 1, "active_local": 2,
                              "roll_ttft_p95": 3000, "roll_service_p95": 4000},
        "busy, queue building": {"pending_local": 3, "active_local": 2, "burst_ratio": 2.0,
                                 "roll_ttft_p95": 3500, "roll_service_p95": 4600},
        "burst, deep queue": {"pending_local": 5, "active_local": 2, "burst_ratio": 3.0,
                              "roll_ttft_p95": 6400, "roll_service_p95": 4600},
    }
    threshold = gov.artifact.thresholds["tail_aware"]
    for name, overrides in scenarios.items():
        feat = {**base, **overrides}
        d = gov.decide(feat, threshold)
        score = "guard" if d.score > 1e6 else f"{d.score:.2f}"
        print(f"{name:22s} -> {d.route:5s}  score={score:>6s}  reason={d.reason_code}")
    print(f"cloud calls charged so far: {gov.budget.total_cloud} of {gov.budget.cap}")


def cfg_feature_names() -> list[str]:
    from apex.core import FEATURE_NAMES
    return FEATURE_NAMES


if __name__ == "__main__":
    main()
