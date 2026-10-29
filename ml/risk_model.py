from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import FEATURE_NAMES


class CalibratedRiskModel:
    def __init__(self, ttft_model, occ_model, medians: dict, feature_names: list[str]):
        self.ttft_model = ttft_model
        self.occ_model = occ_model
        self.medians = medians
        self.feature_names = feature_names

    @classmethod
    def load(cls, model_dir: str | Path) -> "CalibratedRiskModel":
        d = Path(model_dir)
        meta = json.loads((d / "meta.json").read_text())
        occ = None
        if (d / "occ_model.joblib").exists():
            occ = joblib.load(d / "occ_model.joblib")
        return cls(
            ttft_model=joblib.load(d / "ttft_model.joblib"),
            occ_model=occ,
            medians=meta["medians"],
            feature_names=meta["feature_names"],
        )

    def _vec(self, feat: dict) -> np.ndarray:
        row = []
        for name in self.feature_names:
            v = feat.get(name, float("nan"))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                v = self.medians.get(name, 0.0)
            row.append(float(v))
        return np.array(row, dtype=float).reshape(1, -1)

    def p_ttft(self, feat: dict) -> float:
        if self.ttft_model is None:
            return 0.0
        return float(self.ttft_model.predict_proba(self._vec(feat))[0, 1])

    def p_occ(self, feat: dict) -> float:
        if self.occ_model is None:
            return 0.0
        return float(self.occ_model.predict_proba(self._vec(feat))[0, 1])
