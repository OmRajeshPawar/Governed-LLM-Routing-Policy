from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from apex.core import load_config, _repo_root
from apex.events import load_events, resolve_event_files
from apex.core import FEATURE_NAMES
from apex.core import freeze_thresholds, ttft_breach, occupancy_pressure
from apex.policies import tune_tail_aware

COMPARISON_BUDGET = 0.30


def expected_calibration_error(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += abs(p[m].mean() - y[m].mean()) * m.sum() / n
    return float(ece)


def _fit_calibrated(X_tr, y_tr, kind: str):
    if len(np.unique(y_tr)) < 2:
        return None
    n_pos = int(y_tr.sum())
    cv = max(2, min(3, n_pos))
    method = "isotonic" if n_pos >= 15 else "sigmoid"
    if kind == "logreg":
        base = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("sc", StandardScaler()),
                         ("clf", LogisticRegression(max_iter=2000, class_weight="balanced"))])
    else:
        base = Pipeline([("imp", SimpleImputer(strategy="median", keep_empty_features=True)),
                         ("clf", XGBClassifier(n_estimators=120, max_depth=3,
                                               learning_rate=0.1, subsample=0.9,
                                               eval_metric="logloss"))])
    return CalibratedClassifierCV(base, method=method, cv=cv).fit(X_tr, y_tr)


def _evaluate(model, X_te, y_te) -> dict:
    if model is None or len(np.unique(y_te)) < 2:
        return {"pr_auc": None, "roc_auc": None, "brier": None, "ece": None}
    p = model.predict_proba(X_te)[:, 1]
    return {
        "pr_auc": round(float(average_precision_score(y_te, p)), 3),
        "roc_auc": round(float(roc_auc_score(y_te, p)), 3),
        "brier": round(float(brier_score_loss(y_te, p)), 4),
        "ece": round(expected_calibration_error(np.asarray(y_te), p), 4),
    }


def _train_target(name, X_tr, y_tr, X_te, y_te) -> tuple:
    out = {}
    best, best_brier, best_kind = None, np.inf, None
    first_available, first_kind = None, None
    for kind in ("logreg", "xgboost"):
        m = _fit_calibrated(X_tr, y_tr, kind)
        if m is not None and first_available is None:
            first_available, first_kind = m, kind
        ev = _evaluate(m, X_te, y_te)
        out[kind] = ev
        if m is not None and ev["brier"] is not None and ev["brier"] < best_brier:
            best, best_brier, best_kind = m, ev["brier"], kind
    if best is None:
        best, best_kind = first_available, first_kind
    print(f"  [{name}] positives train/test = {int(y_tr.sum())}/{int(y_te.sum())}")
    for kind in ("logreg", "xgboost"):
        print(f"    {kind:8} PR-AUC={out[kind]['pr_auc']} ROC-AUC={out[kind]['roc_auc']} "
              f"Brier={out[kind]['brier']} ECE={out[kind]['ece']}")
    print(f"    -> selected: {best_kind}")
    return best, best_kind, out


def main():
    cfg = load_config()
    events_dir = _repo_root() / cfg.paths.events_dir
    shadow_files, shadow_source = resolve_event_files(events_dir)
    if not shadow_files:
        raise SystemExit(f"No shadow files matched {shadow_source!r}. Run shadow collection first.")
    df = pd.concat([load_events(f) for f in shadow_files], ignore_index=True)
    print(f"Loaded {len(df)} shadow rows from {len(shadow_files)} run(s), source={shadow_source}.")

    fr = freeze_thresholds(df.end_to_end_ttft_ms.tolist(), df.service_ms.tolist())
    df["y_ttft"] = df.end_to_end_ttft_ms.apply(lambda x: ttft_breach(x, fr["sla_ttft_ms"], True))
    df["y_occ"] = df.service_ms.apply(lambda x: occupancy_pressure(x, fr["occupancy_service_ms"]))
    print(f"Frozen thresholds: SLA={fr['sla_ttft_ms']}ms occupancy={fr['occupancy_service_ms']}ms "
          f"| breach={df.y_ttft.mean():.2f} occ={df.y_occ.mean():.2f}")

    cfg.targets.sla_ttft_ms = fr["sla_ttft_ms"]
    cfg.targets.occupancy_service_ms = fr["occupancy_service_ms"]

    df = df.sort_values(["run_id", "trace_pos"]).reset_index(drop=True)
    cut = int(len(df) * 0.7)
    tr, te = df.iloc[:cut], df.iloc[cut:]
    X_tr, X_te = tr[FEATURE_NAMES].values, te[FEATURE_NAMES].values
    medians = {k: (float(np.nanmedian(df[k])) if np.isfinite(np.nanmedian(df[k])) else 0.0)
               for k in FEATURE_NAMES}

    tr_feats = tr[FEATURE_NAMES].to_dict("records")
    tr_breach = (tr.end_to_end_ttft_ms.values > fr["sla_ttft_ms"]).astype(int)
    heuristic_weights = tune_tail_aware(cfg, tr_feats, tr_breach, COMPARISON_BUDGET)
    print(f"Tuned tail-aware weights: "
          f"{ {k: round(v,2) for k,v in heuristic_weights.items()} }")

    print("\nTraining TTFT-breach risk model:")
    ttft_model, ttft_kind, ttft_metrics = _train_target("ttft", X_tr, tr.y_ttft.values,
                                                         X_te, te.y_ttft.values)
    print("Training occupancy-harm risk model:")
    occ_model, occ_kind, occ_metrics = _train_target("occ", X_tr, tr.y_occ.values,
                                                      X_te, te.y_occ.values)

    out_dir = _repo_root() / cfg.paths.models_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(ttft_model, out_dir / "ttft_model.joblib")
    if occ_model is not None:
        joblib.dump(occ_model, out_dir / "occ_model.joblib")
    (out_dir / "meta.json").write_text(json.dumps({
        "feature_names": FEATURE_NAMES, "medians": medians,
        "frozen_thresholds": fr,
        "heuristic_weights": heuristic_weights,
        "comparison_budget": COMPARISON_BUDGET,
        "selected": {"ttft": ttft_kind, "occ": occ_kind},
        "metrics": {"ttft": ttft_metrics, "occ": occ_metrics},
        "n_rows": len(df),
        "shadow_source": shadow_source,
        "shadow_glob": os.getenv("APEX_SHADOW_GLOB", "shadow_*.jsonl"),
        "shadow_files": [p.name for p in shadow_files],
    }, indent=2))
    print(f"\nSaved models + meta to {out_dir}")


if __name__ == "__main__":
    main()
