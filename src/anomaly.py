"""IsolationForest honeypot anomaly guard (offline fit, replay load)."""

from __future__ import annotations

import json
import os

import numpy as np

ANOMALY_FEATURES = [
    "n_hard_flags",
    "n_soft_flags",
    "claimed_unverified_ratio",
    "years_of_experience",
    "mean_tenure_months",
    "jd_skill_coverage",
    "evidence_density",
    "title_chasing_flag",
    "no_recent_ic_flag",
]


def _impute_nan_median(X: np.ndarray) -> np.ndarray:
    out = X.astype(float, copy=True)
    for j in range(out.shape[1]):
        col = out[:, j]
        mask = np.isnan(col)
        if not mask.any():
            continue
        med = float(np.nanmedian(col))
        if np.isnan(med):
            med = 0.0
        col[mask] = med
        out[:, j] = col
    return out


def fit_isolation_forest(X: np.ndarray, *, seed: int = 42):
    from sklearn.ensemble import IsolationForest

    X = _impute_nan_median(X)
    medians = [float(np.nanmedian(X[:, j])) if not np.all(np.isnan(X[:, j])) else 0.0 for j in range(X.shape[1])]
    clf = IsolationForest(
        n_estimators=200,
        contamination=0.002,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(X)
    clf._fit_impute_medians_ = medians
    return clf


def _impute_with_medians(X: np.ndarray, medians: list[float]) -> np.ndarray:
    out = X.astype(float, copy=True)
    for j, med in enumerate(medians):
        mask = np.isnan(out[:, j])
        if mask.any():
            out[mask, j] = med
    return out


def save_model(clf, directory: str, feature_names: list[str]) -> None:
    import pickle

    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, "isolation_forest.pkl"), "wb") as fh:
        pickle.dump(clf, fh)
    with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
        medians = getattr(clf, "_fit_impute_medians_", None)
        json.dump(
            {"feature_names": feature_names, "impute_medians": medians},
            fh,
            indent=2,
        )


def load_model(directory: str):
    import pickle

    with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    with open(os.path.join(directory, "isolation_forest.pkl"), "rb") as fh:
        clf = pickle.load(fh)
    medians = meta.get("impute_medians")
    if medians is not None:
        clf._hackathon_impute_medians_ = medians  # noqa: SLF001
    return clf, meta["feature_names"]


def anomaly_flags(clf, rows: list[dict], feature_names: list[str]) -> list[bool]:
    X = np.array(
        [[r.get(c, np.nan) for c in feature_names] for r in rows], dtype=float
    )
    medians = getattr(clf, "_fit_impute_medians_", None)
    if medians is not None:
        X = _impute_with_medians(X, medians)
    else:
        X = _impute_nan_median(X)
    pred = clf.predict(X)
    return [p == -1 for p in pred]
