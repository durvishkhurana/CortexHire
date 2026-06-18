"""Tests for the LightGBM student: ensemble, save/load, and — critically —
that monotone constraints are honored on the pinned LightGBM version."""

from __future__ import annotations

import numpy as np
import pytest

from src import model as m

lgb = pytest.importorskip("lightgbm")

RNG = np.random.default_rng(0)

# Feature order: two constrained features + one free noise feature.
FEATURES = ["recruiter_response_rate", "notice_period_days", "noise"]
# recruiter_response_rate -> +1 (up), notice_period_days -> -1 (down)


def _synthetic(n=400):
    rr = RNG.uniform(0, 1, n)  # should push score up
    notice = RNG.uniform(0, 120, n)  # should push score down
    noise = RNG.uniform(0, 1, n)
    # target broadly follows the constrained directions + noise
    y = 50 + 40 * rr - 0.3 * notice + 10 * noise + RNG.normal(0, 5, n)
    X = np.column_stack([rr, notice, noise])
    return X, y


def test_build_monotone_vector():
    vec = m.build_monotone_vector(FEATURES)
    assert vec == [1, -1, 0]


def test_monotone_constraints_are_honored():
    X, y = _synthetic()
    ens = m.train_ensemble(
        X, y, FEATURES, head=m.HEAD_POINTWISE, seeds=(7,), num_boost_round=120
    )

    # Sweep the +1 feature with the others fixed -> predictions non-decreasing.
    base = np.array([X[:, 0].mean(), X[:, 1].mean(), X[:, 2].mean()])
    grid_up = np.tile(base, (25, 1))
    grid_up[:, 0] = np.linspace(0, 1, 25)
    preds_up = ens.predict(grid_up)
    assert np.all(np.diff(preds_up) >= -1e-9), "recruiter_response_rate not monotone up"

    # Sweep the -1 feature -> predictions non-increasing.
    grid_dn = np.tile(base, (25, 1))
    grid_dn[:, 1] = np.linspace(0, 120, 25)
    preds_dn = ens.predict(grid_dn)
    assert np.all(np.diff(preds_dn) <= 1e-9), "notice_period_days not monotone down"


def test_ensemble_predict_shape_and_averaging():
    X, y = _synthetic(200)
    ens = m.train_ensemble(X, y, FEATURES, seeds=(1, 2, 3), num_boost_round=40)
    assert len(ens.boosters) == 3
    preds = ens.predict(X)
    assert preds.shape == (200,)
    # average equals mean of individual booster predictions
    indiv = np.mean([b.predict(X) for b in ens.boosters], axis=0)
    assert np.allclose(preds, indiv)


def test_feature_importances():
    X, y = _synthetic(200)
    ens = m.train_ensemble(X, y, FEATURES, seeds=(1,), num_boost_round=40)
    imp = ens.feature_importances()
    assert set(imp.keys()) == set(FEATURES)


def test_save_load_roundtrip(tmp_path):
    X, y = _synthetic(200)
    ens = m.train_ensemble(X, y, FEATURES, seeds=(1, 2), num_boost_round=40)
    out = tmp_path / "model"
    ens.save(str(out))
    loaded = m.Ensemble.load(str(out))
    assert loaded.feature_names == FEATURES
    assert loaded.head == m.HEAD_POINTWISE
    assert np.allclose(loaded.predict(X), ens.predict(X))


def test_lambdarank_head_trains():
    X, y = _synthetic(120)
    rel = np.clip((y - y.min()) / (y.max() - y.min()) * 5, 0, 5).astype(int)
    ens = m.train_ensemble(
        X,
        rel,
        FEATURES,
        head=m.HEAD_LAMBDARANK,
        seeds=(1,),
        num_boost_round=40,
        group=[len(rel)],
    )
    preds = ens.predict(X)
    assert preds.shape == (120,)
    assert np.all(np.isfinite(preds))


def test_nan_features_are_handled():
    X, y = _synthetic(150)
    X[::10, 2] = np.nan  # inject NaNs into the noise column
    ens = m.train_ensemble(X, y, FEATURES, seeds=(1,), num_boost_round=40)
    assert np.all(np.isfinite(ens.predict(X)))
