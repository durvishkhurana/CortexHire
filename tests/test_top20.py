"""Top-20 uncertainty penalty (STEP 13)."""

from __future__ import annotations

from src import top20 as t20


def test_uncertainty_penalty_increases_with_risk():
    low = {"claimed_unverified_ratio": 0.0, "n_soft_flags": 0.0, "yoe_fit": 1.0}
    high = {"claimed_unverified_ratio": 0.9, "n_soft_flags": 2.0, "yoe_fit": 0.2}
    assert t20.uncertainty_penalty(high) > t20.uncertainty_penalty(low)


def test_apply_top20_reorders_band_only():
    ranked = [(f"CAND_{i:07d}", 1.0 - i * 0.001) for i in range(30)]
    feats = {
        ranked[0][0]: {"claimed_unverified_ratio": 0.8, "n_soft_flags": 1.0},
        ranked[1][0]: {"claimed_unverified_ratio": 0.0, "n_soft_flags": 0.0},
    }
    out = t20.apply_top20_penalty(ranked[:25], feats)
    assert out[0][0] == ranked[1][0] or out[0][1] >= out[1][1]
    assert [c for c, _ in out[20:]] == [c for c, _ in ranked[20:25]]
