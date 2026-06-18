"""Tests for the feature store: sentinel handling, soft-YoE, has_* flags."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

from src import features as ft
from src import parse

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_candidates.jsonl"
# reference now = max(last_active_date) across the fixture.
NOW = dt.date(2026, 6, 1)
FOUNDING = {"neostartup": 2023}

C1, C2, C3, C4, C5, C6 = (
    "CAND_0000001",
    "CAND_0000002",
    "CAND_0000003",
    "CAND_0000004",
    "CAND_0000005",
    "CAND_0000006",
)


def _rows():
    recs = parse.load_pool(str(FIXTURE))
    return {
        r["candidate_id"]: ft.build_feature_row(r, now=NOW, founding_years=FOUNDING)
        for r in recs
    }


def test_feature_columns_present():
    rows = _rows()
    for row in rows.values():
        assert set(row.keys()) == set(ft.FEATURE_COLUMNS)


def test_sentinel_github_becomes_nan_with_indicator():
    rows = _rows()
    # candidate 2 has github_activity_score == -1
    assert math.isnan(rows[C2]["github_activity_score"])
    assert rows[C2]["has_github"] == 0.0
    # candidate 1 has a real github score
    assert rows[C1]["github_activity_score"] == 80.0
    assert rows[C1]["has_github"] == 1.0


def test_sentinel_offer_acceptance():
    rows = _rows()
    assert math.isnan(rows[C2]["offer_acceptance_rate"])
    assert rows[C2]["has_prior_offers"] == 0.0
    assert rows[C1]["has_prior_offers"] == 1.0


def test_null_grade_and_unknown_tier_sentinels():
    rows = _rows()
    # candidate 6: grade null, tier "unknown", empty assessments
    assert math.isnan(rows[C6]["best_grade_value"])
    assert rows[C6]["has_grade"] == 0.0
    assert math.isnan(rows[C6]["best_edu_tier"])
    assert rows[C6]["has_edu_tier"] == 0.0
    assert rows[C6]["has_assessments"] == 0.0
    # candidate 1: tier_1 -> ordinal 4, present grade + assessments
    assert rows[C1]["has_grade"] == 1.0
    assert rows[C1]["has_edu_tier"] == 1.0
    assert rows[C1]["best_edu_tier"] == 4.0
    assert rows[C1]["best_grade_value"] == 8.5
    assert rows[C1]["has_assessments"] == 1.0


def test_soft_yoe_fit():
    assert ft.soft_yoe_fit(7) == 1.0
    assert ft.soft_yoe_fit(5) == 1.0
    assert 0.0 < ft.soft_yoe_fit(2) < 1.0
    assert 0.0 < ft.soft_yoe_fit(13) < 1.0
    assert math.isnan(ft.soft_yoe_fit(None))


def test_company_size_enum_mapping():
    rows = _rows()
    # candidate 1 current_company_size "1001-5000"
    assert rows[C1]["current_company_size_num"] == 3000.0
    # candidate 4 "5001-10000"
    assert rows[C4]["current_company_size_num"] == 7500.0


def test_location_fit_jd_preference():
    rows = _rows()
    assert rows[C1]["location_fit"] == 1.0  # Pune, India (preferred)
    assert rows[C6]["location_fit"] == 0.0  # Toronto, Canada, not willing


def test_product_company_features():
    rows = _rows()
    # candidate 1 (Razorpay) and 4 (Swiggy) are product companies
    assert rows[C1]["is_product_current"] == 1.0
    assert rows[C4]["is_product_current"] == 1.0
    assert rows[C1]["product_tenure_months"] > 0


def test_claimed_unverified_ratio_high_for_stuffer():
    rows = _rows()
    # candidate 2: FAISS/Pinecone/LangChain all 0 months, no assessments
    assert rows[C2]["claimed_unverified_ratio"] == 1.0
    # candidate 1: skills have duration + Python assessed -> lower
    assert rows[C1]["claimed_unverified_ratio"] < 1.0


def test_recency_more_recent_is_higher():
    rows = _rows()
    # candidate 4 active 2026-06 (now), candidate 5 active 2025-10 -> 4 > 5
    assert rows[C4]["last_active_recency"] > rows[C5]["last_active_recency"]
    assert rows[C4]["last_active_recency"] == 0.0  # active exactly at "now"


def test_jd_skill_coverage():
    rows = _rows()
    assert rows[C1]["jd_relevant_skill_count"] >= 2  # Python, FAISS, etc.
    assert rows[C2]["jd_skill_coverage"] >= 1  # FAISS/Pinecone -> vector_dbs


def test_coherence_flags_in_features():
    rows = _rows()
    assert rows[C3]["n_hard_flags"] >= 1  # founding-year honeypot
    assert rows[C1]["n_hard_flags"] == 0.0


def test_build_feature_frame():
    recs = parse.load_pool(str(FIXTURE))
    df = ft.build_feature_frame(recs, now=NOW, founding_years=FOUNDING)
    assert len(df) == 8
    assert df.columns[0] == parse.F.CANDIDATE_ID
    for col in ft.OFFLINE_FEATURES:
        assert df[col].isna().all()  # offline columns are NaN placeholders


def test_live_features_excludes_offline_columns():
    recs = parse.load_pool(str(FIXTURE))
    live = ft.live_features(recs[0], now=NOW, founding_years=FOUNDING)
    assert set(live.keys()) == set(ft.SCHEMA_FEATURES)
    for col in ft.OFFLINE_FEATURES:
        assert col not in live


def test_write_features_parquet_streams(tmp_path):
    # The writer accepts ANY iterator (here the lazy streaming parser) and must
    # never require materializing the pool. Read back and verify schema + rows.
    import pyarrow.parquet as pq

    out = tmp_path / "features.parquet"
    n = ft.write_features_parquet(
        parse.stream_candidates(str(FIXTURE)),
        str(out),
        now=NOW,
        founding_years=FOUNDING,
        batch_size=3,  # force multiple batches over the 8-row fixture
    )
    assert n == 8
    table = pq.read_table(str(out))
    assert table.num_rows == 8
    assert table.column_names == [parse.F.CANDIDATE_ID] + ft.FEATURE_COLUMNS
    # candidate_id is a string column; ids round-trip
    ids = table.column(parse.F.CANDIDATE_ID).to_pylist()
    assert ids[0] == C1
    # OFFLINE columns are all-NaN placeholders in the store
    for col in ft.OFFLINE_FEATURES:
        vals = table.column(col).to_pylist()
        assert all(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals)
