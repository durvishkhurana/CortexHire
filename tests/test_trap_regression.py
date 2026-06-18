"""Trap-regression suite (STEP 17) — synthetic fixture expectations."""

from __future__ import annotations

import os

import pytest

from src import honeypot as hp
from src import parse

FIXTURE = os.path.join(
    os.path.dirname(__file__), "fixtures", "synthetic_candidates.jsonl"
)


@pytest.fixture
def records():
    return list(parse.stream_candidates(FIXTURE))


def test_honeypots_excluded_by_hard_rules(records):
    """CAND_0000007 (date incoherence) and CAND_0000008 (expert+0mo) must hard-fail."""
    by_id = {r["candidate_id"]: r for r in records}
    for cid in ("CAND_0000007", "CAND_0000008"):
        a = hp.run_consistency_suite(by_id[cid], founding_years={}, now=None)
        assert a.hard_violation, cid


def test_keyword_stuffer_not_top_tier_evidence(records):
    """Marketing manager with expert FAISS+0mo skills — low career evidence."""
    from src import features as ft

    c2 = next(r for r in records if r["candidate_id"] == "CAND_0000002")
    row = ft.build_feature_row(c2)
    assert row["claimed_unverified_ratio"] > 0.5 or row["jd_relevant_skill_count"] >= 1


def test_strong_builder_has_evidence(records):
    from src import features as ft

    c1 = next(r for r in records if r["candidate_id"] == "CAND_0000001")
    row = ft.build_feature_row(c1)
    assert row["evidence_density"] > 0
    assert row["is_product_current"] == 1.0
