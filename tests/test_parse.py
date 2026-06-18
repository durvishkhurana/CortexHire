"""Tests for the streaming parser, schema helpers, and reference "now"."""

from __future__ import annotations

import datetime as dt
import types
from pathlib import Path

from src import parse

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_candidates.jsonl"


def test_stream_is_lazy_generator(tmp_path):
    # A generator must be returned (not a list) and must NOT parse the whole
    # file eagerly: a broken 2nd line should not prevent reading the 1st record.
    p = tmp_path / "two.jsonl"
    p.write_text('{"candidate_id": "a"}\nNOT_JSON\n', encoding="utf-8")
    gen = parse.stream_candidates(str(p))
    assert isinstance(gen, types.GeneratorType)
    first = next(gen)  # works without touching the broken line
    assert first["candidate_id"] == "a"


def test_projection_drops_unknown_fields(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"candidate_id": "a", "secret_internal_field": 1, '
        '"profile": {"summary": "hi", "junk": 9}}\n',
        encoding="utf-8",
    )
    rec = next(parse.stream_candidates(str(p)))
    assert "secret_internal_field" not in rec
    assert "junk" not in rec  # unknown profile sub-key dropped
    assert rec["candidate_id"] == "a"
    assert rec["summary"] == "hi"  # profile.* flattened to top level


def test_flatten_lifts_profile_and_signals(tmp_path):
    p = tmp_path / "f.jsonl"
    p.write_text(
        '{"candidate_id": "a", "profile": {"years_of_experience": 7}, '
        '"redrob_signals": {"last_active_date": "2026-01-01", '
        '"recruiter_response_rate": 0.5}, "skills": [{"name": "Python"}]}\n',
        encoding="utf-8",
    )
    rec = next(parse.stream_candidates(str(p)))
    assert rec[parse.F.YEARS_OF_EXPERIENCE] == 7
    assert rec[parse.F.LAST_ACTIVE_DATE] == "2026-01-01"
    assert rec[parse.F.RECRUITER_RESPONSE_RATE] == 0.5
    assert rec[parse.F.SKILLS][0]["name"] == "Python"


def test_blank_lines_skipped(tmp_path):
    p = tmp_path / "b.jsonl"
    p.write_text('{"candidate_id": "a"}\n\n\n{"candidate_id": "b"}\n', encoding="utf-8")
    recs = list(parse.stream_candidates(str(p)))
    assert [r["candidate_id"] for r in recs] == ["a", "b"]


def test_fixture_loads_eight_candidates():
    recs = parse.load_pool(str(FIXTURE))
    assert len(recs) == 8
    assert recs[0]["candidate_id"] == "CAND_0000001"


def test_parse_date_formats():
    assert parse.parse_date("2020-06-15") == dt.date(2020, 6, 15)
    assert parse.parse_date("2020-06") == dt.date(2020, 6, 1)
    assert parse.parse_date("2020") == dt.date(2020, 1, 1)
    assert parse.parse_date("2020/06/15") == dt.date(2020, 6, 15)
    assert parse.parse_date("2020-06-15T08:30:00") == dt.date(2020, 6, 15)
    assert parse.parse_date(dt.date(2021, 1, 1)) == dt.date(2021, 1, 1)


def test_parse_date_missingish():
    for v in [None, "", "present", "Current", "n/a", "garbage", 12345]:
        assert parse.parse_date(v) is None


def test_reference_now_picks_max():
    dates = ["2024-01-01", "2025-12-01", "2023-06-01", None, "present"]
    assert parse.reference_now(dates) == dt.date(2025, 12, 1)


def test_reference_now_from_path():
    # reference now = max(last_active_date) across the fixture pool.
    assert parse.reference_now_from_path(str(FIXTURE)) == dt.date(2026, 6, 1)


def test_sentinel_helpers():
    assert parse.is_numeric_sentinel(-1)
    assert not parse.is_numeric_sentinel(0)
    assert not parse.is_numeric_sentinel(None)
    assert parse.is_missing_tier("unknown")
    assert parse.is_missing_tier(None)
    assert not parse.is_missing_tier("3")


def test_months_between():
    m = parse.months_between(dt.date(2020, 1, 1), dt.date(2021, 1, 1))
    assert abs(m - 12) < 0.1
    assert parse.months_between(None, dt.date(2021, 1, 1)) is None


def test_candidate_sort_key_matches_organizer_string_order():
    # CAND_ ids are fixed-width, so string order == numeric order, which is
    # what the organizer validator enforces (it compares ids as strings).
    assert parse.candidate_sort_key("CAND_0000008") < parse.candidate_sort_key(
        "CAND_0000010"
    )
    assert parse.candidate_sort_key("CAND_0000001") < parse.candidate_sort_key(
        "CAND_9999999"
    )
