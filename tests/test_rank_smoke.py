"""End-to-end smoke test for rank.py on the synthetic fixture.

Verifies the replay runs with NO offline artifacts (placeholder scorer +
composer reasoning), produces a validator-clean CSV, excludes honeypots, and
respects deterministic ordering."""

from __future__ import annotations

import csv
from pathlib import Path

import rank
import validate_submission as vs  # organizer's authoritative validator
from src import internal_validate as iv
from src import parse

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_candidates.jsonl"
C1, C3, C7, C8 = "CAND_0000001", "CAND_0000003", "CAND_0000007", "CAND_0000008"


def _read(out_path):
    with open(out_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def test_rank_end_to_end(tmp_path):
    out = tmp_path / "submission.csv"
    # empty artifacts dir -> exercises all graceful-degradation fallbacks
    rc = rank.run(str(FIXTURE), str(out), artifacts_dir=str(tmp_path / "no_artifacts"))
    assert rc == 0
    assert out.exists()

    header, data = _read(out)
    assert header == ["candidate_id", "rank", "score", "reasoning"]
    n_input = len(parse.load_pool(str(FIXTURE)))
    assert 1 <= len(data) <= n_input

    # internal validator passes (sandbox: <100 rows allowed)
    assert iv.validate(str(out), expected_rows=100, allow_fewer=True) == len(data)
    # the organizer's authoritative validator's only complaints are row-count
    # artifacts (it requires exactly 100 rows, and ranks 1..100 each once); no
    # other rule is violated by the sandbox CSV.
    org_errors = vs.validate_submission(str(out))
    rowcount_markers = ("data rows", "must appear exactly once")
    assert all(any(m in e for m in rowcount_markers) for e in org_errors), org_errors


def test_honeypots_excluded(tmp_path):
    # provide a founding-year table so the flagship tenure-vs-founding check
    # can fire (it abstains without the table, by design).
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "founding_years.csv").write_text(
        "company,founding_year\nNeoStartup,2023\n", encoding="utf-8"
    )
    out = tmp_path / "submission.csv"
    rank.run(str(FIXTURE), str(out), artifacts_dir=str(artifacts))
    _, data = _read(out)
    ids = {row[0] for row in data}
    # candidates 3 (founding-year), 7 (end<start), 8 (current+end) are honeypots
    assert C3 not in ids
    assert C7 not in ids
    assert C8 not in ids
    # clean strong candidate present and highly ranked
    assert C1 in ids


def test_honeypot_founding_abstains_without_table(tmp_path):
    # without the founding-year table, candidate 3 cannot be founding-flagged,
    # but date/current honeypots (7, 8) are still excluded.
    out = tmp_path / "submission.csv"
    rank.run(str(FIXTURE), str(out), artifacts_dir=str(tmp_path / "empty"))
    _, data = _read(out)
    ids = {row[0] for row in data}
    assert C7 not in ids and C8 not in ids
    assert C1 in ids


def test_deterministic_repeat(tmp_path):
    out1 = tmp_path / "a.csv"
    out2 = tmp_path / "b.csv"
    rank.run(str(FIXTURE), str(out1), artifacts_dir=str(tmp_path / "x"))
    rank.run(str(FIXTURE), str(out2), artifacts_dir=str(tmp_path / "y"))
    assert out1.read_text(encoding="utf-8") == out2.read_text(encoding="utf-8")


def test_scores_non_increasing_and_six_dp(tmp_path):
    out = tmp_path / "submission.csv"
    rank.run(str(FIXTURE), str(out), artifacts_dir=str(tmp_path / "no_artifacts"))
    _, data = _read(out)
    scores = [float(r[2]) for r in data]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= score <= 100.0 for score in scores)
    for r in data:
        assert len(r[2].split(".")[1]) == 6  # exactly 6 decimal places


def test_display_scores_are_bounded_and_ordered():
    scores = rank._display_scores(
        [("CAND_0000001", 101.0), ("CAND_0000002", 100.5), ("CAND_0000003", 89.0)]
    )
    assert scores[0] == 100.0
    assert scores[-1] == 70.0
    assert scores == sorted(scores, reverse=True)
