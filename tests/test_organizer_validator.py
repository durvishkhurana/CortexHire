"""Tests confirming the ORGANIZER's authoritative validator is adopted verbatim.

These assert the exact rules the grader's Stage-1 auto-validator enforces:
header, exactly 100 data rows, CAND_ id pattern + uniqueness, ranks 1..100,
non-increasing score, and candidate_id-ascending tie-break.
"""

from __future__ import annotations

import csv
from pathlib import Path

import validate_submission as vs


def _write(path: Path, rows, header=None):
    header = header or ["candidate_id", "rank", "score", "reasoning"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    return path


def _full_rows(n=100):
    rows = []
    for i in range(n):
        cid = f"CAND_{i + 1:07d}"
        score = round(1.0 - i * 0.001, 6)
        rows.append([cid, i + 1, f"{score:.6f}", f"reason {i}"])
    return rows


def test_valid_full_submission_has_no_errors(tmp_path):
    p = _write(tmp_path / "team_x.csv", _full_rows(100))
    assert vs.validate_submission(str(p)) == []


def test_requires_exactly_100_rows(tmp_path):
    p = _write(tmp_path / "team_x.csv", _full_rows(99))
    errors = vs.validate_submission(str(p))
    assert any("100" in e and "data rows" in e for e in errors)


def test_candidate_id_pattern_enforced(tmp_path):
    rows = _full_rows(100)
    rows[0][0] = "c1"  # not CAND_XXXXXXX
    p = _write(tmp_path / "team_x.csv", rows)
    errors = vs.validate_submission(str(p))
    assert any("CAND_XXXXXXX" in e for e in errors)


def test_non_csv_extension_rejected(tmp_path):
    p = _write(tmp_path / "team_x.txt", _full_rows(100))
    errors = vs.validate_submission(str(p))
    assert any(".csv extension" in e for e in errors)


def test_increasing_score_rejected(tmp_path):
    rows = _full_rows(100)
    rows[1][2] = "2.000000"  # rank 2 higher than rank 1
    p = _write(tmp_path / "team_x.csv", rows)
    errors = vs.validate_submission(str(p))
    assert any("non-increasing" in e for e in errors)


def test_tie_break_candidate_id_ascending(tmp_path):
    rows = _full_rows(100)
    # make ranks 1 and 2 tie on score but put a larger id first
    rows[0][0], rows[1][0] = "CAND_0000050", "CAND_0000002"
    rows[0][2] = rows[1][2] = "0.999000"
    p = _write(tmp_path / "team_x.csv", rows)
    errors = vs.validate_submission(str(p))
    assert any("tie-break" in e for e in errors)
