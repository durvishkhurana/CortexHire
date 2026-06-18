"""Tests for the internal (stricter) submission validator.

The authoritative organizer validator is tested in
``tests/test_organizer_validator.py``; this file covers our extra checks
(decimal places, all-identical scores, embedded newlines, sandbox allow_fewer).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import internal_validate as vs  # noqa: E402


def _write_csv(path: Path, rows, header=vs.HEADER) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _good_rows(n=3):
    # strictly decreasing scores, 6 dp
    rows = []
    for i in range(n):
        score = round(1.0 - i * 0.1, 6)
        rows.append([f"c{i}", i + 1, f"{score:.6f}", f"reason {i}"])
    return rows


def test_valid_small_csv(tmp_path):
    p = _write_csv(tmp_path / "s.csv", _good_rows(3))
    assert vs.validate(str(p), expected_rows=3) == 3


def test_main_exit_zero_on_valid(tmp_path):
    p = _write_csv(tmp_path / "s.csv", _good_rows(3))
    assert vs.main([str(p), "--expected-rows", "3"]) == 0


def test_wrong_header(tmp_path):
    p = _write_csv(
        tmp_path / "s.csv", _good_rows(3), header=["id", "rank", "score", "reasoning"]
    )
    with pytest.raises(vs.ValidationError, match="header"):
        vs.validate(str(p), expected_rows=3)


def test_wrong_row_count(tmp_path):
    p = _write_csv(tmp_path / "s.csv", _good_rows(3))
    with pytest.raises(vs.ValidationError, match="exactly 100"):
        vs.validate(str(p), expected_rows=100)


def test_allow_fewer(tmp_path):
    p = _write_csv(tmp_path / "s.csv", _good_rows(3))
    assert vs.validate(str(p), expected_rows=100, allow_fewer=True) == 3


def test_ranks_must_start_at_one(tmp_path):
    rows = _good_rows(3)
    rows[0][1] = 0
    rows[1][1] = 1
    rows[2][1] = 2
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="ranks must be exactly"):
        vs.validate(str(p), expected_rows=3)


def test_ranks_out_of_order(tmp_path):
    rows = _good_rows(3)
    rows[0][1], rows[1][1] = 2, 1  # swap order but scores still decreasing
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="ordered by rank"):
        vs.validate(str(p), expected_rows=3)


def test_duplicate_candidate_id(tmp_path):
    rows = _good_rows(3)
    rows[1][0] = rows[0][0]
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="unique"):
        vs.validate(str(p), expected_rows=3)


def test_increasing_score_rejected(tmp_path):
    rows = _good_rows(3)
    rows[0][2] = "0.100000"
    rows[1][2] = "0.500000"
    rows[2][2] = "0.900000"
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="score increases"):
        vs.validate(str(p), expected_rows=3)


def test_all_scores_identical_rejected(tmp_path):
    rows = _good_rows(3)
    for r in rows:
        r[2] = "0.500000"
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="identical"):
        vs.validate(str(p), expected_rows=3)


def test_more_than_6_decimals_rejected(tmp_path):
    rows = _good_rows(3)
    rows[0][2] = "0.1234567"  # 7 dp
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="decimal places"):
        vs.validate(str(p), expected_rows=3)


def test_scientific_notation_rejected(tmp_path):
    rows = _good_rows(3)
    rows[0][2] = "1e-3"
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="scientific"):
        vs.validate(str(p), expected_rows=3)


def test_tie_must_break_by_candidate_id_ascending(tmp_path):
    # distinct top score, then equal scores with candidate_id descending -> invalid
    rows = [
        ["a0", 1, "0.900000", "r"],
        ["c2", 2, "0.500000", "r"],
        ["c1", 3, "0.500000", "r"],  # tie: "c2" before "c1" violates ascending
    ]
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="tie at rank"):
        vs.validate(str(p), expected_rows=3)


def test_tie_numeric_ascending_ok(tmp_path):
    rows = [
        ["0", 1, "0.900000", "r"],
        ["2", 2, "0.500000", "r"],
        ["10", 3, "0.500000", "r"],
    ]
    p = _write_csv(tmp_path / "s.csv", rows)
    assert vs.validate(str(p), expected_rows=3) == 3


def test_embedded_newline_rejected(tmp_path):
    rows = _good_rows(3)
    rows[0][3] = "line1\nline2"
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="newline"):
        vs.validate(str(p), expected_rows=3)


def test_empty_reasoning_rejected(tmp_path):
    rows = _good_rows(3)
    rows[0][3] = "   "
    p = _write_csv(tmp_path / "s.csv", rows)
    with pytest.raises(vs.ValidationError, match="empty reasoning"):
        vs.validate(str(p), expected_rows=3)


def test_missing_file(tmp_path):
    with pytest.raises(vs.ValidationError, match="not found"):
        vs.validate(str(tmp_path / "nope.csv"), expected_rows=3)
