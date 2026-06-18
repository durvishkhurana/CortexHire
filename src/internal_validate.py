"""Extra validation rules used for sandbox runs (subset of organizer checks)."""

from __future__ import annotations

import argparse
import csv
import sys

HEADER = ["candidate_id", "rank", "score", "reasoning"]
DEFAULT_ROWS = 100
MAX_DECIMALS = 6


class ValidationError(Exception):
    """Raised when the submission CSV violates a hard rule."""


def _candidate_sort_key(candidate_id: str):
    """Tiebreak key for candidate_id ascending.

    The organizer compares ids as strings; CAND_ ids are fixed-width so string
    order == numeric order. We keep a numeric-first fallback for non-CAND ids.
    """
    try:
        return (0, int(candidate_id))
    except (TypeError, ValueError):
        return (1, str(candidate_id))


def _decimal_places(score_text: str) -> int:
    """Number of fractional digits in the literal score token."""
    token = score_text.strip()
    if "e" in token.lower():
        raise ValidationError(
            f"score {token!r} uses scientific notation; emit a plain decimal "
            "rounded to 6 dp"
        )
    if "." not in token:
        return 0
    return len(token.split(".", 1)[1])


def _read_rows(path: str) -> list[list[str]]:
    """Read the CSV as UTF-8, returning parsed rows (header + data)."""
    try:
        with open(path, encoding="utf-8", newline="") as fh:
            try:
                return list(csv.reader(fh))
            except UnicodeDecodeError as exc:
                raise ValidationError(f"file is not valid UTF-8: {exc}") from exc
    except FileNotFoundError as exc:
        raise ValidationError(f"file not found: {path}") from exc


def validate(
    path: str, expected_rows: int = DEFAULT_ROWS, allow_fewer: bool = False
) -> int:
    """Validate the submission CSV; return row count or raise ValidationError."""
    rows = _read_rows(path)
    if not rows:
        raise ValidationError("file is empty")

    header, data = rows[0], rows[1:]
    if header != HEADER:
        raise ValidationError(
            f"header must be exactly {','.join(HEADER)} (got {','.join(header)})"
        )

    n = len(data)
    if allow_fewer:
        if not (1 <= n <= expected_rows):
            raise ValidationError(f"expected 1..{expected_rows} data rows, got {n}")
    else:
        if n != expected_rows:
            raise ValidationError(
                f"expected exactly {expected_rows} data rows + header, got {n}"
            )

    candidate_ids: list[str] = []
    ranks: list[int] = []
    scores: list[float] = []

    for i, row in enumerate(data, start=1):
        if len(row) != len(HEADER):
            raise ValidationError(
                f"data row {i} has {len(row)} fields, expected {len(HEADER)}: {row!r}"
            )
        cid, rank_text, score_text, reasoning = row

        if not cid.strip():
            raise ValidationError(f"data row {i}: empty candidate_id")

        try:
            rank = int(rank_text)
        except ValueError as exc:
            raise ValidationError(
                f"data row {i}: rank {rank_text!r} is not an integer"
            ) from exc

        if _decimal_places(score_text) > MAX_DECIMALS:
            raise ValidationError(
                f"data row {i}: score {score_text!r} has more than "
                f"{MAX_DECIMALS} decimal places (round to 6 dp before writing)"
            )
        try:
            score = float(score_text)
        except ValueError as exc:
            raise ValidationError(
                f"data row {i}: score {score_text!r} is not a float"
            ) from exc

        if not reasoning.strip():
            raise ValidationError(f"data row {i}: empty reasoning")
        if "\n" in reasoning or "\r" in reasoning:
            raise ValidationError(
                f"data row {i}: reasoning contains an embedded newline"
            )

        candidate_ids.append(cid)
        ranks.append(rank)
        scores.append(score)

    if sorted(ranks) != list(range(1, n + 1)):
        raise ValidationError(
            f"ranks must be exactly 1..{n} each once (got {sorted(ranks)[:5]}...)"
        )
    if ranks != list(range(1, n + 1)):
        raise ValidationError("data rows must be ordered by rank ascending (1..n)")

    if len(set(candidate_ids)) != n:
        dupes = {c for c in candidate_ids if candidate_ids.count(c) > 1}
        raise ValidationError(f"candidate_id values must be unique; dupes: {dupes}")

    if len(set(scores)) == 1:
        raise ValidationError("all scores are identical (scores must vary)")

    for i in range(n - 1):
        if scores[i] < scores[i + 1] - 1e-12:
            raise ValidationError(
                f"score increases at rank {i + 2}: {scores[i]} -> {scores[i + 1]} "
                "(scores must be non-increasing with rank)"
            )
        if abs(scores[i] - scores[i + 1]) <= 1e-12:
            k_here = _candidate_sort_key(candidate_ids[i])
            k_next = _candidate_sort_key(candidate_ids[i + 1])
            if k_here > k_next:
                raise ValidationError(
                    f"tie at rank {i + 1}/{i + 2} (score {scores[i]}) not broken "
                    f"by candidate_id ascending: {candidate_ids[i]!r} before "
                    f"{candidate_ids[i + 1]!r}"
                )

    return n


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="path to the submission CSV")
    parser.add_argument("--expected-rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--allow-fewer", action="store_true")
    args = parser.parse_args(argv)

    try:
        n = validate(args.csv_path, args.expected_rows, args.allow_fewer)
    except ValidationError as exc:
        print(f"INVALID: {exc}", file=sys.stderr)
        return 1

    print(f"VALID: {args.csv_path} ({n} data rows + header)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
