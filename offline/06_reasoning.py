"""Build ``artifacts/reasoning.json`` for ranked candidates (verifier-gated)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import parse  # noqa: E402
from src import reasoning as rz  # noqa: E402
from src.parse import F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [06_reason] %(message)s")
log = logging.getLogger(__name__)


def _ranking_from_csv(path: str) -> list[tuple[str, int]]:
    """Read (candidate_id, rank) pairs from a submission CSV, if provided."""
    pairs: list[tuple[str, int]] = []
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            pairs.append((str(row["candidate_id"]), int(row["rank"])))
    return pairs


def _load_records_for_ranking(
    candidates_path: str, order: list[tuple[str, int]]
) -> dict[str, dict[str, Any]]:
    """Stream JSONL and materialize only the candidate_ids in ``order``."""
    need = {cid for cid, _ in order}
    records: dict[str, dict[str, Any]] = {}
    for rec in parse.stream_candidates(candidates_path):
        cid = str(rec.get(F.CANDIDATE_ID))
        if cid in need:
            records[cid] = rec
            if len(records) >= len(need):
                break
    return records


def build_reasoning_cache(
    candidates_path: str,
    artifacts_dir: str,
    ranking_csv: str | None = None,
    client=None,
) -> str:
    """Generate + verify + cache reasoning for the final candidates."""
    if ranking_csv and os.path.exists(ranking_csv):
        order = _ranking_from_csv(ranking_csv)
    else:
        log.info("no ranking CSV — streaming first 100 ids from pool")
        order = []
        for i, rec in enumerate(parse.stream_candidates(candidates_path), start=1):
            order.append((str(rec.get(F.CANDIDATE_ID)), i))
            if i >= 100:
                break

    records = _load_records_for_ranking(candidates_path, order)
    now = parse.reference_now_from_path(candidates_path)

    cache: dict[Any, str] = {}
    for cid, rank in order:
        rec = records.get(cid)
        if rec is None:
            continue
        fs = rz.build_fact_sheet(rec, now=now)
        cache[cid] = rz.generate_and_verify(fs, rank, client=client)

    os.makedirs(artifacts_dir, exist_ok=True)
    out_path = os.path.join(artifacts_dir, "reasoning.json")
    rz.save_reasoning_cache(cache, out_path)
    log.info("wrote %d reasoning entries -> %s", len(cache), out_path)
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=os.path.join("data", "candidates.jsonl"))
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--ranking", default=None, help="submission CSV with id,rank order")
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        log.warning("candidates file not found: %s", args.candidates)
        return 0
    build_reasoning_cache(args.candidates, args.artifacts, args.ranking, client=None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
