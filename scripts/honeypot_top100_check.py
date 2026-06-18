"""Report hard-violation honeypots in a submission CSV top-100 (+ margin 10)."""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import honeypot as hp  # noqa: E402
from src import lexicon, parse  # noqa: E402
from src.parse import F  # noqa: E402


def main() -> int:
    sub = sys.argv[1] if len(sys.argv) > 1 else "artifacts/submission_final.csv"
    pool = sys.argv[2] if len(sys.argv) > 2 else "data/candidates.jsonl"
    top_n = int(sys.argv[3]) if len(sys.argv) > 3 else 110

    ranks: list[tuple[int, str]] = []
    with open(sub, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            ranks.append((int(row["rank"]), str(row["candidate_id"])))
    ranks.sort()
    watch = {cid for _, cid in ranks if _ <= top_n}

    now = parse.reference_now_from_path(pool)
    founding = lexicon.load_founding_years("artifacts/founding_years.csv")
    hits = []
    for rec in parse.stream_candidates(pool):
        cid = str(rec.get(F.CANDIDATE_ID))
        if cid not in watch:
            continue
        a = hp.run_consistency_suite(rec, founding_years=founding, now=now)
        if a.hard_violation:
            hits.append((cid, [r for r in a.failed_checks]))
    print(f"checked top {top_n} ranks; hard honeypots in set: {len(hits)}")
    for cid, reasons in hits[:20]:
        print(cid, reasons)
    return 1 if hits else 0


if __name__ == "__main__":
    raise SystemExit(main())
