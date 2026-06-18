"""Throwaway diagnostic: per-check honeypot firing rates over the real pool."""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import honeypot as hp  # noqa: E402
from src import lexicon, parse  # noqa: E402

candidates = sys.argv[1] if len(sys.argv) > 1 else os.path.join("data", "candidates.jsonl")
now = parse.reference_now_from_path(candidates)
founding = lexicon.load_founding_years(os.path.join("artifacts", "founding_years.csv"))

per_check_hard = Counter()
per_check_soft = Counter()
n = 0
n_hard = 0
hard_examples = {}
for rec in parse.stream_candidates(candidates):
    n += 1
    a = hp.run_consistency_suite(rec, founding_years=founding, now=now)
    hv = False
    for r in a.results:
        if r.failed and r.severity == hp.HARD:
            per_check_hard[r.name] += 1
            hv = True
            if r.name not in hard_examples:
                hard_examples[r.name] = (rec.get("candidate_id"), r.detail)
        elif r.failed and r.severity == hp.SOFT:
            per_check_soft[r.name] += 1
    if hv:
        n_hard += 1

print(f"n={n} candidates_with_hard_violation={n_hard} ({100.0*n_hard/n:.2f}%)")
print("--- HARD checks (count, %) ---")
for name, c in per_check_hard.most_common():
    ex = hard_examples.get(name)
    print(f"  {name}: {c} ({100.0*c/n:.3f}%)  e.g. {ex}")
print("--- SOFT checks (count, %) ---")
for name, c in per_check_soft.most_common():
    print(f"  {name}: {c} ({100.0*c/n:.3f}%)")
