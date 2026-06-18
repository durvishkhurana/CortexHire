"""Build ``artifacts/features.parquet`` from ``candidates.jsonl``.

Two streaming passes: reference date = max(last_active_date), then batched
featurization. Text-score columns are filled by ``offline/02_text_scores.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import features as ft  # noqa: E402
from src import lexicon, parse  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [01_features] %(message)s")
log = logging.getLogger(__name__)


def build(candidates_path: str, artifacts_dir: str) -> str:
    """Build features.parquet from the candidate JSONL. Returns the output path."""
    if not os.path.exists(candidates_path):
        raise FileNotFoundError(
            f"candidates file not found: {candidates_path} (expected under data/)"
        )
    os.makedirs(artifacts_dir, exist_ok=True)

    log.info("pass 1/2: streaming for reference now: %s", candidates_path)
    now = parse.reference_now_from_path(candidates_path)
    founding = lexicon.load_founding_years(
        os.path.join(artifacts_dir, "founding_years.csv")
    )

    out_path = os.path.join(artifacts_dir, "features.parquet")
    log.info("pass 2/2: featurizing + writing parquet (reference now=%s)", now)
    n = ft.write_features_parquet(
        parse.stream_candidates(candidates_path),
        out_path,
        now=now,
        founding_years=founding,
    )
    log.info(
        "wrote %s rows x %s cols -> %s", n, 1 + len(ft.FEATURE_COLUMNS), out_path
    )
    return out_path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=os.path.join("data", "candidates.jsonl"))
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    try:
        build(args.candidates, args.artifacts)
    except FileNotFoundError as exc:
        log.warning("%s", exc)
        return 0  # not a failure — just blocked on the human-provided data
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
