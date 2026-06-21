"""OFFLINE Phase 3 — Labeling pool + pseudo-teacher (WORKFLOW STEPS 7–9).

Uses a deterministic offline pseudo-teacher (no paid API). Measures
primary-vs-strict self-consistency on a pilot subset, then labels the full pool.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import labeling_pool as lp  # noqa: E402
from src import pseudo_teacher as pt  # noqa: E402
from src.parse import F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [03_teacher] %(message)s")
log = logging.getLogger(__name__)

PILOT_N = 400
CONSISTENCY_SEED = 17


def run_step7(features_path: str, artifacts_dir: str) -> pd.DataFrame:
    log.info("STEP 7: assembling labeling pool from %s", features_path)
    feats = lp.load_features_parquet(features_path)
    pool = lp.assemble_labeling_pool_df(feats)
    os.makedirs(artifacts_dir, exist_ok=True)
    out = os.path.join(artifacts_dir, "labeling_pool.parquet")
    pool.to_parquet(out, index=False)
    log.info("wrote %d unique ids -> %s", len(pool), out)
    return pool


def run_pilot_consistency(
    feats: pd.DataFrame, pool_ids: list[str]
) -> dict[str, float]:
    log.info("STEP 8: pilot consistency on %d shuffled ids", PILOT_N)
    rng = np.random.default_rng(CONSISTENCY_SEED)
    pilot = list(rng.choice(pool_ids, size=min(PILOT_N, len(pool_ids)), replace=False))
    feat_idx = feats.set_index(F.CANDIDATE_ID)
    tier_agree = 0
    score_mae = []
    for cid in pilot:
        row = feat_idx.loc[cid].to_dict()
        a, b, ok = pt.label_consistency(row)
        if ok:
            tier_agree += 1
        score_mae.append(abs(a["score_100"] - b["score_100"]))
    stats = {
        "pilot_n": len(pilot),
        "pairwise_consistent_rate": tier_agree / len(pilot),
        "mean_abs_score_delta": float(np.mean(score_mae)),
        "max_abs_score_delta": float(np.max(score_mae)),
    }
    log.info("pilot consistency: %s", stats)
    return stats


def run_full_labeling(
    feats: pd.DataFrame,
    pool_ids: list[str],
    artifacts_dir: str,
    *,
    drop_inconsistent: bool = True,
) -> pd.DataFrame:
    log.info("STEP 9: full pseudo-teacher labeling (%d ids)", len(pool_ids))
    feat_idx = feats.set_index(F.CANDIDATE_ID)
    rows = []
    dropped = 0
    for cid in pool_ids:
        row = feat_idx.loc[cid].to_dict()
        a, b, ok = pt.label_consistency(row)
        if drop_inconsistent and not ok:
            dropped += 1
            continue
        lab = pt.score_candidate(row, mode="primary")
        rows.append(
            {
                F.CANDIDATE_ID: cid,
                "tier": lab["tier"],
                "score_100": lab["score_100"],
                "evidence_quote": lab["evidence_quote"],
                "label_consistent": float(ok),
            }
        )
    labels = pd.DataFrame(rows)
    out = os.path.join(artifacts_dir, "teacher_labels.parquet")
    labels.to_parquet(out, index=False)
    log.info(
        "wrote %d labels (%d dropped low-consistency) -> %s",
        len(labels),
        dropped,
        out,
    )
    pilot_path = os.path.join(artifacts_dir, "teacher_pilot_stats.json")
    return labels


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument(
        "--features", default=os.path.join("artifacts", "features.parquet")
    )
    ap.add_argument("--step", default="all", choices=["7", "8", "9", "all"])
    ap.add_argument(
        "--keep-inconsistent",
        action="store_true",
        help="label all pool ids (do not drop primary/strict disagreements)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.features):
        log.error("missing %s", args.features)
        return 1

    feats = lp.load_features_parquet(args.features)
    pool_df = None
    if args.step in ("7", "all"):
        pool_df = run_step7(args.features, args.artifacts)
    else:
        pool_path = os.path.join(args.artifacts, "labeling_pool.parquet")
        pool_df = pd.read_parquet(pool_path)

    pool_ids = pool_df[F.CANDIDATE_ID].astype(str).tolist()

    stats = None
    if args.step in ("8", "all"):
        stats = run_pilot_consistency(feats, pool_ids)
        with open(
            os.path.join(args.artifacts, "teacher_pilot_stats.json"),
            "w",
            encoding="utf-8",
        ) as fh:
            json.dump(stats, fh, indent=2)

    if args.step in ("9", "all"):
        run_full_labeling(
            feats,
            pool_ids,
            args.artifacts,
            drop_inconsistent=not args.keep_inconsistent,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
