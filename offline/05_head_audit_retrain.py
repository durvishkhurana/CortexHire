"""OFFLINE Phase 5 — Head audit -> retrain (WORKFLOW STEP 12).

Pseudo-audit: pairwise compare student top-300 order vs teacher tiers on
near-ties; emit adjusted labels and retrain (no override table).
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

from src import model as student_model  # noqa: E402
from src import pool_score as ps  # noqa: E402
from src.parse import F, candidate_sort_key  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [05_audit] %(message)s")
log = logging.getLogger(__name__)


def pseudo_pairwise_audit(
    ranked_ids: list[str],
    teacher: dict[str, int],
    *,
    passes: int = 2,
) -> list[tuple[str, str, int]]:
    """Return (higher_id, lower_id, preferred_tier_delta) disagreements."""
    pairs = []
    for _ in range(passes):
        for i in range(min(300, len(ranked_ids)) - 1):
            a, b = ranked_ids[i], ranked_ids[i + 1]
            ta, tb = teacher.get(a, 0), teacher.get(b, 0)
            if ta + 1 < tb:
                pairs.append((b, a, int(tb - ta)))
    return pairs


def build_audit_flags(
    feats: pd.DataFrame,
    ranked_ids: list[str],
    teacher: dict[str, int],
    *,
    top_n: int = 500,
) -> set[str]:
    """Offline contradiction hunt → candidate_ids for rank.py triple-guard audit leg."""
    flags: set[str] = set()
    idx = feats.set_index(F.CANDIDATE_ID)
    for cid in ranked_ids[:top_n]:
        if cid not in idx.index:
            continue
        row = idx.loc[cid]
        if float(row.get("n_hard_flags", 0) or 0) >= 1.0:
            flags.add(str(cid))
            continue
        tier = teacher.get(str(cid), -1)
        if tier == 0:
            flags.add(str(cid))
            continue
        if float(row.get("claimed_unverified_ratio", 0) or 0) > 1.2:
            flags.add(str(cid))
        if float(row.get("n_soft_flags", 0) or 0) >= 3.0:
            flags.add(str(cid))
    # Student order vs teacher: demote head when a lower-ranked id has much higher tier.
    for i in range(min(top_n, len(ranked_ids)) - 1):
        a, b = ranked_ids[i], ranked_ids[i + 1]
        ta, tb = teacher.get(a, 0), teacher.get(b, 0)
        if tb >= ta + 2:
            flags.add(str(a))
    return flags


def write_audit_flags(artifacts_dir: str, flags: set[str]) -> str:
    path = os.path.join(artifacts_dir, "audit_flags.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(sorted(flags), fh, indent=2)
    log.info("wrote %d audit flags -> %s", len(flags), path)
    return path


def augment_labels(
    labels: pd.DataFrame, pairs: list[tuple[str, str, int]]
) -> pd.DataFrame:
    out = labels.copy()
    idx = out.set_index(F.CANDIDATE_ID)
    for hi, lo, delta in pairs:
        if hi in idx.index:
            row = idx.loc[hi]
            bump = min(5, int(row["tier"]) + min(1, delta))
            idx.at[hi, "tier"] = bump
            idx.at[hi, "score_100"] = min(100.0, float(row["score_100"]) + 3.0)
        if lo in idx.index:
            row = idx.loc[lo]
            idx.at[lo, "tier"] = max(0, int(row["tier"]) - 1)
            idx.at[lo, "score_100"] = max(0.0, float(row["score_100"]) - 2.0)
    return idx.reset_index()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--features", default=os.path.join("artifacts", "features.parquet"))
    ap.add_argument(
        "--labels", default=os.path.join("artifacts", "teacher_labels.parquet")
    )
    ap.add_argument("--passes", type=int, default=2)
    args = ap.parse_args()

    feats = pd.read_parquet(args.features)
    labels = pd.read_parquet(args.labels)
    model_dir = os.path.join(args.artifacts, "model")
    ens = student_model.Ensemble.load(model_dir)
    scores = ps.score_column_matrix(feats, ens)
    top300 = ps.top_k_ids(feats, scores, k=300)

    teacher = dict(zip(labels[F.CANDIDATE_ID].astype(str), labels["tier"].astype(int)))
    pairs = pseudo_pairwise_audit(top300, teacher, passes=args.passes)
    log.info("audit pass: %d pairwise adjustments from %d passes", len(pairs), args.passes)

    audit_ids = build_audit_flags(feats, top300, teacher)
    write_audit_flags(args.artifacts, audit_ids)

    gold = [{"preferred": hi, "demoted": lo, "delta": d} for hi, lo, d in pairs[:50]]
    with open(os.path.join(args.artifacts, "audit_disagreements.json"), "w") as fh:
        json.dump(gold, fh, indent=2)

    aug = augment_labels(labels, pairs)
    aug_path = os.path.join(args.artifacts, "teacher_labels_audit.parquet")
    aug.to_parquet(aug_path, index=False)

    # Retrain pointwise head on augmented labels (fold audit into training).
    import importlib.util

    train_path = os.path.join(os.path.dirname(__file__), "04_train_student.py")
    spec = importlib.util.spec_from_file_location("train_mod", train_path)
    train_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_mod)

    merged, X, y_score, y_tier, feature_names = train_mod.load_training_matrix(
        args.features, aug_path
    )
    train_mod.train_heads(X, y_score, y_tier, feature_names, args.artifacts)
    log.info("retrained student after audit augmentation")

    # Restore harness-selected head in artifacts/model/ when it is not already there.
    import shutil

    sel_path = os.path.join(args.artifacts, "model", "selection.json")
    if os.path.isfile(sel_path):
        with open(sel_path, encoding="utf-8") as fh:
            sel = json.load(fh)
        if sel.get("selected_head") == student_model.HEAD_LAMBDARANK:
            src = os.path.join(args.artifacts, "model_lambdarank")
            dst = os.path.join(args.artifacts, "model")
            if os.path.isdir(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            with open(os.path.join(dst, "selection.json"), "w", encoding="utf-8") as fh:
                json.dump(sel, fh, indent=2)
            log.info("restored winning lambdarank ensemble -> artifacts/model")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
