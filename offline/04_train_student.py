"""OFFLINE Phase 4 — Student model + harness (WORKFLOW STEP 9–11)."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import anomaly as anom  # noqa: E402
from src import eval as ev  # noqa: E402
from src import features as ft  # noqa: E402
from src import model as student_model  # noqa: E402
from src import pool_score as ps  # noqa: E402
from src.parse import F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [04_train] %(message)s")
log = logging.getLogger(__name__)


def load_training_matrix(features_path: str, labels_path: str):
    feats = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)
    merged = feats.merge(labels, on=F.CANDIDATE_ID, how="inner")
    feature_names = list(ft.FEATURE_COLUMNS)
    X = merged[feature_names].to_numpy(dtype=float)
    y_score = merged["score_100"].to_numpy(dtype=float)
    y_tier = merged["tier"].to_numpy(dtype=int)
    return merged, X, y_score, y_tier, feature_names


def fit_anomaly(merged: pd.DataFrame, artifacts_dir: str) -> None:
    cols = anom.ANOMALY_FEATURES
    X = merged[cols].to_numpy(dtype=float)
    clf = anom.fit_isolation_forest(X)
    out = os.path.join(artifacts_dir, "anomaly")
    anom.save_model(clf, out, cols)
    log.info("saved IsolationForest -> %s", out)


def train_heads(X, y_score, y_tier, feature_names, artifacts_dir: str) -> dict:
    results = {}
    for head in (student_model.HEAD_POINTWISE, student_model.HEAD_LAMBDARANK):
        log.info("training head=%s", head)
        y = y_score if head == student_model.HEAD_POINTWISE else y_tier
        group = [len(y)] if head == student_model.HEAD_LAMBDARANK else None
        ens = student_model.train_ensemble(
            X, y, feature_names, head=head, group=group
        )
        sub = "model" if head == student_model.HEAD_POINTWISE else f"model_{head}"
        out_dir = os.path.join(artifacts_dir, sub)
        ens.save(out_dir)
        results[head] = out_dir
        log.info(
            "saved %s; top importances: %s",
            out_dir,
            list(ens.feature_importances().items())[:6],
        )
    return results


def run_harness(
    feats: pd.DataFrame,
    relevance: dict,
    artifacts_dir: str,
) -> dict:
    """Pooled judging: rules + both student heads + random floor."""
    rng = np.random.default_rng(99)
    all_ids = feats[F.CANDIDATE_ID].astype(str).tolist()
    random_floor = list(rng.choice(all_ids, size=50, replace=False))

    rules_scores = ps.score_column_matrix(feats, None)
    rules_top = ps.top_k_ids(feats, rules_scores, k=100)

    variant_rankings = {"rules_v0": rules_top}
    for head, sub in (
        (student_model.HEAD_POINTWISE, "model"),
        (student_model.HEAD_LAMBDARANK, "model_lambdarank"),
    ):
        ens = student_model.Ensemble.load(os.path.join(artifacts_dir, sub))
        sc = ps.score_column_matrix(feats, ens)
        variant_rankings[head] = ps.top_k_ids(feats, sc, k=100)

    pool = ev.build_pool(variant_rankings, random_floor=random_floor)
    table = ev.pooled_harness(variant_rankings, relevance)
    out = {
        "pool_size": len(pool),
        "metrics": table,
        "winner": max(table.items(), key=lambda kv: kv[1]["composite"])[0],
    }
    with open(os.path.join(artifacts_dir, "harness_results.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    log.info("harness winner=%s composite=%s", out["winner"], table[out["winner"]])
    return out


def select_winner(artifacts_dir: str, harness: dict) -> str:
    winner = harness["winner"]
    meta = {
        "selected_head": winner,
        "metrics": harness["metrics"][winner],
    }
    # copy winning model to artifacts/model/ if lambdarank won
    if winner == student_model.HEAD_LAMBDARANK:
        import shutil

        src = os.path.join(artifacts_dir, "model_lambdarank")
        dst = os.path.join(artifacts_dir, "model")
        if os.path.isdir(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        meta["selected_head"] = student_model.HEAD_LAMBDARANK
    with open(os.path.join(artifacts_dir, "model", "selection.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return winner


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--features", default=os.path.join("artifacts", "features.parquet"))
    ap.add_argument(
        "--labels", default=os.path.join("artifacts", "teacher_labels.parquet")
    )
    args = ap.parse_args()

    merged, X, y_score, y_tier, feature_names = load_training_matrix(
        args.features, args.labels
    )
    log.info("training matrix: %d rows x %d features", X.shape[0], X.shape[1])

    fit_anomaly(merged, args.artifacts)
    train_heads(X, y_score, y_tier, feature_names, args.artifacts)

    relevance = dict(
        zip(merged[F.CANDIDATE_ID].astype(str), merged["tier"].astype(int))
    )
    harness = run_harness(pd.read_parquet(args.features), relevance, args.artifacts)
    select_winner(args.artifacts, harness)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
