"""Batch scoring over the feature store (offline harness / trap regression)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import features as ft
from .labeling_pool import rules_shortlist_score
from .model import Ensemble
from .parse import candidate_sort_key


def score_column_matrix(
    df,
    ensemble: Ensemble | None,
    *,
    id_col: str = "candidate_id",
) -> np.ndarray:
    if ensemble is None:
        scores = []
        for _, row in df.iterrows():
            scores.append(rules_shortlist_score(row.to_dict()))
        return np.asarray(scores, dtype=float)

    X = np.array(
        [
            [row.get(c, math.nan) for c in ensemble.feature_names]
            for _, row in df.iterrows()
        ],
        dtype=float,
    )
    return ensemble.predict(X)


def top_k_ids(
    df,
    scores: np.ndarray,
    k: int = 100,
    *,
    id_col: str = "candidate_id",
    exclude_ids: set[str] | None = None,
) -> list[str]:
    exclude_ids = exclude_ids or set()
    pairs = []
    for i, (_, row) in enumerate(df.iterrows()):
        cid = str(row[id_col])
        if cid in exclude_ids:
            continue
        pairs.append((cid, float(scores[i])))
    pairs.sort(key=lambda kv: (-kv[1], candidate_sort_key(kv[0])))
    return [cid for cid, _ in pairs[:k]]
