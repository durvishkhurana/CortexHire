"""LightGBM student: two heads, monotone constraints, seeded ensemble.

ARCHITECTURE §3.4: compare a ``lambdarank`` head (one query group) against a
pointwise regression head on the teacher's 0-100 score, top-weighted; select on
the pooled harness (``src/eval.py``), not on fold-NDCG. Add monotone constraints
on signals whose direction we know, and average a 3-5 seed ensemble to cut the
one-shot NDCG@10 variance.

Determinism (rule #2): every booster is trained with a fixed ``seed`` and
``deterministic=True``; the ensemble averages raw predictions.

This module is import-safe for ``rank.py`` (no GPU, no network). Training is
invoked from ``offline/04_train_student.py``; ``rank.py`` only calls
:meth:`Ensemble.load` + :meth:`Ensemble.predict`.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .features import MONOTONE_CONSTRAINTS

HEAD_LAMBDARANK = "lambdarank"
HEAD_POINTWISE = "pointwise"
DEFAULT_SEEDS = (13, 29, 41, 57, 73)
# LightGBM lambdarank: max documents per query group (library limit is 10_000).
LAMBDARANK_MAX_GROUP = 5000


def lambdarank_group_sizes(n: int, *, max_per_group: int = LAMBDARANK_MAX_GROUP) -> list[int]:
    """Split n training rows into query groups for lambdarank."""
    if n <= 0:
        return []
    if n <= max_per_group:
        return [n]
    sizes: list[int] = []
    rem = n
    while rem > 0:
        g = min(max_per_group, rem)
        sizes.append(g)
        rem -= g
    return sizes


def build_monotone_vector(
    feature_names: Sequence[str],
    constraints: dict[str, int] | None = None,
) -> list[int]:
    """Map the feature order to a LightGBM ``monotone_constraints`` list.

    +1 = prediction non-decreasing in the feature, -1 = non-increasing, 0 free.
    """
    constraints = constraints if constraints is not None else MONOTONE_CONSTRAINTS
    return [int(constraints.get(name, 0)) for name in feature_names]


def default_params(
    head: str,
    seed: int,
    monotone: Sequence[int],
) -> dict[str, object]:
    """Shallow, deterministic LightGBM params for a single booster."""
    common = {
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "num_leaves": 31,
        "max_depth": 6,
        "min_data_in_leaf": 5,
        "learning_rate": 0.05,
        "monotone_constraints": list(monotone),
    }
    if head == HEAD_LAMBDARANK:
        common.update(
            objective="lambdarank",
            metric="ndcg",
            ndcg_eval_at=[10, 50],
            label_gain=[2**i - 1 for i in range(0, 32)],
        )
    elif head == HEAD_POINTWISE:
        common.update(objective="regression", metric="l2")
    else:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown head {head!r}")
    return common


@dataclass
class Ensemble:
    """A seed-averaged set of LightGBM boosters with their feature schema."""

    boosters: list
    feature_names: list[str]
    head: str
    monotone: list[int]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Average raw predictions across the seed ensemble."""
        X = np.asarray(X, dtype=float)
        preds = [b.predict(X) for b in self.boosters]
        return np.mean(preds, axis=0)

    def feature_importances(self, importance_type: str = "gain") -> dict[str, float]:
        """Mean per-feature importance across the ensemble (interview slide)."""
        mats = [
            b.feature_importance(importance_type=importance_type) for b in self.boosters
        ]
        mean = np.mean(mats, axis=0)
        pairs = sorted(
            zip(self.feature_names, mean), key=lambda kv: kv[1], reverse=True
        )
        return {name: float(val) for name, val in pairs}

    def save(self, directory: str) -> None:
        """Persist boosters + metadata to ``directory`` (created if needed)."""
        os.makedirs(directory, exist_ok=True)
        for i, b in enumerate(self.boosters):
            b.save_model(os.path.join(directory, f"booster_{i}.txt"))
        meta = {
            "n_boosters": len(self.boosters),
            "feature_names": self.feature_names,
            "head": self.head,
            "monotone": self.monotone,
        }
        with open(os.path.join(directory, "meta.json"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, directory: str) -> Ensemble:
        """Load an ensemble previously written by :meth:`save`."""
        import lightgbm as lgb

        with open(os.path.join(directory, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        boosters = [
            lgb.Booster(model_file=os.path.join(directory, f"booster_{i}.txt"))
            for i in range(meta["n_boosters"])
        ]
        return cls(boosters, meta["feature_names"], meta["head"], meta["monotone"])


def _train_one(
    head: str,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    seed: int,
    monotone: Sequence[int],
    num_boost_round: int,
    group: Sequence[int] | None,
    sample_weight: np.ndarray | None,
):
    import lightgbm as lgb

    params = default_params(head, seed, monotone)
    dataset = lgb.Dataset(
        np.asarray(X, dtype=float),
        label=np.asarray(y, dtype=float),
        feature_name=list(feature_names),
        weight=sample_weight,
        group=group,
        free_raw_data=False,
    )
    return lgb.train(params, dataset, num_boost_round=num_boost_round)


def train_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    *,
    head: str = HEAD_POINTWISE,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    num_boost_round: int = 300,
    group: Sequence[int] | None = None,
    constraints: dict[str, int] | None = None,
    top_weighted: bool = True,
) -> Ensemble:
    """Train a seed-ensemble of LightGBM boosters.

    Args:
        X: feature matrix (n, d). NaNs are fine — LightGBM routes them.
        y: targets. For ``pointwise`` the teacher's 0-100 score; for
            ``lambdarank`` integer relevance gains (0-5).
        feature_names: column order (drives monotone constraints).
        head: ``pointwise`` or ``lambdarank``.
        seeds: 3-5 seeds for the ensemble.
        group: required for ``lambdarank`` (one group = ``[len(y)]``).
        top_weighted: pointwise only — weight samples by relevance gain so the
            top of the ranking (which dominates NDCG@10) is fit hardest.
    """
    feature_names = list(feature_names)
    monotone = build_monotone_vector(feature_names, constraints)
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    sample_weight = None
    if head == HEAD_LAMBDARANK:
        if group is None or (
            len(group) == 1 and group[0] > LAMBDARANK_MAX_GROUP
        ):
            group = lambdarank_group_sizes(len(y))
    elif head == HEAD_POINTWISE and top_weighted:
        # weight proportional to (normalized) relevance gain, floored at 1.
        gain = y - y.min()
        denom = gain.max() if gain.max() > 0 else 1.0
        sample_weight = 1.0 + 4.0 * (gain / denom)

    boosters = [
        _train_one(
            head,
            X,
            y,
            feature_names,
            s,
            monotone,
            num_boost_round,
            group,
            sample_weight,
        )
        for s in seeds
    ]
    return Ensemble(boosters, feature_names, head, list(monotone))
