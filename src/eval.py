"""Ranking metrics + TREC-style pooled-judging harness.

The competition score is::

    Final = 0.50*NDCG@10 + 0.30*NDCG@50 + 0.15*MAP + 0.05*P@10

so :func:`composite_score` mirrors it exactly. Model selection happens on the
**pooled-judging harness** (:func:`pooled_harness`), never on a "spectrum"
sample: we label the union of every variant's top-k (+ a random floor) and
score all variants on that same pool (ARCHITECTURE §8 / PANEL_REVIEW FLAW 1).

Relevances are graded 0-5 (honeypots=0, P@10 boundary tier 3+). Metrics take
either graded relevances (NDCG) or a relevance threshold to binarize (MAP/P@k).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

DEFAULT_REL_THRESHOLD = 3  # P@10 counts tier 3+


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with gain = 2^rel - 1, log2 position discount."""
    rels = np.asarray(relevances[:k], dtype=float)
    if rels.size == 0:
        return 0.0
    gains = np.power(2.0, rels) - 1.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(gains * discounts))


def ndcg_at_k(ranked_relevances: Sequence[float], k: int) -> float:
    """NDCG@k for items given in ranked order (their graded relevances).

    Ideal DCG is over the best achievable ordering of the *same* relevances.
    Returns 0.0 when there is no attainable gain."""
    actual = dcg_at_k(ranked_relevances, k)
    ideal = dcg_at_k(sorted(ranked_relevances, reverse=True), k)
    if ideal == 0.0:
        return 0.0
    return actual / ideal


def precision_at_k(binary_relevances: Sequence[int], k: int) -> float:
    """Fraction of the top-k that are relevant."""
    if k <= 0:
        return 0.0
    topk = binary_relevances[:k]
    return float(np.sum(topk)) / float(k)


def average_precision(
    binary_relevances: Sequence[int], n_relevant: int | None = None
) -> float:
    """Average precision for one ranked list.

    ``n_relevant`` is the total number of relevant items in the pool (so AP is
    penalized for relevant items missing from the list). Defaults to the number
    present in the list."""
    binary = np.asarray(binary_relevances, dtype=int)
    total_rel = int(np.sum(binary)) if n_relevant is None else int(n_relevant)
    if total_rel == 0:
        return 0.0
    hits = 0
    score = 0.0
    for i, rel in enumerate(binary, start=1):
        if rel:
            hits += 1
            score += hits / i
    return score / total_rel


def mean_average_precision(
    list_of_binary: Iterable[Sequence[int]],
    n_relevant_per_query: Sequence[int] | None = None,
) -> float:
    """MAP across queries (here typically one query = the fixed JD)."""
    lists = list(list_of_binary)
    if not lists:
        return 0.0
    if n_relevant_per_query is None:
        aps = [average_precision(b) for b in lists]
    else:
        aps = [average_precision(b, n) for b, n in zip(lists, n_relevant_per_query)]
    return float(np.mean(aps))


def composite_score(ndcg10: float, ndcg50: float, map_: float, p10: float) -> float:
    """The competition's weighted composite."""
    return 0.50 * ndcg10 + 0.30 * ndcg50 + 0.15 * map_ + 0.05 * p10


def evaluate_variant(
    ranked_ids: Sequence,
    relevance: dict,
    *,
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
    n_relevant: int | None = None,
) -> dict[str, float]:
    """Compute NDCG@10/@50, MAP-proxy, P@10, and the composite for one variant.

    Args:
        ranked_ids: candidate_ids in the variant's ranked order (best first).
        relevance: ``{candidate_id: graded_relevance}`` from the pooled labels;
            ids absent from the pool are treated as relevance 0.
        rel_threshold: tier counted as relevant for MAP/P@k (default 3).
        n_relevant: total relevant in the pool (for MAP denominator); defaults
            to the count of pool labels >= threshold."""
    rels = [float(relevance.get(cid, 0)) for cid in ranked_ids]
    binary = [1 if relevance.get(cid, 0) >= rel_threshold else 0 for cid in ranked_ids]
    if n_relevant is None:
        n_relevant = sum(1 for v in relevance.values() if v >= rel_threshold)

    ndcg10 = ndcg_at_k(rels, 10)
    ndcg50 = ndcg_at_k(rels, 50)
    p10 = precision_at_k(binary, 10)
    ap = average_precision(binary, n_relevant)
    return {
        "ndcg@10": ndcg10,
        "ndcg@50": ndcg50,
        "map": ap,
        "p@10": p10,
        "composite": composite_score(ndcg10, ndcg50, ap, p10),
    }


def build_pool(
    variant_topk: dict[str, Sequence],
    random_floor: Sequence | None = None,
) -> list:
    """Union of every variant's top-k plus an optional random floor (deduped).

    This is the set a human/teacher labels for the pooled harness; it has real
    overlap with each system's top-100 (unlike a spectrum sample)."""
    pool: list = []
    seen = set()
    for ids in variant_topk.values():
        for cid in ids:
            if cid not in seen:
                seen.add(cid)
                pool.append(cid)
    for cid in random_floor or []:
        if cid not in seen:
            seen.add(cid)
            pool.append(cid)
    return pool


def pooled_harness(
    variant_rankings: dict[str, Sequence],
    relevance: dict,
    *,
    rel_threshold: int = DEFAULT_REL_THRESHOLD,
) -> dict[str, dict[str, float]]:
    """Evaluate every variant on the same pooled labels.

    Args:
        variant_rankings: ``{variant_name: ranked_candidate_ids}``.
        relevance: pooled ``{candidate_id: graded_relevance}``.

    Returns ``{variant_name: metrics_dict}`` — the model-selection table."""
    n_relevant = sum(1 for v in relevance.values() if v >= rel_threshold)
    return {
        name: evaluate_variant(
            ranked, relevance, rel_threshold=rel_threshold, n_relevant=n_relevant
        )
        for name, ranked in variant_rankings.items()
    }
