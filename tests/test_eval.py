"""Known-answer tests for ranking metrics and the pooled-judging harness."""

from __future__ import annotations

import math

from src import eval as ev


def test_ndcg_perfect_is_one():
    rels = [5, 4, 3, 2, 1]
    assert math.isclose(ev.ndcg_at_k(rels, 5), 1.0)


def test_ndcg_reversed_less_than_one():
    assert ev.ndcg_at_k([1, 2, 3, 4, 5], 5) < 1.0


def test_dcg_known_value():
    # rels [3, 2]: gains 7, 3 ; discounts 1/log2(2)=1, 1/log2(3)
    expected = 7 * 1.0 + 3 * (1.0 / math.log2(3))
    assert math.isclose(ev.dcg_at_k([3, 2], 2), expected)


def test_ndcg_zero_when_no_relevance():
    assert ev.ndcg_at_k([0, 0, 0], 3) == 0.0


def test_precision_at_k():
    assert ev.precision_at_k([1, 0, 1, 0], 4) == 0.5
    assert ev.precision_at_k([1, 1, 0, 0], 2) == 1.0


def test_average_precision_known():
    # relevant at positions 1 and 3: (1/1 + 2/3) / 2
    expected = (1.0 + 2.0 / 3.0) / 2
    assert math.isclose(ev.average_precision([1, 0, 1, 0], 2), expected)


def test_average_precision_penalized_for_missing_relevant():
    # one relevant retrieved at pos 1, but 4 relevant in pool -> AP = 1/4
    assert math.isclose(ev.average_precision([1, 0, 0], n_relevant=4), 0.25)


def test_map_averages_queries():
    q1 = [1, 0]  # AP = 1.0
    q2 = [0, 1]  # AP = 0.5
    assert math.isclose(ev.mean_average_precision([q1, q2]), 0.75)


def test_composite_weights():
    c = ev.composite_score(1.0, 1.0, 1.0, 1.0)
    assert math.isclose(c, 1.0)
    c2 = ev.composite_score(1.0, 0.0, 0.0, 0.0)
    assert math.isclose(c2, 0.5)


def test_evaluate_variant():
    relevance = {"a": 5, "b": 4, "c": 0, "d": 3}
    ranked = ["a", "b", "d", "c"]  # perfect-ish
    m = ev.evaluate_variant(ranked, relevance)
    assert m["ndcg@10"] > 0.9
    assert m["p@10"] > 0  # a, b, d are tier >=3 in top 10
    assert 0.0 <= m["composite"] <= 1.0


def test_build_pool_dedup():
    pool = ev.build_pool({"v1": ["a", "b"], "v2": ["b", "c"]}, random_floor=["c", "d"])
    assert pool == ["a", "b", "c", "d"]


def test_pooled_harness_ranks_better_variant_higher():
    relevance = {"a": 5, "b": 4, "c": 0, "d": 3, "e": 0}
    good = ["a", "b", "d", "c", "e"]
    bad = ["c", "e", "a", "b", "d"]
    table = ev.pooled_harness({"good": good, "bad": bad}, relevance)
    assert table["good"]["composite"] > table["bad"]["composite"]
