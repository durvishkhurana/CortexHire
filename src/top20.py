"""Deterministic uncertainty penalty inside the top-20 band (STEP 13)."""

from __future__ import annotations

import math
from typing import Any

from .parse import candidate_sort_key


def uncertainty_penalty(feature_row: dict[str, Any]) -> float:
    """Higher = more uncertain / riskier; subtracted from score in top-20 only."""
    def _f(k: str, d: float = 0.0) -> float:
        v = feature_row.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            return d
        return float(v)

    p = 0.0
    p += 0.15 * _f("claimed_unverified_ratio")
    p += 0.10 * _f("n_soft_flags")
    p += 0.08 * _f("title_chasing_flag")
    p += 0.08 * _f("growth_gap_flag")
    p += 0.07 * _f("production_ownership_gap_flag")
    p += 0.05 * _f("junior_title_flag")
    p += 0.06 * (1.0 - min(1.0, _f("yoe_fit", 0.5)))
    p += 0.05 * _f("no_recent_ic_flag")
    if _f("has_assessments") < 0.5:
        p += 0.04
    return min(0.25, p)


def apply_top20_penalty(
    ranked: list[tuple[Any, float]],
    feature_by_id: dict[str, dict[str, Any]],
) -> list[tuple[Any, float]]:
    """Re-order only ranks 1–20 by score minus penalty; keep 21+ fixed."""
    if len(ranked) <= 20:
        band = list(ranked)
        rest = []
    else:
        band = ranked[:20]
        rest = ranked[20:]

    adjusted = []
    for cid, sc in band:
        row = feature_by_id.get(str(cid), {})
        sc2 = sc - uncertainty_penalty(row)
        adjusted.append((cid, round(sc2, 6)))

    adjusted.sort(key=lambda kv: (-kv[1], candidate_sort_key(kv[0])))
    return adjusted + rest
