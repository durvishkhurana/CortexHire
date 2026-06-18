"""Offline deterministic pseudo-teacher (no paid API).

Implements the reconstructed 0–5 rubric as a transparent feature-based scorer
aligned with ``docs/teacher_rubric.md``. Used when no frontier LLM is
available; double-pass consistency is measured with ``strict`` vs ``primary``
modes (same rubric, different tier boundaries).
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .parse import F


def _f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    v = row.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return default
    return float(v)


def score_candidate(
    row: dict[str, Any],
    *,
    mode: str = "primary",
) -> dict[str, Any]:
    """Return tier (0–5), score_100, evidence_quote for one feature row."""
    hard = _f(row, "n_hard_flags")
    if hard >= 1.0:
        return {
            "tier": 0,
            "score_100": 0.0,
            "evidence_quote": "hard consistency violation (honeypot signal)",
            "mode": mode,
        }

    # Positive career-evidence stack (JD: evidence over keywords).
    evidence = _f(row, "evidence_density")
    product = _f(row, "is_product_current")
    prod_mo = _f(row, "product_tenure_months")
    yoe_fit = _f(row, "yoe_fit", 0.5)
    fusion = _f(row, "fusion_score")
    rerank = _f(row, "reranker_score")
    assess_cov = _f(row, "assessment_coverage")
    loc = _f(row, "location_fit")
    rr = _f(row, "recruiter_response_rate", 0.3)
    icr = _f(row, "interview_completion_rate", 0.3)
    recency = _f(row, "last_active_recency")

    # Normalize offline scores heuristically (pool percentiles approximated).
    sem = 0.0
    if not math.isnan(fusion):
        sem += 0.35 * min(1.0, max(0.0, fusion))
    if not math.isnan(rerank):
        sem += 0.35 * min(1.0, max(0.0, (rerank + 5) / 10.0))
    sem += 0.15 * min(1.0, evidence / 3.0)
    sem += 0.15 * min(1.0, assess_cov)

    career = (
        0.25 * product
        + 0.20 * min(1.0, prod_mo / 48.0)
        + 0.25 * yoe_fit
        + 0.15 * min(1.0, _f(row, "jd_skill_coverage") / 5.0)
        + 0.15 * min(1.0, evidence / 2.0)
    )

    behavioral = (
        0.4 * min(1.0, rr)
        + 0.35 * min(1.0, icr)
        + 0.25 * min(1.0, max(0.0, (recency + 12) / 12.0))
    )

    penalty = (
        1.2 * _f(row, "claimed_unverified_ratio")
        + 1.0 * _f(row, "cv_speech_without_ir_flag")
        + 1.0 * _f(row, "services_only_flag")
        + 1.0 * _f(row, "no_recent_ic_flag")
        + 0.8 * _f(row, "title_chasing_flag")
        + 0.5 * _f(row, "yoe_junior_flag")
        + 0.3 * _f(row, "n_soft_flags")
    )

    raw = 0.45 * sem + 0.40 * career + 0.15 * behavioral - penalty
    raw += 0.08 * loc

    if mode == "strict":
        raw -= 0.12
        tier_shift = 0.5
    else:
        tier_shift = 0.0

    score_100 = float(np.clip((raw + 0.15) * 100, 0, 100))

    # Tier boundaries (0–5); strict mode uses higher bar for top tiers.
    t = score_100 / 100.0 - tier_shift
    if t < 0.12:
        tier = 0
    elif t < 0.28:
        tier = 1
    elif t < 0.45:
        tier = 2
    elif t < 0.62:
        tier = 3
    elif t < 0.78:
        tier = 4
    else:
        tier = 5

    quote_parts = []
    if product > 0:
        quote_parts.append("product-company career")
    if evidence > 0.5:
        quote_parts.append("JD-aligned evidence in role descriptions")
    if not math.isnan(fusion) and fusion > 0.5:
        quote_parts.append("strong semantic match to JD intent")
    if penalty > 0.5:
        quote_parts.append("disqualifier signals present")
    if not quote_parts:
        quote_parts.append("limited career evidence for this JD")
    quote = "; ".join(quote_parts[:2])

    return {
        "tier": int(tier),
        "score_100": round(score_100, 2),
        "evidence_quote": quote[:200],
        "mode": mode,
    }


def label_consistency(
    row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], bool]:
    """Primary vs strict label; consistent if tier within 1 and score within 20."""
    a = score_candidate(row, mode="primary")
    b = score_candidate(row, mode="strict")
    ok = abs(a["tier"] - b["tier"]) <= 1 and abs(a["score_100"] - b["score_100"]) <= 20
    return a, b, ok
