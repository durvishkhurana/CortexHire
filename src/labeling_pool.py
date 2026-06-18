"""High-recall labeling pool (WORKFLOW STEP 7).

Union: top-5K dense ∪ top-5K BM25 ∪ top-5K rules-shortlist ∪ honeypot suspects
∪ 1–2K stratified random. Deduped with provenance bit-flags.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from .parse import F

PROV_DENSE = "dense_top5k"
PROV_BM25 = "bm25_top5k"
PROV_RULES = "rules_top5k"
PROV_HONEYPOT = "honeypot_suspect"
PROV_RANDOM = "random_stratified"

RULES_WEIGHTS = {
    "evidence_density": 1.0,
    "is_product_current": 1.0,
    "product_tenure_months": 0.004,
    "yoe_fit": 1.0,
    "jd_skill_coverage": 0.4,
    "assessment_coverage": 0.8,
    "max_assessment_score": 0.004,
    "location_fit": 0.4,
    "open_to_work": 0.2,
    "has_assessments": 0.1,
    "recruiter_response_rate": 0.6,
    "interview_completion_rate": 0.4,
    "last_active_recency": 0.01,
    "fusion_score": 0.8,
    "dense_score": 0.5,
    "bm25_score": 0.3,
    "reranker_score": 1.2,
    "claimed_unverified_ratio": -1.2,
    "cv_speech_without_ir_flag": -1.2,
    "services_only_flag": -1.0,
    "no_recent_ic_flag": -1.0,
    "title_chasing_flag": -0.8,
    "yoe_junior_flag": -1.0,
    "notice_period_days": -0.002,
    "n_hard_flags": -3.0,
    "n_soft_flags": -0.3,
}


def rules_shortlist_score(row: dict[str, Any]) -> float:
    s = 0.0
    for feat, w in RULES_WEIGHTS.items():
        v = row.get(feat)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        s += w * float(v)
    return s


def assemble_labeling_pool_df(
    features_df: pd.DataFrame,
    *,
    top_k: int = 5000,
    random_n: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """Build deduped pool with provenance columns from a features DataFrame."""
    df = features_df.copy()
    cid_col = F.CANDIDATE_ID
    if cid_col not in df.columns:
        raise ValueError(f"missing {cid_col}")

    df["_rules_score"] = df.apply(
        lambda r: rules_shortlist_score(r.to_dict()), axis=1
    )

    dense_top = set(
        df.nlargest(top_k, "dense_score", keep="first")[cid_col].astype(str)
    )
    bm25_top = set(
        df.nlargest(top_k, "bm25_score", keep="first")[cid_col].astype(str)
    )
    rules_top = set(
        df.nlargest(top_k, "_rules_score", keep="first")[cid_col].astype(str)
    )
    honeypot = set(
        df.loc[df["n_hard_flags"].fillna(0) >= 1, cid_col].astype(str).tolist()
    )

    rng = np.random.default_rng(seed)
    all_ids = df[cid_col].astype(str).tolist()
    random_ids = set(rng.choice(all_ids, size=min(random_n, len(all_ids)), replace=False))

    pool_ids: list[str] = []
    provenance: dict[str, list[str]] = {}

    def _add(cid: str, tag: str) -> None:
        if cid not in provenance:
            provenance[cid] = []
            pool_ids.append(cid)
        if tag not in provenance[cid]:
            provenance[cid].append(tag)

    for cid in dense_top:
        _add(cid, PROV_DENSE)
    for cid in bm25_top:
        _add(cid, PROV_BM25)
    for cid in rules_top:
        _add(cid, PROV_RULES)
    for cid in honeypot:
        _add(cid, PROV_HONEYPOT)
    for cid in random_ids:
        _add(cid, PROV_RANDOM)

    rows = []
    for cid in pool_ids:
        tags = provenance[cid]
        rows.append(
            {
                cid_col: cid,
                "provenance": "|".join(tags),
                "from_dense": float(PROV_DENSE in tags),
                "from_bm25": float(PROV_BM25 in tags),
                "from_rules": float(PROV_RULES in tags),
                "from_honeypot": float(PROV_HONEYPOT in tags),
                "from_random": float(PROV_RANDOM in tags),
            }
        )
    out = pd.DataFrame(rows)
    return out


def load_features_parquet(path: str) -> pd.DataFrame:
    import polars as pl

    return pl.read_parquet(path).to_pandas()
