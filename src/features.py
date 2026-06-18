"""Feature store (offline) and live featurization (replay).

``build_feature_row`` is shared so offline parquet and replay stay aligned.
Embedding/reranker columns are joined from parquet at replay time.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from . import honeypot, lexicon, parse
from .parse import EDU_TIER_ORDINAL, CareerF, EduF, F, SkillF

# Soft-YoE window (JD: 5-9 "a range not a requirement", ideal 6-8): penalize
# gently outside ~4.5-11, hard only at the junior extreme.
YOE_LOW, YOE_HIGH = 4.5, 11.0
YOE_JUNIOR_HARD = 2.0

# JD-preferred / welcome locations (Pune/Noida preferred; Hyd/Mumbai/Delhi NCR ok).
PREFERRED_CITIES = {
    "pune",
    "noida",
    "hyderabad",
    "mumbai",
    "delhi",
    "new delhi",
    "gurgaon",
    "gurugram",
    "ghaziabad",
    "faridabad",
    "greater noida",
    "ncr",
    "delhi ncr",
}
_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# IC ("individual contributor") evidence: coding/building verbs in a role
# description. The JD rejects seniors who "haven't written production code in 18
# months". Text-based, offline, deterministic.
IC_VERBS = {
    "built",
    "build",
    "building",
    "implemented",
    "developed",
    "designed",
    "wrote",
    "coded",
    "shipped",
    "engineered",
    "deployed",
    "optimized",
    "programmed",
    "created",
    "integrated",
    "trained",
    "fine-tuned",
    "prototyped",
    "refactored",
    "debugged",
}
# Titles that signal an engineering IC role.
_IC_TITLE_WORDS = (
    "engineer",
    "developer",
    "programmer",
    "scientist",
    "sde",
    "researcher",
    "ml ",
    "architect",  # architects still code in many orgs; verbs decide
)
# Titles that signal a pure-management role (no IC unless verbs present).
_MGMT_TITLE_WORDS = (
    "manager",
    "director",
    "head of",
    "vp ",
    "vice president",
    "chief",
    "cto",
    "founder",
)
_WORD_TOKEN_RE = re.compile(r"[a-z][a-z\-]+")

# OFFLINE-only feature columns — filled from the parquet store at replay time,
# NOT computed in rank.py (rule #6). Placeholders are NaN here.
OFFLINE_FEATURES: list[str] = [
    "dense_score",  # Qwen3-Embedding cosine vs JD-intent query variants
    "bm25_score",  # BM25 over career-text document
    "fusion_score",  # tuned convex/RRF fusion of dense + bm25
    "reranker_score",  # Qwen3-Reranker score vs the fixed JD
]

# Schema-derived features (replay-safe; computed from one record alone).
SCHEMA_FEATURES: list[str] = [
    # --- experience / fit ---
    "years_of_experience",  # raw YoE (JD: 5-9 soft, ideal 6-8)
    "yoe_fit",  # soft-window fit score in [0,1]
    "yoe_junior_flag",  # hard junior-extreme flag
    "n_roles",  # number of career roles
    "total_career_months",  # career timeline span
    "mean_tenure_months",  # avg tenure (low+inflation => title-chasing)
    # --- career context (product vs services) ---
    "product_tenure_months",  # tenure at product companies (JD: product bg)
    "is_product_current",  # current company is a product company
    "services_only_flag",  # disqualifier: services-only entire career
    "current_company_size_num",  # founding-team-fit (small product = plus)
    # --- skill / JD coverage (anti-stuffer) ---
    "jd_skill_coverage",  # # of JD clusters covered by skills
    "jd_relevant_skill_count",  # # of skills mapping to a JD cluster
    "claimed_unverified_ratio",  # stuffer tell: JD skills with no evidence
    "assessment_coverage",  # JD skills backed by an assessment score
    "max_assessment_score",  # best platform-administered test score
    "mean_assessment_score",  # mean platform-administered test score
    "evidence_density",  # JD-cluster hits in career descriptions / role
    # --- location / availability (JD: Pune/Noida; "actually available") ---
    "location_fit",  # 1 preferred city .. 0 outside-India-unwilling
    "open_to_work",  # availability flag
    "willing_to_relocate",  # availability flag
    "profile_completeness_score",  # platform engagement
    # --- behavioral (sentinels -> NaN + has_*) ---
    "recruiter_response_rate",  # behavioral (monotone up)
    "interview_completion_rate",  # behavioral (monotone up)
    "notice_period_days",  # availability (monotone down)
    "last_active_recency",  # -(months since active); higher=more recent (up)
    "github_activity_score",  # sentinel -1 -> NaN
    "has_github",  # indicator (no GitHub is common)
    "offer_acceptance_rate",  # sentinel -1 -> NaN
    "has_prior_offers",  # indicator
    "best_edu_tier",  # max education institution tier (tier_1=4 .. tier_4=1)
    "has_edu_tier",  # indicator (unknown -> 0)
    "best_grade_value",  # numeric grade extracted from education (best-effort)
    "has_grade",  # indicator (null grade -> 0)
    "has_assessments",  # empty skill_assessment_scores -> 0
    # --- disqualifier detectors (cheap; rule/text-based, offline) ---
    "title_chasing_flag",  # mean tenure < ~20mo over last 3+ hops
    "months_since_last_ic",  # months since last role w/ IC coding evidence
    "no_recent_ic_flag",  # 1 if months_since_last_ic > 18 (JD "writes code")
    "cv_speech_skill_count",  # CV/speech/robotics skills (JD down-rank)
    "cv_speech_without_ir_flag",  # CV/speech present AND no NLP/IR/retrieval evidence
    # --- coherence (honeypot signal, also a learned feature) ---
    "n_hard_flags",  # # hard consistency violations
    "n_soft_flags",  # # soft consistency flags
]

FEATURE_COLUMNS: list[str] = SCHEMA_FEATURES + OFFLINE_FEATURES

# Monotone constraints for LightGBM (ARCHITECTURE §3.4 / redrob_signals_doc).
MONOTONE_CONSTRAINTS: dict[str, int] = {
    "recruiter_response_rate": 1,
    "interview_completion_rate": 1,
    "notice_period_days": -1,
    "last_active_recency": 1,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _num(value: Any) -> float:
    """Coerce to float, mapping the -1 sentinel and non-numerics to NaN."""
    if value is None:
        return math.nan
    if parse.is_numeric_sentinel(value):
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _career(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.CAREER_HISTORY)
    return val if isinstance(val, list) else []


def _skills(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.SKILLS)
    return val if isinstance(val, list) else []


def _education(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.EDUCATION)
    return val if isinstance(val, list) else []


def soft_yoe_fit(yoe: float | None) -> float:
    """Soft-window YoE fit in [0,1]: 1 inside [4.5,11], decaying outside."""
    if yoe is None or (isinstance(yoe, float) and math.isnan(yoe)):
        return math.nan
    if YOE_LOW <= yoe <= YOE_HIGH:
        return 1.0
    if yoe < YOE_LOW:
        return max(0.0, 1.0 - (YOE_LOW - yoe) / YOE_LOW)
    return max(0.0, 1.0 - (yoe - YOE_HIGH) / YOE_HIGH)


def company_size_to_num(value: Any) -> float:
    """Map a company_size enum bucket to a representative headcount."""
    if not isinstance(value, str):
        return math.nan
    return float(parse.COMPANY_SIZE_MIDPOINT.get(value.strip(), math.nan))


def location_fit(record: dict[str, Any]) -> float:
    """JD location preference: Pune/Noida preferred; relocation; no visa sponsor.

    1.0 in a preferred/welcome city; 0.6 elsewhere in India; 0.4 outside India
    but willing to relocate; 0.0 outside India and unwilling (near-gate)."""
    country = str(record.get(F.COUNTRY, "")).strip().lower()
    loc = str(record.get(F.LOCATION, "")).strip().lower()
    in_india = "india" in country
    city_hit = any(city in loc for city in PREFERRED_CITIES)
    willing = bool(record.get(F.WILLING_TO_RELOCATE))
    if in_india and city_hit:
        return 1.0
    if in_india:
        return 0.6
    if willing:
        return 0.4
    return 0.0


def _best_edu_tier(education: list[dict[str, Any]]) -> tuple[float, float]:
    """Return (best_tier_ordinal, has_tier) over education entries."""
    ordinals = []
    for e in education:
        t = e.get(EduF.TIER)
        if not parse.is_missing_tier(t) and t in EDU_TIER_ORDINAL:
            ordinals.append(EDU_TIER_ORDINAL[t])
    if not ordinals:
        return math.nan, 0.0
    return float(max(ordinals)), 1.0


def _best_grade(education: list[dict[str, Any]]) -> tuple[float, float]:
    """Return (best numeric grade extracted, has_grade) — best-effort.

    Grades are free-text ("8.24 CGPA", "First Class", "78%"); we extract the
    leading number when present. ``has_grade`` is 1 if any non-null grade."""
    has = 0.0
    best = math.nan
    for e in education:
        g = e.get(EduF.GRADE)
        if g is None:
            continue
        has = 1.0
        m = _NUM_RE.search(str(g))
        if m:
            val = float(m.group())
            best = val if math.isnan(best) else max(best, val)
    return best, has


# ---------------------------------------------------------------------------
# The featurizer
# ---------------------------------------------------------------------------
def build_feature_row(
    record: dict[str, Any],
    *,
    now: _dt.date | None = None,
    founding_years: dict[str, int] | None = None,
) -> dict[str, float]:
    """Compute the schema-derived features for one candidate.

    OFFLINE columns are included as NaN placeholders so the row has the full
    schema; replay fills them from the parquet join (NaN is fine for LightGBM)."""
    row: dict[str, float] = {c: math.nan for c in FEATURE_COLUMNS}

    career = _career(record)
    skills = _skills(record)
    education = _education(record)

    # --- experience ---
    yoe = _num(record.get(F.YEARS_OF_EXPERIENCE))
    row["years_of_experience"] = yoe
    row["yoe_fit"] = soft_yoe_fit(yoe if not math.isnan(yoe) else None)
    row["yoe_junior_flag"] = (
        1.0 if (not math.isnan(yoe) and yoe < YOE_JUNIOR_HARD) else 0.0
    )
    row["n_roles"] = float(len(career))

    durations = [
        float(r.get(CareerF.DURATION_MONTHS))
        for r in career
        if isinstance(r.get(CareerF.DURATION_MONTHS), (int, float))
    ]
    span = honeypot._career_span_months(record, now)
    row["total_career_months"] = float(span) if span is not None else math.nan
    row["mean_tenure_months"] = float(np.mean(durations)) if durations else math.nan

    # --- career context ---
    product_months = 0.0
    for r in career:
        if lexicon.is_product_company(r.get(CareerF.COMPANY)):
            d = r.get(CareerF.DURATION_MONTHS)
            if isinstance(d, (int, float)):
                product_months += float(d)
    row["product_tenure_months"] = product_months
    row["is_product_current"] = (
        1.0 if lexicon.is_product_company(record.get(F.CURRENT_COMPANY)) else 0.0
    )
    if career:
        types = {lexicon.classify_company(r.get(CareerF.COMPANY)) for r in career}
        row["services_only_flag"] = 1.0 if types == {lexicon.COMPANY_SERVICES} else 0.0
    else:
        row["services_only_flag"] = 0.0
    row["current_company_size_num"] = company_size_to_num(
        record.get(F.CURRENT_COMPANY_SIZE)
    )

    # --- skills / JD coverage (anti-stuffer) ---
    skill_names = [s.get(SkillF.NAME) for s in skills]
    coverage = lexicon.map_skills(skill_names)
    row["jd_skill_coverage"] = float(sum(1 for v in coverage.values() if v > 0))
    jd_rel = [s for s in skills if lexicon.map_skill(s.get(SkillF.NAME))]
    row["jd_relevant_skill_count"] = float(len(jd_rel))

    assessments = record.get(F.SKILL_ASSESSMENT_SCORES)
    assessments = assessments if isinstance(assessments, dict) else {}
    row["has_assessments"] = 1.0 if assessments else 0.0
    score_vals = [float(v) for v in assessments.values() if isinstance(v, (int, float))]
    row["max_assessment_score"] = float(max(score_vals)) if score_vals else math.nan
    row["mean_assessment_score"] = (
        float(np.mean(score_vals)) if score_vals else math.nan
    )

    # claimed_unverified_ratio: of JD-relevant skills, the fraction lacking
    # evidence (0 months AND no assessment). High => keyword stuffer.
    if jd_rel:
        assessed_names = {str(k).strip().lower() for k in assessments}
        unverified = 0
        for s in jd_rel:
            dur = s.get(SkillF.DURATION_MONTHS)
            nm = str(s.get(SkillF.NAME, "")).strip().lower()
            has_dur = isinstance(dur, (int, float)) and dur > 0
            has_assess = nm in assessed_names
            if not has_dur and not has_assess:
                unverified += 1
        row["claimed_unverified_ratio"] = unverified / len(jd_rel)
        row["assessment_coverage"] = sum(
            1
            for s in jd_rel
            if str(s.get(SkillF.NAME, "")).strip().lower() in assessed_names
        ) / len(jd_rel)
    else:
        row["claimed_unverified_ratio"] = math.nan
        row["assessment_coverage"] = math.nan

    # evidence_density: JD-cluster keyword hits in career descriptions / role.
    hits = 0
    for r in career:
        desc = r.get(CareerF.DESCRIPTION)
        if desc:
            hits += len(lexicon.map_skill(desc))  # phrase matching over text
    row["evidence_density"] = hits / len(career) if career else math.nan

    # --- location / availability ---
    row["location_fit"] = location_fit(record)
    row["open_to_work"] = 1.0 if record.get(F.OPEN_TO_WORK_FLAG) else 0.0
    row["willing_to_relocate"] = 1.0 if record.get(F.WILLING_TO_RELOCATE) else 0.0
    row["profile_completeness_score"] = _num(record.get(F.PROFILE_COMPLETENESS_SCORE))

    # --- behavioral (sentinels -> NaN + has_*) ---
    row["recruiter_response_rate"] = _num(record.get(F.RECRUITER_RESPONSE_RATE))
    row["interview_completion_rate"] = _num(record.get(F.INTERVIEW_COMPLETION_RATE))
    row["notice_period_days"] = _num(record.get(F.NOTICE_PERIOD_DAYS))

    gh = record.get(F.GITHUB_ACTIVITY_SCORE)
    row["github_activity_score"] = _num(gh)
    row["has_github"] = 0.0 if (gh is None or parse.is_numeric_sentinel(gh)) else 1.0
    oa = record.get(F.OFFER_ACCEPTANCE_RATE)
    row["offer_acceptance_rate"] = _num(oa)
    row["has_prior_offers"] = (
        0.0 if (oa is None or parse.is_numeric_sentinel(oa)) else 1.0
    )

    # --- education tier/grade (live on education[]) ---
    row["best_edu_tier"], row["has_edu_tier"] = _best_edu_tier(education)
    row["best_grade_value"], row["has_grade"] = _best_grade(education)

    # recency: months since last active relative to reference "now".
    la = parse.parse_date(record.get(F.LAST_ACTIVE_DATE))
    if la is not None and now is not None:
        months_since = parse.months_between(la, now)
        row["last_active_recency"] = (
            -float(months_since) if months_since is not None else math.nan
        )
    else:
        row["last_active_recency"] = math.nan

    # --- disqualifier detectors ---
    row["title_chasing_flag"] = _title_chasing(career)
    msl_ic = _months_since_last_ic(career, now)
    row["months_since_last_ic"] = msl_ic
    row["no_recent_ic_flag"] = (
        1.0 if (not math.isnan(msl_ic) and msl_ic > 18.0) else 0.0
    )
    cv_count = lexicon.cv_speech_robotics_skill_count(skill_names)
    row["cv_speech_skill_count"] = float(cv_count)
    has_ir_nlp = (row["jd_skill_coverage"] or 0) > 0 or (
        not math.isnan(row["evidence_density"]) and row["evidence_density"] > 0
    )
    row["cv_speech_without_ir_flag"] = (
        1.0 if (cv_count > 0 and not has_ir_nlp) else 0.0
    )

    # --- coherence (honeypot signal as a learned feature) ---
    assessment = honeypot.run_consistency_suite(
        record, founding_years=founding_years or {}, now=now
    )
    row["n_hard_flags"] = float(
        sum(1 for r in assessment.results if r.failed and r.severity == honeypot.HARD)
    )
    row["n_soft_flags"] = float(assessment.soft_count)

    return row


def _role_is_ic(role: dict[str, Any]) -> bool:
    """Heuristic: does this role show individual-contributor coding evidence?

    True if the title is an engineering title OR the description contains IC
    "build/ship/implement" verbs — unless the title is purely managerial with no
    such verbs (the JD's "moved into architecture/tech-lead, stopped coding")."""
    title = str(role.get(CareerF.TITLE, "")).lower()
    desc = str(role.get(CareerF.DESCRIPTION, "")).lower()
    desc_tokens = set(_WORD_TOKEN_RE.findall(desc))
    has_verb = bool(desc_tokens & IC_VERBS)
    title_eng = any(w in title for w in _IC_TITLE_WORDS)
    title_mgmt = any(w in title for w in _MGMT_TITLE_WORDS)
    if title_mgmt and not has_verb and not title_eng:
        return False
    return title_eng or has_verb


def _months_since_last_ic(
    career: list[dict[str, Any]], now: _dt.date | None
) -> float:
    """Months from ``now`` to the end of the most recent IC role.

    0.0 if a current role shows IC evidence. If no role shows IC evidence, return
    the full career span (so the >18-month flag fires). NaN only if we cannot
    establish any timeline."""
    if not career:
        return math.nan
    best: float | None = None
    for role in career:
        if not _role_is_ic(role):
            continue
        if role.get(CareerF.IS_CURRENT):
            return 0.0
        end = parse.parse_date(role.get(CareerF.END_DATE))
        if end is not None and now is not None:
            gap = parse.months_between(end, now)
            if gap is not None and gap >= 0:
                best = gap if best is None else min(best, gap)
    if best is not None:
        return float(best)
    # no datable IC role found -> treat as "no recent IC" using the career span
    span = honeypot._career_span_months(
        {F.CAREER_HISTORY: career}, now
    )
    return float(span) if span is not None else math.nan


def _title_chasing(career: list[dict[str, Any]]) -> float:
    """Disqualifier: switching every ~1.5 years over the last 3+ hops.

    Cheap version: mean tenure of the last 3 completed roles < 20 months.
    (Monotone title-inflation is a TODO once titles are normalized.)"""
    durs = [
        float(r.get(CareerF.DURATION_MONTHS))
        for r in career
        if isinstance(r.get(CareerF.DURATION_MONTHS), (int, float))
    ]
    if len(durs) < 3:
        return 0.0
    last3 = durs[:3]
    return 1.0 if float(np.mean(last3)) < 20.0 else 0.0


def live_features(
    record: dict[str, Any],
    *,
    now: _dt.date | None = None,
    founding_years: dict[str, int] | None = None,
) -> dict[str, float]:
    """Replay subset: schema-derived features only (no OFFLINE columns)."""
    full = build_feature_row(record, now=now, founding_years=founding_years)
    return {k: full[k] for k in SCHEMA_FEATURES}


def build_feature_frame(
    records: list[dict[str, Any]],
    *,
    now: _dt.date | None = None,
    founding_years: dict[str, int] | None = None,
) -> pd.DataFrame:
    """OFFLINE: build the full feature DataFrame (one row per candidate).

    Convenience for the sample/tests. For the full 100K pool use
    :func:`write_features_parquet`, which streams and never materializes the pool.
    """
    rows = []
    ids = []
    for rec in records:
        ids.append(rec.get(F.CANDIDATE_ID))
        rows.append(build_feature_row(rec, now=now, founding_years=founding_years))
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    df.insert(0, F.CANDIDATE_ID, ids)
    return df


def _parquet_schema():
    """Explicit, stable Arrow schema: candidate_id string + float64 features.

    Pinning the schema keeps every batch identical (so the all-NaN OFFLINE
    columns don't get inferred as a different type batch-to-batch)."""
    import pyarrow as pa

    fields = [pa.field(F.CANDIDATE_ID, pa.string())]
    fields += [pa.field(c, pa.float64()) for c in FEATURE_COLUMNS]
    return pa.schema(fields)


def write_features_parquet(
    records: Iterable[dict[str, Any]],
    out_path: str,
    *,
    now: _dt.date | None = None,
    founding_years: dict[str, int] | None = None,
    batch_size: int = 5000,
) -> int:
    """Stream candidate records → ``features.parquet`` in fixed-size batches.

    Memory stays flat: only one ``batch_size`` block of feature rows (≈80 floats
    each) is held at a time; the 465 MB pool is never materialized. ``records``
    may be any iterator (e.g. ``parse.stream_candidates``). Returns the row count.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _parquet_schema()
    writer = pq.ParquetWriter(out_path, schema)
    ids: list[Any] = []
    rows: list[dict[str, float]] = []
    total = 0

    def _flush() -> None:
        nonlocal ids, rows
        if not rows:
            return
        arrays = [pa.array([str(i) for i in ids], type=pa.string())]
        for c in FEATURE_COLUMNS:
            arrays.append(pa.array([r[c] for r in rows], type=pa.float64()))
        writer.write_table(pa.Table.from_arrays(arrays, schema=schema))
        ids = []
        rows = []

    try:
        for rec in records:
            ids.append(rec.get(F.CANDIDATE_ID))
            rows.append(
                build_feature_row(rec, now=now, founding_years=founding_years)
            )
            total += 1
            if len(rows) >= batch_size:
                _flush()
        _flush()
    finally:
        writer.close()
    return total
