"""Streaming JSONL parser + centralized schema for the candidate pool.

Why this module exists
----------------------
The grader runs ``rank.py --candidates ./candidates.jsonl`` on the ~487 MB file.
We must **never** ``json.loads`` the whole file into a list. Instead
:func:`stream_candidates` is an ``orjson`` line generator that yields one
*projected* record at a time, holding only the fields we use.

Schema (confirmed against the real ``candidate_schema.json``)
-------------------------------------------------------------
The raw record nests scalars under ``profile`` and ``redrob_signals``::

    { candidate_id, profile{...}, career_history[], education[], skills[],
      certifications[], languages[], redrob_signals{...} }

For ergonomics, :func:`project_record` **flattens** ``profile.*`` and
``redrob_signals.*`` to the top level of the projected record (their keys don't
collide), and keeps the nested *lists* as-is. So downstream code reads e.g.
``record[F.YEARS_OF_EXPERIENCE]`` and ``record[F.LAST_ACTIVE_DATE]`` directly.
``tier`` and ``grade`` live on **education[]** items (not the profile).

Determinism: recency reference "now" = ``max(last_active_date)`` across the pool
(:func:`reference_now`), **never** ``datetime.now()``.
"""

from __future__ import annotations

import datetime as _dt
from collections.abc import Iterable, Iterator
from functools import lru_cache
from typing import Any

import orjson


# ---------------------------------------------------------------------------
# Schema registry (matches candidate_schema.json). Flattened top-level names.
# ---------------------------------------------------------------------------
class F:
    """Top-level field names of a *projected* (flattened) record."""

    CANDIDATE_ID = "candidate_id"

    # --- profile.* (flattened) ---
    ANONYMIZED_NAME = "anonymized_name"
    HEADLINE = "headline"
    SUMMARY = "summary"
    LOCATION = "location"
    COUNTRY = "country"
    YEARS_OF_EXPERIENCE = "years_of_experience"
    CURRENT_TITLE = "current_title"
    CURRENT_COMPANY = "current_company"
    CURRENT_COMPANY_SIZE = "current_company_size"
    CURRENT_INDUSTRY = "current_industry"

    # --- nested collections ---
    CAREER_HISTORY = "career_history"
    EDUCATION = "education"
    SKILLS = "skills"
    CERTIFICATIONS = "certifications"
    LANGUAGES = "languages"

    # --- redrob_signals.* (flattened) — the 23 behavioral signals ---
    PROFILE_COMPLETENESS_SCORE = "profile_completeness_score"
    SIGNUP_DATE = "signup_date"
    LAST_ACTIVE_DATE = "last_active_date"
    OPEN_TO_WORK_FLAG = "open_to_work_flag"
    PROFILE_VIEWS_30D = "profile_views_received_30d"
    APPLICATIONS_30D = "applications_submitted_30d"
    RECRUITER_RESPONSE_RATE = "recruiter_response_rate"
    AVG_RESPONSE_TIME_HOURS = "avg_response_time_hours"
    SKILL_ASSESSMENT_SCORES = "skill_assessment_scores"
    CONNECTION_COUNT = "connection_count"
    ENDORSEMENTS_RECEIVED = "endorsements_received"
    NOTICE_PERIOD_DAYS = "notice_period_days"
    EXPECTED_SALARY_RANGE = "expected_salary_range_inr_lpa"
    PREFERRED_WORK_MODE = "preferred_work_mode"
    WILLING_TO_RELOCATE = "willing_to_relocate"
    GITHUB_ACTIVITY_SCORE = "github_activity_score"
    SEARCH_APPEARANCE_30D = "search_appearance_30d"
    SAVED_BY_RECRUITERS_30D = "saved_by_recruiters_30d"
    INTERVIEW_COMPLETION_RATE = "interview_completion_rate"
    OFFER_ACCEPTANCE_RATE = "offer_acceptance_rate"
    VERIFIED_EMAIL = "verified_email"
    VERIFIED_PHONE = "verified_phone"
    LINKEDIN_CONNECTED = "linkedin_connected"


class CareerF:
    """``career_history[]`` element field names."""

    COMPANY = "company"
    TITLE = "title"
    START_DATE = "start_date"
    END_DATE = "end_date"
    DURATION_MONTHS = "duration_months"
    IS_CURRENT = "is_current"
    INDUSTRY = "industry"
    COMPANY_SIZE = "company_size"
    DESCRIPTION = "description"


class SkillF:
    """``skills[]`` element field names."""

    NAME = "name"
    PROFICIENCY = "proficiency"  # enum: beginner/intermediate/advanced/expert
    ENDORSEMENTS = "endorsements"
    DURATION_MONTHS = "duration_months"


class EduF:
    """``education[]`` element field names (tier/grade live HERE)."""

    INSTITUTION = "institution"
    DEGREE = "degree"
    FIELD_OF_STUDY = "field_of_study"
    START_YEAR = "start_year"
    END_YEAR = "end_year"
    GRADE = "grade"  # str OR null
    TIER = "tier"  # enum: tier_1..tier_4, unknown


class CertF:
    """``certifications[]`` element field names."""

    NAME = "name"
    ISSUER = "issuer"
    YEAR = "year"


# Sentinel values (rule #5): absence is never a penalty -> NaN + has_* indicator.
SENTINEL_NUMERIC = -1
TIER_UNKNOWN = "unknown"

# Skill proficiency enum ordering.
PROFICIENCY_ORDINAL = {"beginner": 1, "intermediate": 2, "advanced": 3, "expert": 4}

# Education institution-tier ordinal (tier_1 best). "unknown" -> missing.
EDU_TIER_ORDINAL = {"tier_1": 4, "tier_2": 3, "tier_3": 2, "tier_4": 1}

# company_size enum -> representative headcount (midpoint-ish) for numeric use.
COMPANY_SIZE_MIDPOINT = {
    "1-10": 5,
    "11-50": 30,
    "51-200": 125,
    "201-500": 350,
    "501-1000": 750,
    "1001-5000": 3000,
    "5001-10000": 7500,
    "10001+": 15000,
}

# Fields flattened from the nested profile / redrob_signals objects.
_PROFILE_KEYS = (
    F.ANONYMIZED_NAME,
    F.HEADLINE,
    F.SUMMARY,
    F.LOCATION,
    F.COUNTRY,
    F.YEARS_OF_EXPERIENCE,
    F.CURRENT_TITLE,
    F.CURRENT_COMPANY,
    F.CURRENT_COMPANY_SIZE,
    F.CURRENT_INDUSTRY,
)
_SIGNAL_KEYS = (
    F.PROFILE_COMPLETENESS_SCORE,
    F.SIGNUP_DATE,
    F.LAST_ACTIVE_DATE,
    F.OPEN_TO_WORK_FLAG,
    F.PROFILE_VIEWS_30D,
    F.APPLICATIONS_30D,
    F.RECRUITER_RESPONSE_RATE,
    F.AVG_RESPONSE_TIME_HOURS,
    F.SKILL_ASSESSMENT_SCORES,
    F.CONNECTION_COUNT,
    F.ENDORSEMENTS_RECEIVED,
    F.NOTICE_PERIOD_DAYS,
    F.EXPECTED_SALARY_RANGE,
    F.PREFERRED_WORK_MODE,
    F.WILLING_TO_RELOCATE,
    F.GITHUB_ACTIVITY_SCORE,
    F.SEARCH_APPEARANCE_30D,
    F.SAVED_BY_RECRUITERS_30D,
    F.INTERVIEW_COMPLETION_RATE,
    F.OFFER_ACCEPTANCE_RATE,
    F.VERIFIED_EMAIL,
    F.VERIFIED_PHONE,
    F.LINKEDIN_CONNECTED,
)
_NESTED_LISTS = (F.CAREER_HISTORY, F.EDUCATION, F.SKILLS, F.CERTIFICATIONS)

# Behavioral signals with known monotone direction (used by the model).
NAMED_BEHAVIORAL = (
    F.RECRUITER_RESPONSE_RATE,
    F.INTERVIEW_COMPLETION_RATE,
    F.GITHUB_ACTIVITY_SCORE,
    F.OFFER_ACCEPTANCE_RATE,
)


# ---------------------------------------------------------------------------
# Streaming parse
# ---------------------------------------------------------------------------
def project_record(raw: dict[str, Any]) -> dict[str, Any]:
    """Project + flatten a raw candidate dict to the fields we use.

    Scalars from ``profile`` and ``redrob_signals`` are lifted to the top level;
    the nested *lists* are kept as-is. ``languages`` is dropped (unused).
    """
    rec: dict[str, Any] = {}
    if F.CANDIDATE_ID in raw:
        rec[F.CANDIDATE_ID] = raw[F.CANDIDATE_ID]

    profile = raw.get("profile") or {}
    for key in _PROFILE_KEYS:
        if key in profile:
            rec[key] = profile[key]

    signals = raw.get("redrob_signals") or {}
    for key in _SIGNAL_KEYS:
        if key in signals:
            rec[key] = signals[key]

    for key in _NESTED_LISTS:
        if key in raw:
            rec[key] = raw[key]
    return rec


def stream_candidates(path: str, *, project: bool = True) -> Iterator[dict[str, Any]]:
    """Yield candidate records one at a time from a JSONL file.

    The honest streaming primitive: the file is read **line by line** and each
    line is parsed with ``orjson`` independently. The whole file is *never*
    materialized. Blank lines are skipped.

    Args:
        path: path to a ``.jsonl`` file (one JSON object per line).
        project: if True (default), yield flattened projected records; if False,
            yield the full raw object (e.g. for schema discovery).
    """
    with open(path, "rb") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = orjson.loads(line)
            yield project_record(obj) if project else obj


def stream_json_array(path: str, *, project: bool = True) -> Iterator[dict[str, Any]]:
    """Yield records from a pretty-printed JSON *array* (e.g. sample_candidates.json).

    Used only for the small sample file. The 487 MB pool is JSONL and must use
    :func:`stream_candidates`. This still avoids holding the parsed objects in a
    list (it yields one at a time), though it does read the sample bytes once.
    """
    with open(path, "rb") as fh:
        data = orjson.loads(fh.read())
    for obj in data:
        yield project_record(obj) if project else obj


# ---------------------------------------------------------------------------
# Date handling + reference "now"
# ---------------------------------------------------------------------------
@lru_cache(maxsize=200_000)
def parse_date(value: Any) -> _dt.date | None:
    """Best-effort parse of a date-ish value into a ``date``.

    Handles ``YYYY-MM-DD``, ``YYYY-MM``, ``YYYY``, ISO datetimes, and a few
    common separators. Returns ``None`` for None/empty/sentinel-ish strings and
    anything unparseable.
    """
    if value is None:
        return None
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    if isinstance(value, int):
        # bare year (e.g. education start_year)
        if 1900 <= value <= 2100:
            return _dt.date(value, 1, 1)
        return None
    if not isinstance(value, str):
        return None

    s = value.strip()
    if not s or s.lower() in {"present", "current", "now", "n/a", "na", "null"}:
        return None

    s = s.replace("/", "-")
    s = s.split("T")[0].split(" ")[0]
    parts = s.split("-")
    try:
        if len(parts) == 1:
            return _dt.date(int(parts[0]), 1, 1)
        if len(parts) == 2:
            return _dt.date(int(parts[0]), int(parts[1]), 1)
        if len(parts) >= 3:
            return _dt.date(int(parts[0]), int(parts[1]), int(parts[2]))
    except (ValueError, TypeError):
        return None
    return None


def reference_now(dates: Iterable[Any]) -> _dt.date | None:
    """Reference "now" for recency features = ``max(last_active_date)``.

    Deterministic by construction. Pass the pool's ``last_active_date`` values.
    """
    best: _dt.date | None = None
    for d in dates:
        parsed = d if isinstance(d, _dt.date) else parse_date(d)
        if parsed is None:
            continue
        if best is None or parsed > best:
            best = parsed
    return best


def reference_now_from_path(path: str) -> _dt.date | None:
    """Stream the JSONL pool once and return ``max(last_active_date)``."""
    return reference_now(rec.get(F.LAST_ACTIVE_DATE) for rec in stream_candidates(path))


# ---------------------------------------------------------------------------
# Sentinel + small helpers (rule #5)
# ---------------------------------------------------------------------------
def is_numeric_sentinel(value: Any) -> bool:
    """True if a numeric field carries the ``-1`` "absent" sentinel."""
    return value is not None and value == SENTINEL_NUMERIC


def is_missing_tier(value: Any) -> bool:
    """True if an education tier is absent/unknown."""
    return value is None or (
        isinstance(value, str) and value.strip().lower() == TIER_UNKNOWN
    )


def months_between(start: _dt.date | None, end: _dt.date | None) -> float | None:
    """Whole-ish months between two dates (end - start). None if either missing."""
    if start is None or end is None:
        return None
    return (
        (end.year - start.year) * 12
        + (end.month - start.month)
        + ((end.day - start.day) / 30.0)
    )


def candidate_sort_key(candidate_id: Any):
    """Tiebreak key for candidate_id ascending — matches the organizer validator.

    The organizer compares candidate_ids as strings (``c1 > c2``). CAND_ ids are
    fixed-width (``CAND_`` + 7 digits) so string order == numeric order. We
    therefore use plain string comparison to stay byte-for-byte consistent.
    """
    return str(candidate_id)


def load_pool(path: str) -> list[dict[str, Any]]:
    """Materialize the projected pool into a list (sandbox/tests convenience).

    Auto-detects a JSON array (``.json``) vs JSONL by extension. For the full
    100K JSONL pool prefer iterating :func:`stream_candidates` and building
    columnar arrays incrementally; this helper is for the sample + tests.
    """
    if path.lower().endswith(".json"):
        return list(stream_json_array(path))
    return list(stream_candidates(path))
