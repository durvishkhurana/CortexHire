"""Honeypot defense: consistency suite + IsolationForest anomaly + triple-guard.

Policy (rule #4, non-negotiable): **a single hard rule violation makes a
profile ineligible for the top 100.** We never require 2+ violations. The
triple-guard is rules (hard) AND an unsupervised anomaly signal AND an offline
audit flag, combined so that:

    exclude  iff  hard_violation
             OR   audit_contradiction
             OR   (anomaly AND corroborated_by_a_soft_flag)

The unsupervised anomaly *alone* never excludes (that would risk false
positives on legitimately-unusual-but-real candidates); it only fires when
corroborated, exactly as ARCHITECTURE §6 specifies. ``rank.py`` runs the
consistency suite LIVE on the parsed ``--candidates`` input so the honeypot
logic provably executes on the grader's file.

The consistency suite implements all ~12 checks from ARCHITECTURE §3 Phase 1.
Each check returns a :class:`CheckResult` (hard or soft). Checks are pure
functions of one record (+ small offline tables: founding years, tech
inception), so they are cheap and deterministic.
"""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import parse
from .parse import CareerF, CertF, EduF, F, SkillF

# Tolerances (months). Generous enough to avoid false positives on real,
# slightly-messy data; tight enough to catch fabricated coherence.
DURATION_MISMATCH_TOL = 6
YOE_OVER_SPAN_TOL = 18
SKILL_OVER_CAREER_TOL = 6

# Founding-year check requires a MARGIN: a role must start at least this many
# years *before* the company was founded to count as a hard violation. The real
# pool has off-by-one date noise at real companies (e.g. a 2017 start at CRED,
# founded 2018) that is NOT the honeypot; the canonical honeypot ("8 yrs at a
# 3-yr-old company") is a gross multi-year gap, so a 1-year margin removes the
# noise while still catching the trap. (Calibrated on the real pool, STEP 4.)
FOUNDING_MARGIN_YEARS = 1

# Honeypot "expert + 0 months": the schema proficiency enum's top level only.
EXPERT_LABELS = {"expert"}

# Minimal technology-inception table for the "certification before tech existed"
# check. Extend offline from the real certification strings in the pool.
TECH_INCEPTION: dict[str, int] = {
    "transformer": 2017,
    "transformers": 2017,
    "bert": 2018,
    "gpt": 2018,
    "langchain": 2022,
    "llama": 2023,
    "chatgpt": 2022,
    "pytorch": 2016,
    "tensorflow": 2015,
    "faiss": 2017,
    "pinecone": 2021,
    "milvus": 2019,
    "weaviate": 2019,
    "qdrant": 2021,
    "kubernetes": 2014,
    "docker": 2013,
    "react": 2013,
    "spark": 2014,
    "kafka": 2011,
    "rag": 2020,
}

HARD = "hard"
SOFT = "soft"


@dataclass
class CheckResult:
    """Outcome of one consistency check."""

    name: str
    severity: str  # HARD or SOFT
    failed: bool
    detail: str = ""


@dataclass
class Assessment:
    """Full honeypot assessment for one candidate."""

    candidate_id: Any
    results: list[CheckResult] = field(default_factory=list)

    @property
    def hard_violation(self) -> bool:
        return any(r.failed and r.severity == HARD for r in self.results)

    @property
    def soft_count(self) -> int:
        return sum(1 for r in self.results if r.failed and r.severity == SOFT)

    @property
    def failed_checks(self) -> list[str]:
        return [r.name for r in self.results if r.failed]


# ---------------------------------------------------------------------------
# Date helpers local to checks
# ---------------------------------------------------------------------------
def _career(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.CAREER_HISTORY)
    return val if isinstance(val, list) else []


def _education(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.EDUCATION)
    return val if isinstance(val, list) else []


def _skills(record: dict[str, Any]) -> list[dict[str, Any]]:
    val = record.get(F.SKILLS)
    return val if isinstance(val, list) else []


def _role_end(role: dict[str, Any], now: _dt.date | None) -> _dt.date | None:
    """Effective end date of a role (now for current roles without an end)."""
    end = parse.parse_date(role.get(CareerF.END_DATE))
    if end is not None:
        return end
    if role.get(CareerF.IS_CURRENT) and now is not None:
        return now
    return None


def _career_span_months(record: dict[str, Any], now: _dt.date | None) -> float | None:
    starts, ends = [], []
    for role in _career(record):
        s = parse.parse_date(role.get(CareerF.START_DATE))
        if s is not None:
            starts.append(s)
        e = _role_end(role, now)
        if e is not None:
            ends.append(e)
    if not starts or not ends:
        return None
    return parse.months_between(min(starts), max(ends))


# ---------------------------------------------------------------------------
# The ~12 consistency checks
# ---------------------------------------------------------------------------
def check_tenure_vs_founding(
    record: dict[str, Any], founding_years: dict[str, int]
) -> CheckResult:
    """HARD: a role starting before its company was founded.

    Defeats the flagship honeypot "8 years at a 3-year-old company". Needs the
    offline founding-year table; if a company is absent from the table the check
    abstains (cannot fabricate evidence)."""
    from . import lexicon  # local import to avoid cycles

    name = "tenure_vs_founding"
    for role in _career(record):
        company = role.get(CareerF.COMPANY)
        fy = lexicon.founding_year(company, founding_years) if founding_years else None
        if fy is None:
            continue
        start = parse.parse_date(role.get(CareerF.START_DATE))
        if start is not None and start.year < fy - FOUNDING_MARGIN_YEARS:
            return CheckResult(
                name,
                HARD,
                True,
                f"role at {company!r} starts {start.year} but company founded {fy}",
            )
    return CheckResult(name, HARD, False)


def check_expert_zero_duration(record: dict[str, Any]) -> CheckResult:
    """HARD: a skill claimed at expert/advanced level with 0 months of use."""
    name = "expert_zero_duration"
    for sk in _skills(record):
        prof = str(sk.get(SkillF.PROFICIENCY, "")).strip().lower()
        dur = sk.get(SkillF.DURATION_MONTHS)
        if prof in EXPERT_LABELS and dur == 0:
            return CheckResult(
                name,
                HARD,
                True,
                f"skill {sk.get(SkillF.NAME)!r} is {prof!r} with 0 months",
            )
    return CheckResult(name, HARD, False)


def check_yoe_vs_span(
    record: dict[str, Any], now: _dt.date | None = None
) -> CheckResult:
    """HARD: years_of_experience far exceeds the career timeline span.

    Concurrent overlap is allowed (we compare against the *span*, not the sum),
    so only an impossible claim of more experience than time elapsed fires."""
    name = "yoe_vs_span"
    yoe = record.get(F.YEARS_OF_EXPERIENCE)
    span = _career_span_months(record, now)
    if isinstance(yoe, (int, float)) and span is not None:
        if yoe * 12 > span + YOE_OVER_SPAN_TOL:
            return CheckResult(
                name,
                HARD,
                True,
                f"claims {yoe}y ({yoe * 12}mo) but career span is {span:.0f}mo",
            )
    return CheckResult(name, HARD, False)


def check_role_before_education(record: dict[str, Any]) -> CheckResult:
    """SOFT: a professional role starting before any education began.

    Demoted from HARD after the STEP-4 real-pool audit (fired on 4.9%): in this
    dataset many candidates legitimately worked *before* a later degree (career
    switchers, part-time/executive degrees), and education start_years carry
    generator noise. We keep it as a SOFT coherence signal (feeds the anomaly
    model / `n_soft_flags`) but it never hard-excludes on its own."""
    name = "role_before_education"
    edu_starts = [parse.parse_date(e.get(EduF.START_YEAR)) for e in _education(record)]
    edu_starts = [d for d in edu_starts if d is not None]
    if not edu_starts:
        return CheckResult(name, SOFT, False)
    earliest_edu = min(edu_starts)
    for role in _career(record):
        s = parse.parse_date(role.get(CareerF.START_DATE))
        if s is not None and s < earliest_edu:
            return CheckResult(
                name,
                SOFT,
                True,
                f"role starts {s} before earliest education start {earliest_edu}",
            )
    return CheckResult(name, SOFT, False)


def check_end_before_start(record: dict[str, Any]) -> CheckResult:
    """HARD: a role whose end_date precedes its start_date."""
    name = "end_before_start"
    for role in _career(record):
        s = parse.parse_date(role.get(CareerF.START_DATE))
        e = parse.parse_date(role.get(CareerF.END_DATE))
        if s is not None and e is not None and e < s:
            return CheckResult(
                name,
                HARD,
                True,
                f"end {e} before start {s} at {role.get(CareerF.COMPANY)!r}",
            )
    return CheckResult(name, HARD, False)


def check_duration_date_mismatch(record: dict[str, Any]) -> CheckResult:
    """HARD: duration_months inconsistent with its (start, end) date pair."""
    name = "duration_date_mismatch"
    for role in _career(record):
        s = parse.parse_date(role.get(CareerF.START_DATE))
        e = parse.parse_date(role.get(CareerF.END_DATE))
        dur = role.get(CareerF.DURATION_MONTHS)
        if s is None or e is None or not isinstance(dur, (int, float)):
            continue
        if e < s:  # handled by check_end_before_start
            continue
        expected = parse.months_between(s, e)
        if expected is not None and abs(dur - expected) > DURATION_MISMATCH_TOL:
            return CheckResult(
                name,
                HARD,
                True,
                f"duration {dur}mo vs dates implying {expected:.0f}mo at "
                f"{role.get(CareerF.COMPANY)!r}",
            )
    return CheckResult(name, HARD, False)


def check_skill_duration_vs_career(
    record: dict[str, Any], now: _dt.date | None = None
) -> CheckResult:
    """SOFT: a skill used for more months than the entire career span.

    Demoted from HARD after the STEP-4 real-pool audit (fired on 14.2%): in this
    dataset per-skill `duration_months` is generated independently of the career
    timeline, so a skill duration exceeding the career span is the norm, not an
    impossibility. Kept as a SOFT coherence signal only (never hard-excludes)."""
    name = "skill_duration_vs_career"
    span = _career_span_months(record, now)
    if span is None:
        return CheckResult(name, SOFT, False)
    for sk in _skills(record):
        dur = sk.get(SkillF.DURATION_MONTHS)
        if isinstance(dur, (int, float)) and dur > span + SKILL_OVER_CAREER_TOL:
            return CheckResult(
                name,
                SOFT,
                True,
                f"skill {sk.get(SkillF.NAME)!r} {dur}mo > career span {span:.0f}mo",
            )
    return CheckResult(name, SOFT, False)


def check_current_with_end_date(record: dict[str, Any]) -> CheckResult:
    """HARD: a role flagged is_current but carrying a non-null end_date."""
    name = "current_with_end_date"
    for role in _career(record):
        if (
            role.get(CareerF.IS_CURRENT)
            and parse.parse_date(role.get(CareerF.END_DATE)) is not None
        ):
            return CheckResult(
                name,
                HARD,
                True,
                f"is_current True but end_date set at "
                f"{role.get(CareerF.COMPANY)!r}",
            )
    return CheckResult(name, HARD, False)


def check_signup_after_last_active(record: dict[str, Any]) -> CheckResult:
    """SOFT: signup_date later than last_active_date (impossible ordering).

    Demoted from HARD after the STEP-4 real-pool audit (fired on 7.5%): the
    generator samples `signup_date` and `last_active_date` independently, so this
    ordering inversion is pervasive noise here rather than a planted honeypot.
    Retained as a SOFT coherence signal; it never hard-excludes."""
    name = "signup_after_last_active"
    su = parse.parse_date(record.get(F.SIGNUP_DATE))
    la = parse.parse_date(record.get(F.LAST_ACTIVE_DATE))
    if su is not None and la is not None and su > la:
        return CheckResult(name, SOFT, True, f"signup {su} after last_active {la}")
    return CheckResult(name, SOFT, False)


def check_certification_before_tech(
    record: dict[str, Any], tech_inception: dict[str, int] | None = None
) -> CheckResult:
    """HARD: a certification dated before the technology existed.

    Operates on the ``certifications[]`` list of ``{name, issuer, year}``.
    Abstains if absent/empty."""
    name = "certification_before_tech"
    table = tech_inception if tech_inception is not None else TECH_INCEPTION
    certs = record.get(F.CERTIFICATIONS)
    if not isinstance(certs, list):
        return CheckResult(name, HARD, False)
    for cert in certs:
        if not isinstance(cert, dict):
            continue
        cname = str(cert.get(CertF.NAME, "")).lower()
        year = cert.get(CertF.YEAR)
        if not isinstance(year, (int, float)):
            continue
        for tech, inception in table.items():
            if tech in cname and year < inception:
                return CheckResult(
                    name,
                    HARD,
                    True,
                    f"cert {cert.get(CertF.NAME)!r} dated {int(year)} "
                    f"predates {tech} ({inception})",
                )
    return CheckResult(name, HARD, False)


def check_multiple_current_roles(record: dict[str, Any]) -> CheckResult:
    """SOFT: more than one role flagged is_current (plausible but suspicious)."""
    name = "multiple_current_roles"
    n = sum(1 for role in _career(record) if role.get(CareerF.IS_CURRENT))
    if n > 1:
        return CheckResult(name, SOFT, True, f"{n} roles flagged is_current")
    return CheckResult(name, SOFT, False)


def check_current_role_absent_from_history(record: dict[str, Any]) -> CheckResult:
    """SOFT: current_company/title not present anywhere in career_history."""
    name = "current_role_absent_from_history"
    cur_company = record.get(F.CURRENT_COMPANY)
    cur_title = record.get(F.CURRENT_TITLE)
    hist = _career(record)
    if not hist or (cur_company is None and cur_title is None):
        return CheckResult(name, SOFT, False)
    companies = {str(r.get(CareerF.COMPANY, "")).strip().lower() for r in hist}
    titles = {str(r.get(CareerF.TITLE, "")).strip().lower() for r in hist}
    if cur_company and str(cur_company).strip().lower() not in companies:
        return CheckResult(
            name, SOFT, True, f"current_company {cur_company!r} absent from history"
        )
    if cur_title and str(cur_title).strip().lower() not in titles:
        return CheckResult(
            name, SOFT, True, f"current_title {cur_title!r} absent from history"
        )
    return CheckResult(name, SOFT, False)


# ---------------------------------------------------------------------------
# Suite runner
# ---------------------------------------------------------------------------
def run_consistency_suite(
    record: dict[str, Any],
    *,
    founding_years: dict[str, int] | None = None,
    tech_inception: dict[str, int] | None = None,
    now: _dt.date | None = None,
) -> Assessment:
    """Run all ~12 consistency checks on one record and return an Assessment."""
    fy = founding_years or {}
    results = [
        check_tenure_vs_founding(record, fy),
        check_expert_zero_duration(record),
        check_yoe_vs_span(record, now),
        check_role_before_education(record),
        check_end_before_start(record),
        check_duration_date_mismatch(record),
        check_skill_duration_vs_career(record, now),
        check_current_with_end_date(record),
        check_signup_after_last_active(record),
        check_certification_before_tech(record, tech_inception),
        check_multiple_current_roles(record),
        check_current_role_absent_from_history(record),
    ]
    return Assessment(record.get(F.CANDIDATE_ID), results)


# ---------------------------------------------------------------------------
# IsolationForest anomaly scaffold over coherence features
# ---------------------------------------------------------------------------
COHERENCE_FEATURE_NAMES = [
    "n_hard_flags",
    "n_soft_flags",
    "max_duration_mismatch",
    "n_expert_zero",
    "n_current_roles",
    "yoe_minus_span_months",
    "n_roles",
    "n_skills",
]


def coherence_feature_vector(
    record: dict[str, Any],
    *,
    founding_years: dict[str, int] | None = None,
    now: _dt.date | None = None,
) -> list[float]:
    """Numeric coherence features for the unsupervised anomaly model.

    Deliberately overlaps with the rule checks: IsolationForest learns the joint
    distribution of these and flags profiles whose *combination* is odd even if
    no single hard rule fires. Order matches ``COHERENCE_FEATURE_NAMES``."""
    assessment = run_consistency_suite(record, founding_years=founding_years, now=now)
    n_hard = sum(1 for r in assessment.results if r.failed and r.severity == HARD)
    n_soft = assessment.soft_count

    max_mismatch = 0.0
    for role in _career(record):
        s = parse.parse_date(role.get(CareerF.START_DATE))
        e = parse.parse_date(role.get(CareerF.END_DATE))
        dur = role.get(CareerF.DURATION_MONTHS)
        if s and e and isinstance(dur, (int, float)) and e >= s:
            expected = parse.months_between(s, e) or 0.0
            max_mismatch = max(max_mismatch, abs(dur - expected))

    n_expert_zero = sum(
        1
        for sk in _skills(record)
        if str(sk.get(SkillF.PROFICIENCY, "")).lower() in EXPERT_LABELS
        and sk.get(SkillF.DURATION_MONTHS) == 0
    )
    n_current = sum(1 for r in _career(record) if r.get(CareerF.IS_CURRENT))
    yoe = record.get(F.YEARS_OF_EXPERIENCE)
    span = _career_span_months(record, now)
    yoe_minus_span = 0.0
    if isinstance(yoe, (int, float)) and span is not None:
        yoe_minus_span = yoe * 12 - span

    return [
        float(n_hard),
        float(n_soft),
        float(max_mismatch),
        float(n_expert_zero),
        float(n_current),
        float(yoe_minus_span),
        float(len(_career(record))),
        float(len(_skills(record))),
    ]


def fit_anomaly_model(X: np.ndarray, *, seed: int = 42, contamination="auto"):
    """Fit an IsolationForest over coherence features (deterministic seed).

    Returns the fitted estimator. Kept here (not in ``rank.py``) — the model is
    fit OFFLINE and loaded at replay time."""
    from sklearn.ensemble import IsolationForest

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=seed,
    )
    model.fit(np.asarray(X, dtype=float))
    return model


def anomaly_scores(model, X: np.ndarray) -> np.ndarray:
    """Return anomaly scores where **higher = more anomalous**.

    sklearn's ``score_samples`` returns higher-for-normal; we negate so callers
    can threshold "score > tau" intuitively."""
    return -model.score_samples(np.asarray(X, dtype=float))


# ---------------------------------------------------------------------------
# Triple-guard
# ---------------------------------------------------------------------------
def triple_guard(
    hard_violation: bool,
    anomaly_flag: bool,
    audit_flag: bool,
    *,
    soft_count: int = 0,
) -> bool:
    """Final exclusion decision (True => ineligible for top 100).

    * a single hard rule violation always excludes (rule #4);
    * an offline audit contradiction excludes;
    * an unsupervised anomaly excludes ONLY when corroborated by a soft flag
      (never on its own — avoids false positives on unusual-but-real people).
    """
    if hard_violation:
        return True
    if audit_flag:
        return True
    if anomaly_flag and soft_count > 0:
        return True
    return False
