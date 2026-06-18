"""Grounded reasoning: fact-sheet -> (offline gen | composer) -> verifier -> cache.

Stage-4 penalizes templated, hallucinated, or rank-inconsistent reasoning. We
pass all six checks **by construction**:

1. Build a :class:`FactSheet` of *only literal profile values* for a candidate.
2. Generate 1-2 sentences grounded solely in the fact sheet, tone banded by
   rank (1-10 confident · 11-50 strong-with-caveat · 51-90 mixed · 91-100
   borderline). Offline this is an LLM (:func:`generate_reasoning_offline`,
   dependency-injected, no network here); at replay time the deterministic
   :func:`compose_reasoning` is the fallback.
3. :func:`verify_reasoning` rejects any number, named skill, or company that is
   not literally in the fact sheet; regenerate on failure.
4. Cache verified strings by ``candidate_id`` (``artifacts/reasoning.json``).

``rank.py`` only reads the cache and, on a miss, runs the composer fallback —
no network, no LLM.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import lexicon, parse
from .parse import F, SkillF

# Rank tone bands (1-indexed rank).
BAND_CONFIDENT = "confident"
BAND_STRONG = "strong"
BAND_MIXED = "mixed"
BAND_BORDERLINE = "borderline"

# Concrete, checkable skill/tech tokens the verifier will not allow unless they
# appear in the candidate's fact sheet. Generic cluster words (e.g. "ranking")
# are intentionally excluded to avoid false rejects on tone phrasing.
_VERIFIABLE_TOKENS: set[str] = {
    "faiss",
    "pinecone",
    "milvus",
    "weaviate",
    "qdrant",
    "chroma",
    "pgvector",
    "elasticsearch",
    "opensearch",
    "vespa",
    "scann",
    "annoy",
    "hnsw",
    "pytorch",
    "tensorflow",
    "langchain",
    "python",
    "numpy",
    "pandas",
    "lora",
    "qlora",
    "peft",
    "rlhf",
    "lambdamart",
    "ranknet",
    "xgboost",
    "kubernetes",
    "docker",
    "spark",
    "kafka",
}
# Known company tokens (verifier rejects a company name not in the sheet).
_COMPANY_TOKENS: set[str] = {
    lexicon.normalize_company(c)
    for c in (lexicon._PRODUCT_COMPANIES | lexicon._SERVICES_COMPANIES)
} - {""}

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_WORD_RE = re.compile(r"[a-z0-9+#.]+")


@dataclass
class FactSheet:
    """Literal-only profile facts a sentence may be grounded in."""

    candidate_id: Any
    years_of_experience: float | None = None
    current_title: str | None = None
    current_company: str | None = None
    skills: list[tuple[str, int | None]] = field(default_factory=list)
    concern: str | None = None
    allowed_numbers: set[str] = field(default_factory=set)
    allowed_terms: set[str] = field(default_factory=set)


def _fmt_number(value: float) -> str:
    """Format a number the way the composer will emit it (int if whole)."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)


def build_fact_sheet(
    record: dict[str, Any],
    *,
    now: _dt.date | None = None,
    max_skills: int = 3,
) -> FactSheet:
    """Extract a literal-only fact sheet for one candidate.

    Only values that physically appear in the profile are included; the
    ``allowed_numbers`` / ``allowed_terms`` sets are what the verifier checks
    generated text against."""
    cid = record.get(F.CANDIDATE_ID)
    fs = FactSheet(candidate_id=cid)

    yoe = record.get(F.YEARS_OF_EXPERIENCE)
    if isinstance(yoe, (int, float)) and not parse.is_numeric_sentinel(yoe):
        fs.years_of_experience = float(yoe)
        fs.allowed_numbers.add(_fmt_number(yoe))

    title = record.get(F.CURRENT_TITLE)
    if title:
        fs.current_title = str(title)
        fs.allowed_terms.update(_WORD_RE.findall(str(title).lower()))
        fs.allowed_numbers.update(_NUMBER_RE.findall(str(title)))

    company = record.get(F.CURRENT_COMPANY)
    if company:
        fs.current_company = str(company)
        fs.allowed_terms.update(_WORD_RE.findall(str(company).lower()))
        fs.allowed_numbers.update(_NUMBER_RE.findall(str(company)))

    # JD-relevant skills, highest-duration first, capped.
    skills = record.get(F.SKILLS) if isinstance(record.get(F.SKILLS), list) else []
    jd_skills = [s for s in skills if lexicon.map_skill(s.get(SkillF.NAME))]
    jd_skills.sort(key=lambda s: (s.get(SkillF.DURATION_MONTHS) or 0), reverse=True)
    for s in jd_skills[:max_skills]:
        name = str(s.get(SkillF.NAME))
        dur = s.get(SkillF.DURATION_MONTHS)
        dur_i = int(dur) if isinstance(dur, (int, float)) else None
        fs.skills.append((name, dur_i))
        fs.allowed_terms.update(_WORD_RE.findall(name.lower()))
        fs.allowed_numbers.update(_NUMBER_RE.findall(name))
        if dur_i is not None:
            fs.allowed_numbers.add(_fmt_number(float(dur_i)))

    fs.concern = _derive_concern(record, now)
    return fs


def _derive_concern(record: dict[str, Any], now: _dt.date | None) -> str | None:
    """One honest, literal concern (no fabricated numbers)."""
    yoe = record.get(F.YEARS_OF_EXPERIENCE)
    if isinstance(yoe, (int, float)) and yoe < 4.5:
        return "relatively early in their career for this role"
    la = parse.parse_date(record.get(F.LAST_ACTIVE_DATE))
    if la is not None and now is not None:
        months = parse.months_between(la, now)
        if months is not None and months > 6:
            return "has not been active on the platform recently"
    rr = record.get(F.RECRUITER_RESPONSE_RATE)
    if isinstance(rr, (int, float)) and rr < 0.2:
        return "has a low recruiter-response rate"
    if not lexicon.is_product_company(record.get(F.CURRENT_COMPANY)):
        return "limited product-company experience in the current role"
    return None


def tone_band(rank: int) -> str:
    """Map a 1-indexed rank to its tone band."""
    if rank <= 10:
        return BAND_CONFIDENT
    if rank <= 50:
        return BAND_STRONG
    if rank <= 90:
        return BAND_MIXED
    return BAND_BORDERLINE


_BAND_LEAD = {
    BAND_CONFIDENT: "Excellent match",
    BAND_STRONG: "Strong candidate",
    BAND_MIXED: "Reasonable match",
    BAND_BORDERLINE: "Borderline fit",
}
_BAND_USES_CONCERN = {
    BAND_CONFIDENT: False,
    BAND_STRONG: True,
    BAND_MIXED: True,
    BAND_BORDERLINE: True,
}


def compose_reasoning(fact_sheet: FactSheet, rank: int) -> str:
    """Deterministic, grounded composer (replay fallback).

    Uses only literal fact-sheet values so it passes :func:`verify_reasoning`
    by construction. Output is a single CSV-safe line (no newlines)."""
    band = tone_band(rank)
    lead = _BAND_LEAD[band]

    clauses: list[str] = []
    role_bits = []
    if fact_sheet.current_title:
        role_bits.append(fact_sheet.current_title)
    if fact_sheet.current_company:
        role_bits.append(f"at {fact_sheet.current_company}")
    role = " ".join(role_bits)

    if role and fact_sheet.years_of_experience is not None:
        clauses.append(
            f"{lead}: {role} with {_fmt_number(fact_sheet.years_of_experience)} "
            f"years of experience"
        )
    elif role:
        clauses.append(f"{lead}: {role}")
    elif fact_sheet.years_of_experience is not None:
        clauses.append(
            f"{lead}: {_fmt_number(fact_sheet.years_of_experience)} years of experience"
        )
    else:
        clauses.append(f"{lead} for the role")

    if fact_sheet.skills:
        names = ", ".join(name for name, _ in fact_sheet.skills)
        clauses.append(f"with demonstrated experience in {names}")

    sentence = ", ".join(clauses) + "."

    if _BAND_USES_CONCERN[band] and fact_sheet.concern:
        sentence += f" Note: {fact_sheet.concern}."

    return _csv_safe(sentence)


def _csv_safe(text: str) -> str:
    """Collapse whitespace and strip newlines for a single CSV field."""
    return re.sub(r"\s+", " ", text.replace("\n", " ").replace("\r", " ")).strip()


def verify_reasoning(text: str, fact_sheet: FactSheet) -> tuple[bool, list[str]]:
    """Reject any number/skill/company in ``text`` not literally in the sheet.

    Returns ``(ok, violations)``. ``ok`` is True only if every numeric token is
    in ``allowed_numbers`` and every checkable skill/company token is in
    ``allowed_terms``."""
    violations: list[str] = []
    lowered = text.lower()

    for num in _NUMBER_RE.findall(text):
        if num in fact_sheet.allowed_numbers:
            continue
        # Ignore digit substrings of an allowed decimal (e.g. "25" inside "4.25").
        if any(num != a and num in a for a in fact_sheet.allowed_numbers):
            continue
        violations.append(f"number {num!r} not in fact sheet")

    words = set(_WORD_RE.findall(lowered))
    for token in words:
        if token in _VERIFIABLE_TOKENS or token in _COMPANY_TOKENS:
            if token not in fact_sheet.allowed_terms:
                violations.append(f"term {token!r} not in fact sheet")

    if "\n" in text or "\r" in text:
        violations.append("embedded newline")

    return (len(violations) == 0, violations)


def generate_reasoning_offline(
    fact_sheet: FactSheet,
    rank: int,
    *,
    client: Callable[[str], str] | None = None,
) -> str | None:
    """Optional LLM client for offline reasoning generation."""
    if client is None:
        return None
    prompt = build_prompt(fact_sheet, rank)
    return _csv_safe(client(prompt))


def build_prompt(fact_sheet: FactSheet, rank: int) -> str:
    """Build the grounded-generation prompt (literal facts + tone instruction)."""
    band = tone_band(rank)
    facts = {
        "years_of_experience": fact_sheet.years_of_experience,
        "current_title": fact_sheet.current_title,
        "current_company": fact_sheet.current_company,
        "skills": fact_sheet.skills,
        "concern": fact_sheet.concern,
    }
    return (
        "Write 1-2 sentences explaining this candidate's fit for a Senior AI "
        f"Engineer role. Tone: {band}. Use ONLY these literal facts; do not "
        "invent any number, skill, or company.\n"
        f"FACTS: {json.dumps(facts, default=str)}"
    )


def generate_and_verify(
    fact_sheet: FactSheet,
    rank: int,
    *,
    client: Callable[[str], str] | None = None,
    max_attempts: int = 3,
) -> str:
    """Generate (offline LLM if available), verify, regenerate on failure.

    Always returns a verified string: the composer fallback is guaranteed to
    pass verification, so this never raises."""
    for _ in range(max_attempts):
        candidate = generate_reasoning_offline(fact_sheet, rank, client=client)
        if candidate is None:
            break
        ok, _violations = verify_reasoning(candidate, fact_sheet)
        if ok:
            return candidate
    # deterministic, verified fallback
    composed = compose_reasoning(fact_sheet, rank)
    ok, violations = verify_reasoning(composed, fact_sheet)
    assert ok, f"composer produced unverifiable text: {violations}"
    return composed


# ---------------------------------------------------------------------------
# Cache (artifacts/reasoning.json), keyed by candidate_id
# ---------------------------------------------------------------------------
def save_reasoning_cache(mapping: dict[Any, str], path: str) -> None:
    """Persist ``{candidate_id: reasoning}`` to JSON (keys coerced to str)."""
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(
            {str(k): v for k, v in mapping.items()}, fh, ensure_ascii=False, indent=2
        )


def load_reasoning_cache(path: str) -> dict[str, str]:
    """Load the reasoning cache, or ``{}`` if absent."""
    import os

    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def reasoning_for(
    record: dict[str, Any],
    rank: int,
    cache: dict[str, str],
    *,
    now: _dt.date | None = None,
) -> str:
    """Return cached reasoning for a candidate, else compose a verified fallback.

    Used by ``rank.py``: cache hit is the offline-verified string; on a miss
    (e.g. the <=100-candidate sandbox demo) the deterministic composer runs."""
    cid = str(record.get(F.CANDIDATE_ID))
    if cid in cache and cache[cid]:
        return _csv_safe(cache[cid])
    fs = build_fact_sheet(record, now=now)
    return compose_reasoning(fs, rank)
