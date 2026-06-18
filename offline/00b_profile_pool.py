"""OFFLINE — Pool profiling for STEP 2 (founding-year table + sample profiling).

Streams ``candidates.jsonl`` once (orjson, line-by-line — never materialized) and
enumerates the distributions we need to (a) seed/curate ``founding_years.csv``,
(b) extend the company-type lexicon + skill ontology against real data, and
(c) sanity-check honeypot/sentinel rates before the rules baseline.

Outputs (under ``artifacts/``):
  * ``company_counts.csv``       — every distinct company + frequency (current + history)
  * ``industry_counts.csv``      — distinct ``current_industry`` + frequency
  * ``skill_counts.csv``         — distinct skill names + frequency
  * ``profiling_notes.md``       — human-readable summary (top companies, sentinel
                                   rates, quick honeypot-signal counts, lexicon coverage)

No network / GPU / LLM. Reference "now" = max(last_active_date) (reported only).
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import lexicon, parse  # noqa: E402
from src.parse import CareerF, F, SkillF  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [profile] %(message)s")
log = logging.getLogger(__name__)


def _norm_city(loc: str) -> str:
    return str(loc or "").split(",")[0].strip().lower()


def profile(candidates_path: str, artifacts_dir: str) -> dict:
    os.makedirs(artifacts_dir, exist_ok=True)

    company_counts: Counter = Counter()
    current_company_counts: Counter = Counter()
    industry_counts: Counter = Counter()
    skill_counts: Counter = Counter()
    city_counts: Counter = Counter()
    country_counts: Counter = Counter()
    size_counts: Counter = Counter()
    proficiency_counts: Counter = Counter()
    company_type_counts: Counter = Counter()

    n = 0
    max_last_active = None
    # cheap honeypot/sentinel signal tallies
    n_expert_zero = 0
    n_no_github = 0
    n_no_offers = 0
    n_empty_assess = 0
    n_null_grade = 0
    n_unknown_tier = 0
    yoe_sum = 0.0
    yoe_n = 0

    for rec in parse.stream_candidates(candidates_path):
        n += 1
        cc = rec.get(F.CURRENT_COMPANY)
        if cc:
            current_company_counts[str(cc).strip()] += 1
            company_counts[str(cc).strip()] += 1
            company_type_counts[lexicon.classify_company(cc)] += 1
        ind = rec.get(F.CURRENT_INDUSTRY)
        if ind:
            industry_counts[str(ind).strip()] += 1
        loc = rec.get(F.LOCATION)
        if loc:
            city_counts[_norm_city(loc)] += 1
        ctry = rec.get(F.COUNTRY)
        if ctry:
            country_counts[str(ctry).strip()] += 1
        size = rec.get(F.CURRENT_COMPANY_SIZE)
        if size:
            size_counts[str(size).strip()] += 1

        la = parse.parse_date(rec.get(F.LAST_ACTIVE_DATE))
        if la is not None and (max_last_active is None or la > max_last_active):
            max_last_active = la

        career = rec.get(F.CAREER_HISTORY) or []
        for r in career:
            comp = r.get(CareerF.COMPANY)
            if comp:
                company_counts[str(comp).strip()] += 1

        for s in rec.get(F.SKILLS) or []:
            nm = s.get(SkillF.NAME)
            if nm:
                skill_counts[str(nm).strip()] += 1
            prof = str(s.get(SkillF.PROFICIENCY, "")).strip().lower()
            if prof:
                proficiency_counts[prof] += 1
            if prof in {"expert"} and s.get(SkillF.DURATION_MONTHS) == 0:
                n_expert_zero += 1

        sig_gh = rec.get(F.GITHUB_ACTIVITY_SCORE)
        if sig_gh is None or parse.is_numeric_sentinel(sig_gh):
            n_no_github += 1
        oa = rec.get(F.OFFER_ACCEPTANCE_RATE)
        if oa is None or parse.is_numeric_sentinel(oa):
            n_no_offers += 1
        assess = rec.get(F.SKILL_ASSESSMENT_SCORES)
        if not (isinstance(assess, dict) and assess):
            n_empty_assess += 1
        for e in rec.get(F.EDUCATION) or []:
            if e.get("grade") is None:
                n_null_grade += 1
            if parse.is_missing_tier(e.get("tier")):
                n_unknown_tier += 1
        yoe = rec.get(F.YEARS_OF_EXPERIENCE)
        if isinstance(yoe, (int, float)):
            yoe_sum += float(yoe)
            yoe_n += 1

        if n % 20000 == 0:
            log.info("…%d candidates", n)

    log.info("done streaming %d candidates", n)

    def _write_counts(counter: Counter, path: str, headers=("value", "count")):
        with open(path, "w", encoding="utf-8", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(headers)
            for val, cnt in counter.most_common():
                w.writerow([val, cnt])

    _write_counts(company_counts, os.path.join(artifacts_dir, "company_counts.csv"),
                  ("company", "count"))
    _write_counts(industry_counts, os.path.join(artifacts_dir, "industry_counts.csv"),
                  ("industry", "count"))
    _write_counts(skill_counts, os.path.join(artifacts_dir, "skill_counts.csv"),
                  ("skill", "count"))

    return {
        "n": n,
        "max_last_active": str(max_last_active),
        "company_counts": company_counts,
        "current_company_counts": current_company_counts,
        "industry_counts": industry_counts,
        "skill_counts": skill_counts,
        "city_counts": city_counts,
        "country_counts": country_counts,
        "size_counts": size_counts,
        "proficiency_counts": proficiency_counts,
        "company_type_counts": company_type_counts,
        "n_expert_zero": n_expert_zero,
        "n_no_github": n_no_github,
        "n_no_offers": n_no_offers,
        "n_empty_assess": n_empty_assess,
        "n_null_grade": n_null_grade,
        "n_unknown_tier": n_unknown_tier,
        "yoe_mean": (yoe_sum / yoe_n) if yoe_n else None,
    }


def write_notes(stats: dict, artifacts_dir: str, founding_path: str) -> None:
    n = stats["n"]
    founding = lexicon.load_founding_years(founding_path)
    # coverage: how many candidate current-company mentions are covered by the
    # founding table (by normalized name).
    covered = sum(
        c for name, c in stats["current_company_counts"].items()
        if lexicon.founding_year(name, founding) is not None
    )
    lines = []
    lines.append("# Pool profiling notes (STEP 2)\n")
    lines.append(f"- Candidates streamed: **{n:,}**")
    lines.append(f"- Reference now = max(last_active_date): **{stats['max_last_active']}**")
    lines.append(f"- Mean years_of_experience: **{stats['yoe_mean']:.2f}**\n")

    lines.append("## Sentinel / sparsity rates (→ NaN + has_* indicators)")
    for label, key in [
        ("no GitHub (github_activity_score == -1)", "n_no_github"),
        ("no prior offers (offer_acceptance_rate == -1)", "n_no_offers"),
        ("empty skill_assessment_scores", "n_empty_assess"),
    ]:
        v = stats[key]
        lines.append(f"- {label}: {v:,} ({100.0 * v / n:.1f}%)")
    lines.append(f"- null education grade (rows): {stats['n_null_grade']:,}")
    lines.append(f"- unknown education tier (rows): {stats['n_unknown_tier']:,}")
    lines.append(f"- skills that are expert + 0 months (honeypot tell, rows): "
                 f"{stats['n_expert_zero']:,}\n")

    lines.append("## Company-type coverage (current employer, by lexicon)")
    for t, c in stats["company_type_counts"].most_common():
        lines.append(f"- {t}: {c:,} ({100.0 * c / n:.1f}%)")
    lines.append(f"- founding_years.csv covers **{covered:,}** current-company "
                 f"mentions ({100.0 * covered / n:.1f}% of pool)\n")

    lines.append("## Top 60 companies (current + history mentions)")
    for name, c in stats["company_counts"].most_common(60):
        ftype = lexicon.classify_company(name)
        fy = lexicon.founding_year(name, founding)
        fy_s = f"founded {fy}" if fy else "—"
        lines.append(f"- {name} ({c:,}) [{ftype}] {fy_s}")
    lines.append("")

    lines.append("## Top 40 current industries")
    for name, c in stats["industry_counts"].most_common(40):
        lines.append(f"- {name}: {c:,}")
    lines.append("")

    lines.append("## Top 60 skills (raw names)")
    for name, c in stats["skill_counts"].most_common(60):
        clusters = ",".join(sorted(lexicon.map_skill(name))) or "—"
        lines.append(f"- {name}: {c:,} [{clusters}]")
    lines.append("")

    lines.append("## Locations / countries / sizes")
    lines.append("Top 25 cities: " + ", ".join(
        f"{k}({v})" for k, v in stats["city_counts"].most_common(25)))
    lines.append("Countries: " + ", ".join(
        f"{k}({v})" for k, v in stats["country_counts"].most_common(15)))
    lines.append("Company sizes: " + ", ".join(
        f"{k}({v})" for k, v in stats["size_counts"].most_common()))
    lines.append("Proficiency: " + ", ".join(
        f"{k}({v})" for k, v in stats["proficiency_counts"].most_common()))
    lines.append("")

    with open(os.path.join(artifacts_dir, "profiling_notes.md"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    log.info("wrote profiling_notes.md (founding coverage %.1f%% of pool)",
             100.0 * covered / n)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=os.path.join("data", "candidates.jsonl"))
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()
    if not os.path.exists(args.candidates):
        log.warning("🛑 candidates file not found: %s", args.candidates)
        return 0
    stats = profile(args.candidates, args.artifacts)
    write_notes(stats, args.artifacts,
                os.path.join(args.artifacts, "founding_years.csv"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
