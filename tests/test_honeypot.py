"""Tests for the honeypot consistency suite, anomaly model, and triple-guard.

Each hard check gets a crafted record that MUST fire and a clean record that
must NOT, plus an end-to-end run over the synthetic fixture."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src import honeypot as hp
from src import parse

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_candidates.jsonl"
FOUNDING = {"neostartup": 2023, "razorpay": 2014, "swiggy": 2014}


def _clean_record():
    return {
        "candidate_id": "clean",
        "years_of_experience": 5,
        "signup_date": "2019-01-01",
        "last_active_date": "2025-01-01",
        "current_company": "Razorpay",
        "current_title": "Senior AI Engineer",
        "career_history": [
            {
                "company": "Razorpay",
                "title": "Senior AI Engineer",
                "start_date": "2019-01",
                "end_date": None,
                "duration_months": 72,
                "is_current": True,
            }
        ],
        "education": [{"institution": "IIT", "start_year": 2011, "end_year": 2015}],
        "skills": [{"name": "Python", "duration_months": 60, "endorsements": 10}],
    }


def test_clean_record_no_hard_violation():
    a = hp.run_consistency_suite(_clean_record(), founding_years=FOUNDING)
    assert not a.hard_violation
    assert a.soft_count == 0


# --- individual hard checks --------------------------------------------------
def test_tenure_vs_founding_fires():
    rec = _clean_record()
    rec["career_history"][0]["company"] = "NeoStartup"
    rec["career_history"][0]["start_date"] = "2017-01"
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "tenure_vs_founding" in a.failed_checks


def test_tenure_vs_founding_abstains_without_table():
    rec = _clean_record()
    rec["career_history"][0]["company"] = "NeoStartup"
    rec["career_history"][0]["start_date"] = "2017-01"
    a = hp.run_consistency_suite(rec, founding_years={})  # no table -> abstain
    assert "tenure_vs_founding" not in a.failed_checks


def test_expert_zero_duration_fires():
    rec = _clean_record()
    rec["skills"] = [{"name": "FAISS", "duration_months": 0, "proficiency": "expert"}]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "expert_zero_duration" in a.failed_checks


def test_yoe_vs_span_fires():
    rec = _clean_record()
    rec["years_of_experience"] = 20  # 240 months but career ~ a few years
    rec["career_history"] = [
        {
            "company": "X",
            "start_date": "2022-01",
            "end_date": "2024-01",
            "duration_months": 24,
            "is_current": False,
        }
    ]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "yoe_vs_span" in a.failed_checks


def test_yoe_vs_span_allows_overlap():
    # concurrent roles: sum of durations >> span but yoe within span -> OK
    rec = _clean_record()
    rec["years_of_experience"] = 6
    rec["career_history"] = [
        {
            "company": "A",
            "start_date": "2019-01",
            "end_date": "2025-01",
            "duration_months": 72,
            "is_current": False,
        },
        {
            "company": "B",
            "start_date": "2019-01",
            "end_date": "2025-01",
            "duration_months": 72,
            "is_current": False,
        },
    ]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert "yoe_vs_span" not in a.failed_checks


def test_role_before_education_soft():
    # Demoted to SOFT (STEP 4): legitimate career-switchers work before a later
    # degree; must NOT hard-exclude. Still surfaced as a soft coherence flag.
    rec = _clean_record()
    rec["current_title"] = "Engineer"
    rec["current_company"] = "PrivateCo"
    rec["education"] = [{"institution": "IIT", "start_year": 2014, "end_year": 2018}]
    rec["career_history"] = [
        {
            "company": "PrivateCo",
            "title": "Engineer",
            "start_date": "2010-01",
            "end_date": None,
            "duration_months": 180,
            "is_current": True,
        }
    ]
    rec["years_of_experience"] = 15
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert not a.hard_violation
    assert "role_before_education" in a.failed_checks
    assert a.soft_count >= 1


def test_end_before_start_fires():
    rec = _clean_record()
    rec["career_history"] = [
        {
            "company": "X",
            "start_date": "2020-01",
            "end_date": "2018-01",
            "duration_months": 10,
            "is_current": False,
        }
    ]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "end_before_start" in a.failed_checks


def test_duration_date_mismatch_fires():
    rec = _clean_record()
    rec["career_history"] = [
        {
            "company": "X",
            "start_date": "2020-01",
            "end_date": "2021-01",
            "duration_months": 120,
            "is_current": False,
        }  # 12mo dates, claims 120
    ]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "duration_date_mismatch" in a.failed_checks


def test_skill_duration_vs_career_soft():
    # Demoted to SOFT (STEP 4): per-skill duration is generated independently of
    # tenure in this dataset, so it never hard-excludes; soft coherence flag only.
    rec = _clean_record()
    rec["years_of_experience"] = 6
    rec["career_history"] = [
        {
            "company": "Razorpay",
            "title": "Senior AI Engineer",
            "start_date": "2019-01",
            "end_date": "2025-01",
            "duration_months": 72,
            "is_current": False,
        }
    ]
    rec["skills"] = [{"name": "Python", "duration_months": 200, "endorsements": 1}]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert not a.hard_violation
    assert "skill_duration_vs_career" in a.failed_checks
    assert a.soft_count >= 1


def test_current_with_end_date_fires():
    rec = _clean_record()
    rec["career_history"][0]["end_date"] = "2023-01"  # is_current True + end set
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "current_with_end_date" in a.failed_checks


def test_signup_after_last_active_soft():
    # Demoted to SOFT (STEP 4): signup/last_active are sampled independently in
    # this dataset (pervasive noise), so it never hard-excludes; soft flag only.
    rec = _clean_record()
    rec["signup_date"] = "2025-06-01"
    rec["last_active_date"] = "2024-01-01"
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert not a.hard_violation
    assert "signup_after_last_active" in a.failed_checks
    assert a.soft_count >= 1


def test_certification_before_tech_fires():
    rec = _clean_record()
    rec["certifications"] = [{"name": "LangChain Expert", "year": 2015}]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert a.hard_violation
    assert "certification_before_tech" in a.failed_checks


# --- soft checks -------------------------------------------------------------
def test_multiple_current_roles_soft():
    rec = _clean_record()
    rec["career_history"] = [
        {
            "company": "A",
            "start_date": "2020-01",
            "end_date": None,
            "duration_months": 50,
            "is_current": True,
        },
        {
            "company": "B",
            "start_date": "2021-01",
            "end_date": None,
            "duration_months": 40,
            "is_current": True,
        },
    ]
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert not a.hard_violation
    assert a.soft_count >= 1
    assert "multiple_current_roles" in a.failed_checks


def test_current_role_absent_from_history_soft():
    rec = _clean_record()
    rec["current_company"] = "GhostCompany"
    a = hp.run_consistency_suite(rec, founding_years=FOUNDING)
    assert not a.hard_violation
    assert "current_role_absent_from_history" in a.failed_checks


# --- fixture sweep -----------------------------------------------------------
def test_fixture_honeypots_flagged_and_clean_ok():
    recs = {r["candidate_id"]: r for r in parse.load_pool(str(FIXTURE))}
    now = parse.reference_now_from_path(str(FIXTURE))
    flagged = {
        cid: hp.run_consistency_suite(
            rec, founding_years=FOUNDING, now=now
        ).hard_violation
        for cid, rec in recs.items()
    }
    # honeypots: 3 (tenure), 7 (end<start), 8 (current+end / expert-zero)
    assert flagged["CAND_0000003"]
    assert flagged["CAND_0000007"]
    assert flagged["CAND_0000008"]
    # clean strong + twins + sentinel profile must NOT be hard-flagged
    for cid in ["CAND_0000001", "CAND_0000004", "CAND_0000005", "CAND_0000006"]:
        assert not flagged[cid], f"candidate {cid} wrongly hard-flagged"


# --- anomaly model + triple-guard -------------------------------------------
def test_anomaly_model_flags_outlier():
    recs = parse.load_pool(str(FIXTURE))
    now = parse.reference_now_from_path(str(FIXTURE))
    X = np.array(
        [hp.coherence_feature_vector(r, founding_years=FOUNDING, now=now) for r in recs]
    )
    model = hp.fit_anomaly_model(X, seed=42)
    scores = hp.anomaly_scores(model, X)
    assert scores.shape == (len(recs),)
    assert np.all(np.isfinite(scores))


def test_triple_guard_logic():
    # single hard violation always excludes
    assert hp.triple_guard(True, False, False)
    # audit contradiction excludes
    assert hp.triple_guard(False, False, True)
    # anomaly alone does NOT exclude
    assert not hp.triple_guard(False, True, False, soft_count=0)
    # anomaly + corroborating soft flag excludes
    assert hp.triple_guard(False, True, False, soft_count=1)
    # all clean -> keep
    assert not hp.triple_guard(False, False, False)
