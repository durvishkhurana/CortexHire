"""Tests for fact-sheet extraction, composer, verifier, tone bands, and cache."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from src import parse
from src import reasoning as rz

FIXTURE = Path(__file__).parent / "fixtures" / "synthetic_candidates.jsonl"
NOW = dt.date(2026, 6, 1)
C1 = "CAND_0000001"
C2 = "CAND_0000002"
C6 = "CAND_0000006"


def _records():
    return {r["candidate_id"]: r for r in parse.load_pool(str(FIXTURE))}


def test_tone_bands():
    assert rz.tone_band(1) == rz.BAND_CONFIDENT
    assert rz.tone_band(10) == rz.BAND_CONFIDENT
    assert rz.tone_band(11) == rz.BAND_STRONG
    assert rz.tone_band(50) == rz.BAND_STRONG
    assert rz.tone_band(51) == rz.BAND_MIXED
    assert rz.tone_band(90) == rz.BAND_MIXED
    assert rz.tone_band(91) == rz.BAND_BORDERLINE
    assert rz.tone_band(100) == rz.BAND_BORDERLINE


def test_fact_sheet_literal_only():
    fs = rz.build_fact_sheet(_records()[C1], now=NOW)
    assert fs.years_of_experience == 7.0
    assert fs.current_company == "Razorpay"
    assert "7" in fs.allowed_numbers
    assert "razorpay" in fs.allowed_terms
    assert any("python" in name.lower() for name, _ in fs.skills)


def test_composer_passes_verifier_for_all_candidates():
    for cid, rec in _records().items():
        fs = rz.build_fact_sheet(rec, now=NOW)
        for rank in (1, 25, 75, 99):
            text = rz.compose_reasoning(fs, rank)
            ok, violations = rz.verify_reasoning(text, fs)
            assert ok, f"candidate {cid} rank {rank}: {violations}"
            assert "\n" not in text and "\r" not in text


def test_composer_tone_changes_with_rank():
    fs = rz.build_fact_sheet(_records()[C1], now=NOW)
    assert "Excellent match" in rz.compose_reasoning(fs, 1)
    assert "Borderline fit" in rz.compose_reasoning(fs, 95)


def test_verifier_rejects_hallucinated_skill():
    fs = rz.build_fact_sheet(_records()[C6], now=NOW)  # candidate without FAISS
    bad = "Strong candidate with demonstrated experience in FAISS and Pinecone."
    ok, violations = rz.verify_reasoning(bad, fs)
    assert not ok
    assert any("faiss" in v for v in violations)


def test_verifier_rejects_hallucinated_number():
    fs = rz.build_fact_sheet(_records()[C1], now=NOW)
    bad = "Strong candidate with 42 years of experience."
    ok, violations = rz.verify_reasoning(bad, fs)
    assert not ok
    assert any("42" in v for v in violations)


def test_verifier_rejects_wrong_company():
    fs = rz.build_fact_sheet(_records()[C1], now=NOW)  # Razorpay
    bad = "Strong candidate at Infosys with 7 years of experience."
    ok, violations = rz.verify_reasoning(bad, fs)
    assert not ok
    assert any("infosys" in v for v in violations)


def test_generate_and_verify_uses_client_when_valid():
    rec = _records()[C1]
    fs = rz.build_fact_sheet(rec, now=NOW)

    def good_client(prompt: str) -> str:
        return "Excellent match: Senior AI Engineer at Razorpay with 7 years of experience."

    out = rz.generate_and_verify(fs, 1, client=good_client)
    assert "Razorpay" in out


def test_generate_and_verify_falls_back_when_client_hallucinates():
    rec = _records()[C1]
    fs = rz.build_fact_sheet(rec, now=NOW)

    def bad_client(prompt: str) -> str:
        return "Amazing engineer with 99 years at Google and Kubernetes mastery."

    out = rz.generate_and_verify(fs, 1, client=bad_client, max_attempts=2)
    # falls back to the verified composer (no hallucinated tokens)
    ok, violations = rz.verify_reasoning(out, fs)
    assert ok, violations
    assert "99" not in out


def test_cache_roundtrip_and_reasoning_for(tmp_path):
    recs = _records()
    cache_path = tmp_path / "reasoning.json"
    mapping = {C1: "Cached grounded sentence for candidate 1."}
    rz.save_reasoning_cache(mapping, str(cache_path))
    loaded = rz.load_reasoning_cache(str(cache_path))
    assert loaded[C1].startswith("Cached")

    # cache hit
    assert rz.reasoning_for(recs[C1], 1, loaded, now=NOW).startswith("Cached")
    # cache miss -> composer fallback
    out = rz.reasoning_for(recs[C2], 5, loaded, now=NOW)
    assert out and "\n" not in out


def test_load_cache_missing_is_empty():
    assert rz.load_reasoning_cache("does_not_exist.json") == {}
