"""Tests for the company-type lexicon, skill ontology, and founding-year lookup."""

from __future__ import annotations

from src import lexicon as lx


def test_product_companies():
    assert lx.classify_company("Swiggy") == lx.COMPANY_PRODUCT
    assert lx.classify_company("Razorpay Pvt Ltd") == lx.COMPANY_PRODUCT
    assert (
        lx.classify_company("Flipkart Internet Private Limited") == lx.COMPANY_PRODUCT
    )
    assert lx.is_product_company("Google")


def test_services_companies():
    assert lx.classify_company("TCS") == lx.COMPANY_SERVICES
    assert lx.classify_company("Infosys Limited") == lx.COMPANY_SERVICES
    assert lx.classify_company("Cognizant Technology Solutions") == lx.COMPANY_SERVICES
    assert not lx.is_product_company("Wipro")


def test_research_before_product():
    # "Google Research" must classify as research, not product.
    assert lx.classify_company("Google Research") == lx.COMPANY_RESEARCH
    assert lx.classify_company("Microsoft Research India") == lx.COMPANY_RESEARCH


def test_unknown_company():
    assert lx.classify_company("Some Random Co") == lx.COMPANY_UNKNOWN
    assert lx.classify_company("") == lx.COMPANY_UNKNOWN
    assert lx.classify_company(None) == lx.COMPANY_UNKNOWN


def test_normalize_company():
    assert lx.normalize_company("Razorpay Software Pvt Ltd") == "razorpay"
    assert lx.normalize_company("Infosys Technologies Limited") == "infosys"


def test_skill_ontology_mapping():
    assert "vector_dbs" in lx.map_skill("FAISS")
    assert "vector_dbs" in lx.map_skill("Pinecone")
    assert "retrieval" in lx.map_skill("Recommendation Systems")
    assert "retrieval" in lx.map_skill("Search Relevance")
    assert "ltr" in lx.map_skill("Learning to Rank")
    assert "fine_tuning" in lx.map_skill("LoRA")
    assert "python" in lx.map_skill("Python")


def test_offtopic_skill_maps_to_nothing():
    assert lx.map_skill("Brand Strategy") == set()
    assert lx.map_skill("Public Speaking") == set()
    assert lx.map_skill("") == set()


def test_map_skills_counts():
    counts = lx.map_skills(["FAISS", "Pinecone", "Python", "Brand Strategy"])
    assert counts["vector_dbs"] == 2
    assert counts["python"] == 1
    assert counts["retrieval"] == 0


def test_jd_relevant_skill_count():
    assert lx.jd_relevant_skill_count(["FAISS", "Cooking", "Python"]) == 2


def test_load_founding_years_missing_is_empty():
    assert lx.load_founding_years("does_not_exist.csv") == {}


def test_load_and_lookup_founding_years(tmp_path):
    p = tmp_path / "founding_years.csv"
    p.write_text(
        "company,founding_year\nNeoStartup,2023\nRazorpay Pvt Ltd,2014\n",
        encoding="utf-8",
    )
    table = lx.load_founding_years(str(p))
    assert lx.founding_year("NeoStartup", table) == 2023
    assert lx.founding_year("Razorpay", table) == 2014
    assert lx.founding_year("UnknownCo", table) is None
