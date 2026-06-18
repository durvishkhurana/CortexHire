"""Company-type lexicon, skill ontology, and founding-year lookup.

What each piece defends against (the JD's traps)
------------------------------------------------
* **Company-type lexicon** — the "career-context filter" layer: product vs
  services vs research, India-market aware. Defeats the "services-only career"
  disqualifier and rewards real product-company production experience.
* **Skill ontology** — maps raw skill strings to the JD's intent clusters with
  *buzzword-free synonyms* ("recommendation systems", "search relevance" ->
  retrieval). Lets the model reward a Swiggy recsys engineer who never wrote
  "RAG" (the Plain-Language Tier 5 trap), and stops keyword stuffers from
  scoring on skill *names* alone.
* **Founding-year lookup** — enables the flagship honeypot check ("8 years at a
  3-year-old company"). The table itself is curated offline from the sample
  data (``artifacts/founding_years.csv``); this module provides the loader with
  a safe empty default so the code runs before that artifact exists.

All names below are real, hand-curated lexicons (India-dominated). They are
extended offline from the distinct ``company`` / ``skills`` values in the pool.
"""

from __future__ import annotations

import csv
import os
import re
from functools import lru_cache

# ---------------------------------------------------------------------------
# Company-type lexicon
# ---------------------------------------------------------------------------
COMPANY_PRODUCT = "product"
COMPANY_SERVICES = "services"
COMPANY_RESEARCH = "research"
COMPANY_UNKNOWN = "unknown"

# India-market-aware product companies (consumer + B2B SaaS + global product).
_PRODUCT_COMPANIES: set[str] = {
    # India consumer / fintech / commerce
    "swiggy",
    "zomato",
    "razorpay",
    "flipkart",
    "ola",
    "olacabs",
    "paytm",
    "phonepe",
    "cred",
    "meesho",
    "myntra",
    "dunzo",
    "sharechat",
    "nykaa",
    "bigbasket",
    "urbancompany",
    "groww",
    "zerodha",
    "unacademy",
    "byjus",
    "dream11",
    "games24x7",
    "policybazaar",
    "freshworks",
    "zoho",
    "postman",
    "browserstack",
    "chargebee",
    "hasura",
    "druva",
    "innovaccer",
    "gupshup",
    "delhivery",
    "rapido",
    "blinkit",
    "zepto",
    "navi",
    "slice",
    "jupiter",
    # Global product / big tech
    "google",
    "microsoft",
    "amazon",
    "meta",
    "facebook",
    "apple",
    "netflix",
    "uber",
    "airbnb",
    "linkedin",
    "stripe",
    "adobe",
    "atlassian",
    "salesforce",
    "nvidia",
    "intel",
    "qualcomm",
    "twitter",
    "snap",
    "pinterest",
    "spotify",
    "walmartlabs",
    "walmart global tech",
    "expedia",
    "booking",
    "shopify",
    "databricks",
    "snowflake",
    "confluent",
    "elastic",
    "mongodb",
    # --- additional India product / startups present in the real pool ---
    "inmobi",
    "vedantu",
    "byju's",
    "upgrad",
    "pharmeasy",
    "glance",
    "rephrase.ai",
    "sarvam ai",
    "aganitha",
    "niramai",
    "saarthi.ai",
    "krutrim",
    "wysa",
    "mad street den",
    "haptik",
    "verloop.io",
    "observe.ai",
    "yellow.ai",
    "locobuzz",
}

# IT-services / consulting (the "services-only career" disqualifier context).
_SERVICES_COMPANIES: set[str] = {
    "tcs",
    "tata consultancy services",
    "infosys",
    "wipro",
    "cognizant",
    "capgemini",
    "accenture",
    "hcl",
    "hcltech",
    "tech mahindra",
    "mindtree",
    "ltimindtree",
    "lti",
    "larsen toubro infotech",
    "mphasis",
    "hexaware",
    "dxc",
    "dxc technology",
    "birlasoft",
    "coforge",
    "zensar",
    "persistent systems",
    "cybage",
    "nttdata",
    "ntt data",
    "atos",
    "sopra",
    "igate",
    "syntel",
    "virtusa",
    "happiest minds",
    "sonata software",
    "kpit",
    "l&t infotech",
    "wns",
    "genpact",
    "ibm services",
}

# Research labs / institutions.
_RESEARCH_COMPANIES: set[str] = {
    "google research",
    "deepmind",
    "google deepmind",
    "microsoft research",
    "ibm research",
    "adobe research",
    "openai",
    "anthropic",
    "fair",
    "meta ai",
    "allen institute",
    "ai2",
    "nvidia research",
    "amazon science",
    "iisc",
    "indian institute of science",
    "isro",
    "drdo",
    "cdac",
    "iit",
    "iiit",
    "tifr",
    "raman research institute",
}

# Corporate suffixes stripped during normalization.
_SUFFIXES = (
    "private limited",
    "pvt ltd",
    "pvt. ltd.",
    "pvt",
    "limited",
    "ltd",
    "inc",
    "incorporated",
    "llc",
    "llp",
    "corp",
    "corporation",
    "co",
    "technologies",
    "technology",
    "labs",
    "lab",
    "solutions",
    "services",
    "software",
    "systems",
    "global",
    "india",
    "consulting",
)


@lru_cache(maxsize=200_000)
def normalize_company(name: str | None) -> str:
    """Lowercase, strip punctuation and common corporate suffixes.

    Cached: the same few thousand company strings recur across the 100K pool, so
    memoizing turns the per-role normalization into a dict hit on the hot path.
    """
    if not name:
        return ""
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9& ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # strip trailing suffix tokens repeatedly
    changed = True
    while changed:
        changed = False
        for suf in _SUFFIXES:
            if s.endswith(" " + suf):
                s = s[: -(len(suf) + 1)].strip()
                changed = True
    return s


# Precomputed normalized key sets (so multi-word/suffixed names match).
_PRODUCT_KEYS = {normalize_company(c) for c in _PRODUCT_COMPANIES} - {""}
_SERVICES_KEYS = {normalize_company(c) for c in _SERVICES_COMPANIES} - {""}


def _compile_word_alternation(keys: set[str]) -> re.Pattern[str]:
    """One whole-word alternation regex for a whole key set.

    Replaces N per-key ``re.search`` calls (each previously re-escaping and
    re-compiling) with a SINGLE precompiled search. Longest keys first so the
    alternation prefers the most specific phrase; word boundaries use
    alphanumeric look-arounds so "ola" never matches "solar".
    """
    if not keys:
        return re.compile(r"(?!x)x")  # never matches
    alt = "|".join(re.escape(k) for k in sorted(keys, key=len, reverse=True))
    return re.compile(rf"(?<![a-z0-9])(?:{alt})(?![a-z0-9])")


_PRODUCT_RE = _compile_word_alternation(_PRODUCT_KEYS)
_SERVICES_RE = _compile_word_alternation(_SERVICES_KEYS)


@lru_cache(maxsize=200_000)
def classify_company(name: str | None) -> str:
    """Return the company-type label for a raw company name.

    One of ``product`` / ``services`` / ``research`` / ``unknown``. Research is
    checked before product so "google research" != "google". Matching is
    whole-word against curated keys, so "Flipkart Internet Pvt Ltd" -> product.
    Cached + single-regex matching: the dominant ``rank.py`` hot path.
    """
    norm = normalize_company(name)
    if not norm:
        return COMPANY_UNKNOWN
    # research first (more specific names); check raw lowercase since suffix
    # stripping can remove the word "research".
    raw = (name or "").lower()
    for r in _RESEARCH_COMPANIES:
        if r in raw:
            return COMPANY_RESEARCH
    if _PRODUCT_RE.search(norm):
        return COMPANY_PRODUCT
    if _SERVICES_RE.search(norm):
        return COMPANY_SERVICES
    return COMPANY_UNKNOWN


def is_product_company(name: str | None) -> bool:
    """True iff the company is a known product company."""
    return classify_company(name) == COMPANY_PRODUCT


# ---------------------------------------------------------------------------
# Skill ontology -> JD intent clusters
# ---------------------------------------------------------------------------
# JD clusters: retrieval, vector_dbs, ranking_eval, python, ltr, fine_tuning.
# Values are buzzword-free synonyms matched as substrings of the normalized
# skill string (so "search relevance engineering" -> retrieval).
SKILL_ONTOLOGY: dict[str, set[str]] = {
    "retrieval": {
        "retrieval",
        "semantic search",
        "search relevance",
        "information retrieval",
        "rag",
        "recommendation system",
        "recommender",
        "recsys",
        "recommendation engine",
        "neural search",
        "dense retrieval",
        "hybrid search",
        "nearest neighbor",
        "similarity search",
        "embeddings",
        "embedding",
        "search engine",
        "personalization",
        "candidate generation",
        # real pool names:
        "vector search",
        "sentence transformers",
        "bm25",
        "search infrastructure",
        "search backend",
        "search & discovery",
        "search and discovery",
        "vector representations",
        "ranking systems",
    },
    "vector_dbs": {
        "faiss",
        "pinecone",
        "milvus",
        "weaviate",
        "qdrant",
        "chroma",
        "vector database",
        "vector db",
        "pgvector",
        "annoy",
        "hnsw",
        "scann",
        "vector store",
        "vector index",
        "elasticsearch",
        "opensearch",
        "vespa",
    },
    "ranking_eval": {
        "ranking",
        "ndcg",
        "mean average precision",
        "mrr",
        "ranking metric",
        "relevance evaluation",
        "evaluation framework",
        "offline evaluation",
        "ab testing",
        "a/b testing",
        "search quality",
        "ranking evaluation",
        "precision recall",
    },
    "ltr": {
        "learning to rank",
        "learning-to-rank",
        "ltr",
        "lambdamart",
        "lambdarank",
        "ranknet",
        "gradient boosted ranking",
        "xgboost ranking",
        "listwise",
        "pairwise ranking",
    },
    "python": {
        "python",
        "numpy",
        "pandas",
        "scipy",
        "pytorch",
        "tensorflow",
        "scikit-learn",
        "sklearn",
    },
    "fine_tuning": {
        "fine-tuning",
        "fine tuning",
        "finetuning",
        "lora",
        "qlora",
        "peft",
        "sft",
        "rlhf",
        "instruction tuning",
        "transfer learning",
        "model training",
        "distillation",
        "model finetuning",
        # real pool names:
        "fine-tuning llms",
        "hugging face",
        "transformers",
        "llm",
        "llms",
    },
    "nlp": {
        "nlp",
        "natural language processing",
        "text classification",
        "named entity recognition",
        "question answering",
        "language model",
    },
}

# CV / speech / robotics skills — a JD **down-rank** signal ("primary expertise is
# computer vision, speech, or robotics WITHOUT significant NLP/IR exposure").
# These are NOT JD-positive clusters; they feed a negative feature.
CV_SPEECH_ROBOTICS_SKILLS: set[str] = {
    "computer vision",
    "image classification",
    "object detection",
    "image segmentation",
    "segmentation",
    "pose estimation",
    "ocr",
    "face recognition",
    "speech recognition",
    "asr",
    "tts",
    "text to speech",
    "text-to-speech",
    "speech synthesis",
    "voice",
    "robotics",
    "slam",
    "lidar",
}


@lru_cache(maxsize=200_000)
def normalize_skill(skill: str | None) -> str:
    """Lowercase and collapse whitespace/punctuation for matching."""
    if not skill:
        return ""
    s = skill.lower().strip()
    s = re.sub(r"[_/]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _phrase_hit(norm: str, tokens: set[str], syn: str) -> bool:
    """Match a synonym against a normalized skill string.

    Long synonyms (>= 5 chars) match as substrings (so "recommendation system"
    hits "recommendation systems"); short synonyms (< 5 chars, e.g. "rag",
    "bm25", "asr", "nlp", "ltr") must match a WHOLE token, so "rag" does not
    falsely fire on "storage" and "ltr" not on "filter"."""
    if len(syn) >= 5:
        return syn in norm
    return syn in tokens


@lru_cache(maxsize=200_000)
def map_skill(skill: str | None) -> frozenset[str]:
    """Map a single raw skill to the set of JD clusters it belongs to.

    A skill can map to multiple clusters (e.g. "FAISS" -> vector_dbs). Returns
    an empty set for off-topic skills (the keyword-stuffer's irrelevant ones).
    Cached + returns a ``frozenset`` so the memoized value is never mutated.
    """
    norm = normalize_skill(skill)
    if not norm:
        return frozenset()
    tokens = frozenset(norm.split())
    clusters: set[str] = set()
    for cluster, synonyms in SKILL_ONTOLOGY.items():
        for syn in synonyms:
            if _phrase_hit(norm, tokens, syn):
                clusters.add(cluster)
                break
    return frozenset(clusters)


@lru_cache(maxsize=200_000)
def is_cv_speech_robotics(skill: str | None) -> bool:
    """True iff the skill is a computer-vision / speech / robotics skill.

    JD down-rank signal when not paired with NLP/IR exposure."""
    norm = normalize_skill(skill)
    if not norm:
        return False
    tokens = frozenset(norm.split())
    return any(_phrase_hit(norm, tokens, s) for s in CV_SPEECH_ROBOTICS_SKILLS)


def cv_speech_robotics_skill_count(skills: list[str | None]) -> int:
    """Number of skills that are computer-vision / speech / robotics."""
    return sum(1 for sk in skills if is_cv_speech_robotics(sk))


def map_skills(skills: list[str | None]) -> dict[str, int]:
    """Count how many of the given raw skills hit each JD cluster."""
    counts: dict[str, int] = {c: 0 for c in SKILL_ONTOLOGY}
    for sk in skills:
        for cluster in map_skill(sk):
            counts[cluster] += 1
    return counts


def jd_relevant_skill_count(skills: list[str | None]) -> int:
    """Number of skills that map to at least one JD cluster."""
    return sum(1 for sk in skills if map_skill(sk))


# ---------------------------------------------------------------------------
# Founding-year lookup (table built offline)
# ---------------------------------------------------------------------------
_DEFAULT_FOUNDING_PATH = os.path.join("artifacts", "founding_years.csv")


def load_founding_years(path: str = _DEFAULT_FOUNDING_PATH) -> dict[str, int]:
    """Load the founding-year table as ``{normalized_company: year}``.

    Safe empty default: returns ``{}`` if the file is missing (it is a
    documented TODO produced by ``offline/00``/``offline/02`` from the sample
    data). Expected CSV columns: ``company,founding_year``.
    """
    table: dict[str, int] = {}
    if not os.path.exists(path):
        return table
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            name = row.get("company")
            year = row.get("founding_year")
            if not name or not year:
                continue
            try:
                table[normalize_company(name)] = int(str(year).strip())
            except (ValueError, TypeError):
                continue
    return table


def founding_year(name: str | None, table: dict[str, int]) -> int | None:
    """Look up a company's founding year from a loaded table (None if unknown)."""
    norm = normalize_company(name)
    if not norm:
        return None
    return table.get(norm)
