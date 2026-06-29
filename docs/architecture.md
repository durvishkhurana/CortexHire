# ARCHITECTURE — "Offline Intelligence, Honest Replay"
### Intelligent Candidate Discovery & Ranking (Redrob hackathon)

This is the engineering design for *what we build*. Official rules: [`CHALLENGE_BRIEF.md`](./CHALLENGE_BRIEF.md). Offline scripts: `offline/01`–`06`, `scripts/run_pipeline_no_api.py`. Rationale: [`decisions.md`](./decisions.md). **Benchmarks:** [`RESULTS.md`](./RESULTS.md).

---

## 0. Design thesis

**The problem is 100% precomputable.** One fixed JD, 100,000 fixed candidates, no new data at ranking time ⇒ every score is an offline quantity. `rank.py` is a **deterministic replay**, not an inference engine.

Three layers, all learned where possible:
1. **Skill/experience fit** — JD *semantics*, not keyword overlap.
2. **Career-context filter** — product vs services, real production AI vs framework tutorials.
3. **Behavioral modulation** — availability/engagement, **learned** (not a hand-tuned multiplier).

Two phases:
- **OFFLINE** (no time limit; GPU + network allowed): all intelligence.
- **REPLAY** `rank.py` (≤5 min, CPU, no network): parse → join → predict → sort → reason → CSV.

---

## 1. The scoring target (design backwards from this)

```
Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
Tiebreaks: P@5 → P@10 → earlier submission timestamp
```

Implications baked into the design:
- **The top 10 is 50% of the score and you get one blind shot.** Spend disproportionate offline effort there; reduce its variance (seed ensembles; uncertainty penalty inside the top-20 band).
- **P@10 counts tier 3+; honeypots are tier 0.** A single honeypot in the top 10 hits NDCG, MAP, *and* P@10 simultaneously — and >10 in the top 100 is **instant disqualification**.
- **Relevance is graded on ~0–5** (honeypots=0, P@10 boundary=3+, planted "Tier 5s"). Confirm exact scale from `submission_spec.docx`.

---

## 2. The traps and the layer that defeats each

| Trap | What it looks like | Defeated by |
|---|---|---|
| **Keyword stuffers** | Marketing Manager with "FAISS, Pinecone, LangChain" skills | Career-text evidence weighted over skill *names*; `claimed_unverified_ratio`; assessment-backed coverage; teacher reads full context |
| **Honeypots (~80)** | "8 yrs at a 3-yr-old company"; "expert in 10 skills, 0 months each" | **Triple-guard**: hard rules ∧ IsolationForest anomaly ∧ audit-LLM contradiction hunt; founding-year table; **single violation → ineligible** |
| **Behavioral twins** | Identical skills, opposite availability (active vs ghost) | **Learned** signal weights on the organizers' own rubric; monotone constraints |
| **Plain-Language Tier 5s** | Swiggy recsys engineer who never wrote "RAG" | Instruction-aware Qwen3 embeddings; high-recall labeling union; rules-shortlist entry path so retrieval misses still get labeled |

---

## 3. OFFLINE pipeline

### Phase 0 — Forensics & rubric reconstruction *(highest fidelity move)*
- **Mine the three docx files** — `job_description.docx` (incl. hackathon notes), `redrob_signals_doc.docx`, `submission_spec.docx` — for: the **exact tier scale**, the organizers' **rubric language**, **signal-weighting philosophy**, salary band (if any), and **whether Stage-3 Docker has network access** (if not → ship artifacts via Git LFS, keep repo < ~2 GB).
- Diff every normative sentence in those docx files against the brief summary; the teacher prompt is built from *their* words, near-verbatim.
- Profile the 50-sample (`sample_candidates.json`); enumerate distinct `company` values → seed the **founding-year table** (curate the top few hundred by frequency by hand; India-dominated).
- Label **~50 calibration anchors** spanning the spectrum, fixed to the JD's three published example judgments. *This set tunes the rubric — it does NOT compare systems.*

### Phase 1 — Feature store (`features.parquet`, ~70–90 columns)
Streaming `orjson` parse → parquet. Reference "now" = `max(last_active_date)` across the pool (never wall-clock).

- **Career-evidence:** ontology hits in `career_history[].description` (recency-weighted), product-company tenure, IC-vs-management signal, "founding-team-fit" (early-stage product experience).
- **Company-type lexicon:** product / services / startup / research, India-market aware (Swiggy/Razorpay/Flipkart/Zomato/Ola = product; TCS/Infosys/Wipro/Cognizant/Capgemini = services).
- **Skill ontology:** raw skills → JD clusters (retrieval, vector DBs, ranking/eval, Python, LTR, fine-tuning) with **buzzword-free synonyms** ("recommendation systems", "search relevance" → retrieval).
- **Verification-gap (anti-stuffer):** `claimed_unverified_ratio`, max/mean `skill_assessment_scores` over JD-relevant skills, duration-weighted JD-skill coverage, `evidence_density`.
- **All 23 behavioral signals**, sentinels → `NaN` + indicators (`has_github`, `has_prior_offers`).
- **Disqualifier detectors:** 18-month-no-IC; title-chasing (mean tenure < ~20 mo over last 3+ hops + monotone title inflation); services-only career; CV/speech-only; wrapper-only.
- **Location/YoE:** relocation compatibility tiers; soft YoE window (penalize gently outside ~4.5–11; hard only at junior extreme).
- **Honeypot:** rule-violation flags (~12 checks, below) + IsolationForest anomaly score over coherence features.

**Consistency suite (~12 checks — ship all, not 3):** tenure vs company founding year · `expert` + `duration_months=0` · Σ/span of career durations vs `years_of_experience` (allow concurrent-role overlap) · role start before plausible education completion · `end_date` < `start_date` · `duration_months` inconsistent with its date pair · skill `duration_months` > total career months · `is_current` with non-null `end_date` · `signup_date` > `last_active_date` · certification year before tech existed · multiple "current" roles (soft) · `current_company/title` absent from history (soft).

### Phase 2 — Text scores as features
- **Qwen3-Embedding-0.6B** (instruction-aware, 1024-D, Apache-2.0), **3–5 JD-intent query variants** ("what the JD means") → cosine per candidate.
- **BM25** over the same career-text document (career_history + summary + headline; **never the skills list**).
- **Tuned fusion** → one float (convex α≈0.5 of normalized scores, or RRF small-k; tune on the harness).
- **Qwen3-Reranker-0.6B/4B** score over **all 100K** (offline GPU, 1–3 GPU-hrs) → one float. *(This is the work S2 wrongly tried to do online.)*

### Phase 3 — Teacher labeling
- **High-recall union pool (12–20K unique):** top-5K dense ∪ top-5K BM25 ∪ top-5K rules-shortlist ∪ **every honeypot suspect** ∪ 1–2K stratified random. A plain-language Tier 5 missed by retrieval still enters via rules/random.
- **Teacher = frontier LLM** (or local Qwen2.5-32B / Llama-3.3-70B on Colab if budget=0). Per candidate: **tier (0–5)** + **continuous 0–100** + **one-line evidence quote**. Rubric = organizers' docx language + the three anchors + explicit inconsistency-hunting + **"ignore names and institution prestige beyond the `tier` field."**
- **Self-consistency:** double-label 300–500 in shuffled order; measure agreement; **drop low-consistency labels** (ConFit v3: quality > quantity).
- Cost: ~15K × ~1.3K tokens ≈ 20M tokens ≈ **$10–50** (≈half with batch APIs) — trivial for a ₹10L track.

### Phase 4 — Student model
- **LightGBM**, compare two heads on the pooled harness: **`lambdarank`** (one query group) vs **pointwise regression on the 0–100 score, top-weighted** (weight ∝ relevance gain). The boring pointwise model is a genuine contender (arXiv 2604.21264).
- **Monotone constraints:** `recruiter_response_rate`↑, `last_active` recency↑, `interview_completion_rate`↑, `notice_period_days`↓ (unit-test constraint support on the pinned version).
- **3–5-seed ensemble, averaged.** Shallow (≤500 trees, depth ≤6). Export feature importances.

### Phase 5 — Head refinement (audit as *training signal*, never an override table)
- Student top ~300 → **multi-pass (≥2)** LLM setwise/pairwise audit hunting contradictions and mis-orderings.
- Convert outcomes → labeled pairs / adjusted labels → **retrain** the student → verify the refit's head agrees.
- Hand-review every teacher/student disagreement inside the top-50 (gold list).
- **Risk-adjusted ordering** inside the top-20 band only: a small, deterministic, in-code uncertainty penalty prefers low-variance candidates among near-ties (NOT a hidden override). Slots 51–100 absorb more variance.

### Phase 6 — Honeypot triple-guard
`exclude if (hard_rule_violation) OR (anomaly_score > τ AND corroborated) OR (audit flags contradiction)` → **zero honeypots in top 100 by construction, margin 10.**

### Phase 7 — Reasoning (grounded → verified → cached)
1. Per final-100 candidate, build a **fact sheet** of *only literal profile values* (years, titles, named skills + durations, signal values, the one honest concern if any).
2. Offline LLM writes 1–2 sentences **grounded solely in the fact sheet**, tone banded by rank: **1–10** confident · **11–50** strong-with-caveat · **51–90** mixed · **91–100** explicitly borderline.
3. **Automated verifier:** every number, title, and skill string in the sentence must appear literally in the fact sheet → regenerate on failure.
4. Cache as a `candidate_id`-keyed artifact. Passes specific-facts / JD-connection / honest-concerns / no-hallucination / variation / rank-consistency **by construction.**

### Phase 8 — Validation (the only feedback loop)
- **TREC-style pooled judging:** label the **union of every variant's top-100** (+~50 random floor). Compute NDCG@10/@50, P@5/P@10, MAP-proxy for all variants on the same pool.
- **Trap regression suite:** named stuffers/twins/plain-language finds from the sample + forensics must land where expected.
- **Model selection on the harness only.** Freeze on harness evidence, never on a hunch.

---

## 4. REPLAY — `rank.py` (≤5 min · CPU · no network · ≤16 GB)

```
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

```
stream-parse JSONL (orjson, project only needed fields)      # ~30–60 s, never json.loads the whole file
  → compute gates + consistency checks LIVE on parsed input   # honeypot logic provably runs on the grader's file
  → join precomputed features by candidate_id                 # parquet, lazy/memory-mapped; peak < 4 GB
      → on-the-fly fallback featurizer (bundled INT8 bge-small) for IDs absent from the store  # required for the ≤100 sandbox demo
  → student.predict (seed-ensemble average)                   # < 15 s for 100K
  → apply exclusions (honeypot triple-guard, hard gates)
  → sort by score desc; round to 6 dp; tiebreak candidate_id asc
  → attach cached reasoning (composer fallback if missing)
  → write UTF-8 CSV: candidate_id,rank,score,reasoning  (exactly 100 rows + header)
  → self-run validate_submission.py on the emitted file
```

**Budget:** ~2–3 min wall-clock, ~2–4 GB RAM, artifacts well under 5 GB. **Determinism:** fixed seeds, pinned versions, "now" = `max(last_active_date)`, deterministic tiebreaks.

---

## 5. Repository layout (what Stage 3/4 will inspect)

```
.
├── rank.py                      # the ONE reproduce command's entry point
├── requirements.txt             # pinned versions (==), CPU-only wheels
├── submission_metadata.yaml     # filled from the provided template, at repo root
├── README.md                    # problem + solution + reproduce + results
├── docs/decisions.md            # defend-your-work log (Stage 5 script)
├── src/
│   ├── parse.py                 # streaming orjson parser → records
│   ├── features.py              # feature store builder (offline) + live feature subset (replay)
│   ├── lexicon.py               # company-type + skill ontology + founding-year table
│   ├── honeypot.py              # consistency suite + IsolationForest + triple-guard
│   ├── model.py                 # LightGBM train/predict, monotone constraints, seed ensemble
│   ├── reasoning.py             # fact-sheet → grounded gen → verifier → cache (offline) + composer (replay)
│   └── eval.py                  # pooled-judging harness, NDCG/MAP/P@k, trap regression
├── offline/                     # NOT run at ranking time — builds artifacts
│   ├── 00_docx_forensics.py
│   ├── 01_build_features.py
│   ├── 02_text_scores.py        # Qwen3 embed/rerank + BM25 + fusion
│   ├── 03_teacher_label.py
│   ├── 04_train_student.py
│   ├── 05_head_audit_retrain.py
│   └── 06_reasoning.py
├── artifacts/                   # precomputed: features.parquet, model/, reasoning.json, founding_years.csv
└── data/                        # candidates.jsonl, sample, schema (gitignored if large; documented)
```

---

## 6. Defensibility (Stages 3–5)

- **Stage 3 (reproduction + honeypot):** fresh-clone dress rehearsal under `timeout 300` + Docker mem cap; triple-guard → 0 honeypots with margin 10; `rank.py` honestly parses `--candidates`.
- **Stage 4 (manual review):** reasoning passes all 6 checks by construction; **real incremental git history** (commit the labeling/eval notebooks — they're proof, not something to hide); methodology coherent.
- **Stage 5 (defend-your-work):** the reproduce command runs a **model, not a lookup**; the design is the **LANTERN / ConFit-v3 distillation pattern** from the organizers' own field; **monotone constraints + feature importances** are clean slides; **teacher self-consistency stats + fairness note** (teacher ignores names/prestige) pre-empt the "LLM-as-truth" caution (Soboroff). [`decisions.md`](./decisions.md) is the script.

---

## 7. Honest risk register

| Risk | Mitigation |
|---|---|
| Teacher = ceiling (systemic bias) | Anchor on 3 published judgments; measure self-consistency; drop low-consistency labels; hand-review top-50 disagreements; teacher ignores names/prestige |
| Pooled labels measure *rubric agreement*, not hidden truth | Reconstruct rubric near-verbatim from docx; say this plainly at interview |
| Unseen IDs in the Stage-3 sandbox crash `rank.py` | Bundled INT8 fallback featurizer guarantees a valid 100-row output |
| OOM on 465 MB join | Streaming parse + lazy/memory-mapped parquet; peak < 4 GB; tested under Docker cap |
| Stage-3 Docker has no network for artifact rebuild | Confirm in Step 1 (forensics); if no → Git LFS artifacts, repo < ~2 GB |
| Late idea breaks the final | Submission 3 is never an experiment; frozen harness-validated version is final; submitted early |

---

## 8. Measured results (this repo)

Full tables, tier histograms, and reproduce command: **[`RESULTS.md`](./RESULTS.md)**.

| Area | Latest (2026-06-29) |
|------|---------------------|
| Replay `rank.py` | **14.64 s** / 100K, **215** excluded, validator **PASS** |
| Text scores | BM25 + **TF-IDF** dense + fusion + BM25-JD proxy reranker (100K non-null) |
| Labels | **9,989** pseudo-teacher labels; pool from hybrid retrieval |
| Deployed model | LightGBM **pointwise** (5-seed ensemble) |
| Harness (teacher tiers) | Pointwise composite **0.852**; lambdarank **0.811**; rules **0.713** |
| Honeypots | **0** in top 110 (hard check) |
| Tests | **122** pytest passed |

**No-API orchestration:** `python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv`
