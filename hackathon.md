# hackathon.md — The Complete Project Guide

> One file that explains **everything** about this project: what it is, why it
> exists, how it works, the folder structure, the tech stack, and the workflow.
> Written in two registers — **simple language** (plain English, no jargon) and
> **technical** (the real details an engineer/judge needs). Important points
> pulled in from the other `.md` files (`README.md`, `docs/architecture.md`,
> `docs/CHALLENGE_BRIEF.md`, `docs/RESULTS.md`, `docs/decisions.md`,
> `_local/CLAUDE.md`, `_local/WORKFLOW.md`) are folded in here so you don't have
> to hunt through them.

---

## 1. What is this project?

### In simple language
Imagine a company posts **one** job — a "Senior AI Engineer" for a founding team.
Then **100,000 people** apply. A human recruiter could never read all of them
fairly. This project is a computer system that reads every single profile, scores
how good a fit each person is for that exact job, and hands back a **ranked
shortlist of the best 100 candidates** — each with a short, honest sentence
explaining *why* they made the list.

The hard part is that the data is full of **traps** designed to fool dumb
keyword-matching systems:
- People who **stuff their profile with buzzwords** ("FAISS, Pinecone, LangChain")
  but have never actually done the work.
- People who are **genuinely great** but never use the trendy words (a Swiggy
  recommendations engineer who never wrote "RAG").
- **"Behavioral twins"** — two people with identical skills, but one is active and
  responsive while the other has ghosted recruiters for 6 months.
- **~80 "honeypots"** — fake profiles that are subtly impossible (e.g. "8 years of
  experience at a company that was founded 3 years ago", or "expert in 10 skills
  with 0 months of using any of them"). If too many of these sneak into your top
  100, you're **disqualified**.

This project is built to see through all of those tricks the way a *smart*
recruiter would — by reading career history and context, not just matching words.

### In technical language
This is the submission for **Track 1 of the India Runs Data & AI Challenge**
(organizer: Redrob). The fixed task: rank **100,000** candidates in
`candidates.jsonl` against one fixed **"Senior AI Engineer — Founding Team"** JD,
and emit the **top 100** as a CSV with columns `candidate_id, rank, score,
reasoning`.

**The governing design principle:** *the problem is 100% precomputable.* There is
one fixed JD and 100K fixed candidates, and **no new data arrives at ranking
time**. Therefore every score is an **offline quantity**. All the heavy
intelligence (embeddings, reranking, LLM-style labeling, model training, reasoning
generation) happens **offline** and is baked into `artifacts/`. The actual
ranking command, `rank.py`, is a **thin, deterministic replay** — not a live
inference engine.

The scoring metric the organizers use (on hidden ground-truth labels) is:

```
Final = 0.50·NDCG@10 + 0.30·NDCG@50 + 0.15·MAP + 0.05·P@10
Tiebreaks: P@5 → P@10 → earlier submission timestamp
```

The **top 10 is 50% of the score**, and you get exactly one blind shot at it (no
live leaderboard, max 3 submissions, last valid one counts). So the design spends
disproportionate effort on the very top of the list and on reducing its variance.

---

## 2. The hard constraints (why the design looks the way it does)

At submission time, the ranking step is reproduced by the organizers in a sandbox
("Stage 3"). It **must** run inside this box:

| Constraint | Limit |
|------------|-------|
| Wall-clock time | ≤ **5 minutes** |
| RAM | ≤ **16 GB** |
| Compute | **CPU only** (no GPU at ranking time) |
| Network | **OFF** (no hosted LLM / API calls during ranking) |
| Disk (intermediate) | ≤ **5 GB** |

Two consequences baked into everything:
1. **No online model calls.** Embedding/reranker/LLM work is precomputed offline
   and stored as features in `artifacts/`. Confirmed from the spec: the Stage-3
   Docker has **no network**, so artifacts ship with the repo (via Git LFS for the
   large binaries), nothing is downloaded at rank time.
2. **No loading the 465 MB file into memory.** `rank.py` **streams** the JSONL
   line-by-line with `orjson`, projecting only the fields it needs. Peak working
   set stays well under the cap (~92 MB observed at feature-build).

The whole system is **deterministic**: every model seed is pinned, every
dependency is pinned with `==`, and the reference "now" date is
`max(last_active_date)` across the pool — **never** the wall-clock `datetime.now()`
(so the same input always produces the same output).

---

## 3. How it works — the two phases

The system has exactly two phases. This split is the core idea.

```
┌─────────────────────────────────────────────┐     ┌──────────────────────────────────┐
│  OFFLINE  (no time limit; GPU+network OK)    │     │  REPLAY  rank.py                  │
│  builds all intelligence → artifacts/        │ ──▶ │  ≤5 min · CPU · no network        │
│  offline/01..06 + scripts/run_pipeline...    │     │  parse→join→predict→sort→reason   │
└─────────────────────────────────────────────┘     └──────────────────────────────────┘
```

### Phase A — OFFLINE (builds the brain)

Run once, slowly, with whatever compute you have. Produces the files in
`artifacts/`. Steps (`offline/00`–`06`):

| Step | Script | What it does (simple) | What it does (technical) |
|------|--------|------------------------|--------------------------|
| 00 | `00_docx_forensics.py`, `00b_profile_pool.py` | Read the rules and study the data | Mine the 3 `.docx` spec files for the exact tier scale (0–5), rubric wording, signal philosophy; profile the 100K pool (companies, skills, sentinel rates) |
| 01 | `01_build_features.py` | Turn each profile into numbers | Streaming `orjson` parse → `features.parquet` (~46 columns): career evidence, company type, behavioral signals, disqualifier flags, honeypot consistency flags |
| 02 | `02_text_scores.py` | Score how well each career text matches the job | BM25 over career text + dense TF-IDF (or Qwen3 embeddings) cosine to JD query variants + convex fusion + a BM25-vs-JD proxy reranker — written back into the feature store |
| 03 | `03_teacher_label.py` | Give training examples "grades" | A teacher labels a high-recall pool (~10K candidates) with tier 0–5 + a 0–100 score using the reconstructed rubric |
| 04 | `04_train_student.py` | Train the model that does the ranking | LightGBM student — compares `lambdarank` vs pointwise regression, monotone constraints, 5-seed ensemble; evaluated on a pooled-judging harness |
| 05 | `05_head_audit_retrain.py` | Double-check and fix the top of the list | Multi-pass audit of the top candidates → labeled pairs → **retrain** (never a hardcoded override table); produces `audit_flags.json` |
| 06 | `06_reasoning.py` | Write the "why" sentence for each finalist | Fact-sheet → grounded generation → automated verifier (every number/skill must appear literally) → cache in `reasoning.json` |

There's a **one-command, no-API** orchestrator that runs `01`→`06` + `rank.py` +
the validator:

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

### Phase B — REPLAY (`rank.py`, the actual submission command)

This is the only thing reproduced under the 5-min/CPU/no-network cap:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

What it does, in order (it streams the pool **twice**):

1. **Pass 1:** stream the file once just to compute the reference "now" =
   `max(last_active_date)` (deterministic).
2. **Pass 2:** stream again — for each candidate: build the live feature row, join
   the precomputed offline features by `candidate_id`, run the **honeypot
   consistency suite live**, and if it passes, score it with the model ensemble.
   Keep a running **top-K heap** (never materializes all 100K records).
3. **Exclude honeypots** via the triple-guard (see §5).
4. **Sort** by score descending, round to 6 dp, tie-break by `candidate_id`
   ascending; apply a small deterministic **uncertainty penalty inside the top-20
   band** to prefer low-variance candidates among near-ties.
5. **Attach reasoning** from the cache (a deterministic composer fills in if a
   candidate is missing — needed for the ≤100-candidate sandbox demo).
6. **Write the CSV** and **self-validate** with the organizer's own
   `validate_submission.py`.

If model artifacts are missing, `rank.py` **degrades gracefully** to a transparent
rules-based scorer (`_RULES_V0_WEIGHTS` in `rank.py`) so it always emits a valid
CSV.

---

## 4. The traps and how each is defeated

This table (from `docs/architecture.md`) is the heart of the "smart recruiter"
claim:

| Trap | What it looks like | How we defeat it |
|------|--------------------|------------------|
| **Keyword stuffers** | A "Marketing Manager" listing "FAISS, Pinecone, LangChain" | Weight **career-text evidence** over skill *names*; `claimed_unverified_ratio`; assessment-backed coverage; teacher reads full context |
| **Honeypots (~80)** | "8 yrs at a 3-yr-old company"; "expert in 10 skills, 0 months each" | **Triple-guard**: hard rules ∧ IsolationForest anomaly ∧ audit contradiction; founding-year table; **single violation → ineligible** |
| **Behavioral twins** | Identical skills, opposite availability (active vs ghost) | **Learned** signal weights with monotone constraints — not a hand-tuned multiplier |
| **Plain-language Tier 5s** | A Swiggy recsys engineer who never wrote "RAG" | Instruction-aware embeddings + high-recall labeling union + a rules path so retrieval misses still get labeled |

---

## 5. The honeypot triple-guard (the disqualification defense)

Honeypots are forced to tier 0; **>10% of them in your top 100 = instant Stage-3
disqualification.** The rule is intentionally strict: **a single hard violation
makes a candidate ineligible** for the top 100 (we never require 2+).

`exclude if (hard_rule_violation) OR (anomaly_score high AND corroborated) OR (audit flags contradiction)`

- **Consistency suite (~12 checks)** in `src/honeypot.py`, e.g. tenure vs company
  founding year; `expert` proficiency + `duration_months == 0`; years-of-experience
  ≫ career span; certification dated before the technology existed; `end_date <
  start_date`; `is_current` with a non-null `end_date`. **7 are HARD, ~5 are SOFT.**
- **Founding-year table** (`artifacts/founding_years.csv`) — curated for **55 real
  companies**; fictional placeholder companies (Wayne Enterprises, Initech, Hooli,
  etc.) are **deliberately omitted** so the tenure-vs-founding check **abstains**
  rather than fire a false positive. `FOUNDING_MARGIN_YEARS = 1` requires a
  multi-year gap so off-by-one noise doesn't fire.
- **IsolationForest** anomaly model (`src/anomaly.py`, scikit-learn) over coherence
  features — but it **never excludes alone** (avoids false positives on
  unusual-but-real people); it must be corroborated.
- Calibrated **hard-violation rate ≈ 0.18% (181 / 100,000)** pool-wide — sane, not
  trigger-happy. Latest replay excluded **251** candidates; **0** hard honeypots in
  the top 110.

Why so strict? Falsely excluding 1 real person out of 100K costs almost nothing
(one substitute moves up); a single missed honeypot in the top 10 hits NDCG, MAP,
*and* P@10 at once, and >10 in the top 100 ends the run.

---

## 6. The model (the "student")

- **Algorithm:** LightGBM (gradient-boosted trees), CPU-friendly and fast at 100K.
- **Two heads compared** on the harness: `lambdarank` (learning-to-rank, one query
  group) vs **pointwise regression** on the teacher's 0–100 score (top-weighted).
- **Monotone constraints** encode known truths: recruiter response rate ↑, last-active
  recency ↑, interview-completion rate ↑, notice-period days ↓. (Recency is stored
  as `-(months_since_active)` so "more recent = higher" matches a +1 constraint.)
- **5-seed ensemble**, averaged; shallow trees (≤500, depth ≤6); feature
  importances exported.
- **Teacher → student distillation** pattern (the LANTERN / ConFit-v3 idea from the
  organizers' own field): an LLM-style teacher grades a pool, the LightGBM student
  learns to reproduce those grades at scale.
- **Deployed head:** `lambdarank` (`artifacts/model/`). Note: on the pooled harness
  the *pointwise* head scores higher against teacher tiers (0.852 vs 0.811), but
  lambdarank is deployed for the final ranking (rationale in
  `artifacts/model/selection.json`) — because the pointwise "win" partly reflects
  it fitting the teacher's own scores, and lambdarank optimizes the ranking metric
  directly.
- **Top lambdarank feature importances (gain):** `product_tenure_months`,
  `n_soft_flags`, `total_career_months`, `is_product_current`,
  `max_assessment_score`, `dense_score`.

### The reasoning column (Stage-4 quality check)
Each of the 100 finalists gets a 1–2 sentence explanation, **grounded in a fact
sheet of only literal profile values**, then **automatically verified** (every
number, title, and concrete skill in the sentence must appear literally in the
fact sheet, else regenerate). Tone is **banded by rank**: ranks 1–10 confident,
11–50 strong-with-caveat, 51–90 mixed, 91–100 explicitly borderline. This passes
the 6 Stage-4 reasoning checks (specific facts, JD connection, honest concerns, no
hallucination, variation/not-templated, rank-consistent tone) **by construction**.

---

## 7. Tech stack & resources

### Languages / runtime
- **Python 3.10+** (3.11 recommended; developed/tested on 3.10.11).

### Replay dependencies (what `rank.py` needs — pinned, CPU-only)
| Package | Version | Role |
|---------|---------|------|
| `orjson` | 3.11.9 | Streaming JSONL parse (only needed fields) |
| `numpy` | 1.26.4 | Numerics |
| `pandas` | 2.2.2 | Feature frames / CSV IO |
| `polars` | 1.41.2 | Lazy / memory-mapped parquet join at replay |
| `pyarrow` | 24.0.0 | Parquet engine |
| `lightgbm` | 4.6.0 | Student model (monotone constraints) |
| `scikit-learn` | 1.5.1 | IsolationForest anomaly guard |
| `PyYAML` | 6.0.3 | `submission_metadata.yaml` handling |

### Offline-only / dev dependencies
- `rank-bm25` 0.2.2 (BM25 text scores) · `pytest` 8.3.3 · `ruff` 0.6.9 / `black`
  24.8.0 (lint/format).
- **Optional GPU upgrade path** (not required, not in the committed replay env):
  `torch` / `sentence-transformers` / `transformers` for **Qwen3-Embedding-0.6B**
  and **Qwen3-Reranker** — enabled with `offline/02_text_scores.py --dense-backend
  hf`. There's a separate `requirements-offline-gpu.txt` and a `.venv_gpu/` for
  this. **No hosted LLM API** is used in the default build — everything works
  no-API with BM25 + TF-IDF and a deterministic pseudo-teacher.

### Key resources / artifacts shipped
- `artifacts/features.parquet` — the feature store (100K × ~46 columns)
- `artifacts/model/` — the trained LightGBM ensemble + `selection.json`
- `artifacts/reasoning.json` — verified reasoning cache
- `artifacts/audit_flags.json` — audit exclusion set (56 ids)
- `artifacts/founding_years.csv` — 55-company founding-year table
- `artifacts/harness_results.json` — model-comparison metrics

---

## 8. Folder / repository structure

```
D:\Hackthon\
├── rank.py                      # ENTRY POINT — the ≤5-min CPU/no-network replay
├── validate_submission.py       # organizer's CSV validator (byte-identical copy)
├── submission.csv               # the portal-ready output (100 rows + header)
├── submission_metadata.yaml     # team/portal metadata (fill TODOs before upload)
├── requirements.txt             # pinned, CPU-only replay deps
├── requirements-offline-gpu.txt # optional GPU stack for stronger embeddings
├── pyproject.toml               # project config (lint/format/build)
├── conftest.py                  # pytest config
│
├── src/                         # core library (imported by rank.py + offline/)
│   ├── parse.py                 # streaming orjson parser; schema registry (F/CareerF/SkillF/EduF); sort keys
│   ├── features.py              # feature-store builder + live replay feature subset
│   ├── lexicon.py               # company-type + skill ontology + founding-year loader
│   ├── honeypot.py              # ~12-check consistency suite + triple-guard
│   ├── anomaly.py               # IsolationForest anomaly model
│   ├── model.py                 # LightGBM train/predict, monotone constraints, seed ensemble
│   ├── reasoning.py             # fact-sheet → grounded gen → verifier → cache + composer fallback
│   ├── top20.py                 # deterministic uncertainty penalty in the top-20 band
│   ├── labeling_pool.py         # high-recall union pool assembly
│   ├── pseudo_teacher.py        # deterministic no-API teacher (tier 0–5 + 0–100)
│   ├── pool_score.py            # pooled-judging scoring helpers
│   ├── eval.py                  # NDCG/MAP/P@k harness + trap regression
│   └── internal_validate.py     # stricter validator (sandbox allow_fewer mode)
│
├── offline/                     # NOT run at ranking time — builds artifacts/
│   ├── 00_docx_forensics.py     # mine the spec .docx for rubric/tier scale
│   ├── 00b_profile_pool.py      # profile the 100K pool (companies/skills/sentinels)
│   ├── 01_build_features.py     # streaming featurize → features.parquet
│   ├── 02_text_scores.py        # BM25 + dense + fusion + proxy reranker
│   ├── 03_teacher_label.py      # teacher labels the high-recall pool
│   ├── 04_train_student.py      # train LightGBM (lambdarank vs pointwise) + harness
│   ├── 05_head_audit_retrain.py # multi-pass audit → retrain
│   ├── 06_reasoning.py          # generate + verify + cache reasoning
│   └── _honeypot_audit.py       # calibrate the consistency suite on the real pool
│
├── scripts/
│   ├── run_pipeline_no_api.py   # one command: offline 01→06 + rank.py + validate
│   └── honeypot_top100_check.py # sanity check: count honeypots in top N
│
├── tests/                       # 119 pytest tests (parse, features, honeypot, model,
│   │                            #   reasoning, top20, eval, trap regression, validators,
│   │                            #   rank smoke test) + fixtures/
│
├── artifacts/                   # PRECOMPUTED outputs (ship large ones via Git LFS)
│   ├── features.parquet, model/, model_lambdarank/, reasoning.json,
│   ├── founding_years.csv, audit_flags.json, audit_disagreements.json,
│   ├── harness_results.json, teacher_labels*.parquet, labeling_pool.parquet,
│   ├── anomaly/, hf_cache/, company_counts.csv, skill_counts.csv, ...
│
├── data/                        # organizer inputs (gitignored if large; local only)
│   ├── candidates.jsonl         # 100,000 profiles (~465 MB)
│   ├── sample_candidates.json   # first 50 (schema inspection)
│   ├── candidate_schema.json    # JSON Schema for records
│   ├── job_description.docx + submission_spec.docx + redrob_signals_doc.docx
│   ├── _txt_*.txt               # extracted text of the docx (for search/diff)
│   ├── validate_submission.py   # organizer copy (root copy is byte-identical)
│   └── submission_metadata_template.yaml
│
├── docs/                        # the documentation set (see §10)
├── notebooks/                   # labeling/eval notebooks (Stage-4 evidence)
└── _local/                      # internal planning docs (CLAUDE.md, WORKFLOW.md, PANEL_REVIEW.md)
```

---

## 9. How the project is organised & the workflow

### The organising philosophy
- **`offline/` builds the brain; `rank.py` replays it.** All slow, smart,
  network/GPU-using work lives in `offline/*.py` and writes to `artifacts/`. The
  submission command stays thin, honest, and memory-safe.
- **`src/` is the shared library** both phases import — so the *exact same*
  parsing, feature, and honeypot logic that built the artifacts also runs live in
  `rank.py` (no train/serve skew).
- **`docs/` is the paper trail** — design, decisions, results — built to survive a
  judge's scrutiny (Stages 3–5).

### The development workflow (from `_local/WORKFLOW.md`)
Work proceeds as numbered **STEPS** chained by **EXIT GATES** (not calendar days).
A step is done only when its gate is literally true (tests pass, file exists). The
flow runs through phases:

- **Phase A — Forensics & baseline:** read the docx spec, build the founding-year
  table, the streaming parser, the consistency suite, and a rules baseline that
  already emits a valid CSV (submission insurance).
- **Phase B — Semantic features + teacher:** career-text BM25/embeddings/fusion,
  reranker, high-recall labeling pool, teacher labels, first LightGBM student +
  pooled harness.
- **Phase C — Feature iteration:** verification-gap and disqualifier features,
  reranker feature, sentinel hygiene, re-select the model on the harness.
- **Phase D — Head refinement + reasoning + reproduction:** multi-pass audit →
  retrain, top-20 risk adjustment, grounded+verified reasoning, honeypot sign-off,
  Stage-3 dress rehearsal on a fresh clone.
- **Phase E — Final eval, freeze, submit early.**
- **Phase F — Interview prep** (the defend-your-work one-pager).

**Engineering discipline:** commit after every step (many honest, time-spread
commits — a flat git history is penalized at Stage 4); update `docs/decisions.md`
on every non-obvious choice (it's the Stage-5 interview script); re-run tests +
the replay smoke test after any change to the `rank.py` path.

### The validation loop (the only real feedback signal)
There's **no live leaderboard**, so validation uses **TREC-style pooled judging**:
label the union of every model variant's top-100 (+ a random floor), then compute
NDCG@10/@50, MAP, P@10 for all variants on that same pool. Models are selected on
this harness only — never on a "spectrum" sample (which has ~zero overlap with any
system's top-100 and just measures noise).

---

## 10. The other docs (what each contains)

| File | What's in it |
|------|--------------|
| `README.md` | Quick start, reproduce command, portal checklist, repo layout |
| `docs/CHALLENGE_BRIEF.md` | Official rules mirror: problem, requirements, constraints, scoring, evaluation stages, bundle contents |
| `docs/architecture.md` | The full engineering design: design thesis, traps, offline pipeline phases 0–8, replay flow, defensibility, risk register |
| `docs/decisions.md` | The "defend-your-work" log — every non-obvious decision with why/rejected/evidence; the running engineering log |
| `docs/RESULTS.md` | Canonical benchmark numbers: timings, harness metrics, feature stats, tier distributions, top-5 output |
| `docs/teacher_rubric.md` | The 0–5 relevance rubric reconstructed near-verbatim from the organizer docx |
| `docs/RECOMMENDED_APPROACH.md` | Target state / roadmap / gaps |
| `docs/interview_one_pager.md` | Stage-5 interview prep |
| `data/README.md` | How to place the organizer inputs into `data/` |
| `artifacts/README.md` | What each artifact is |
| `_local/CLAUDE.md` | Operating rules + hard constraints + anti-patterns (internal) |
| `_local/WORKFLOW.md` | The step-and-gate execution plan (internal) |
| `_local/PANEL_REVIEW.md` | Why every choice was made + anti-patterns (internal) |

---

## 11. Latest measured results (from `docs/RESULTS.md`, run 2026-06-21)

Environment: Windows, Python 3.10.11, CPU-only replay, **no hosted LLM API**.
Reference now = **2026-05-27**.

| Check | Result | Limit / note |
|-------|--------|--------------|
| Replay `rank.py` on 100K | **~52 s** | ≤ 5 min |
| GPU at rank time | **No** | CPU only |
| Network at rank time | **No** | Off |
| Peak memory (feature build) | **~92 MB** | ≤ 16 GB |
| CSV rows | **100** + header | exactly 100 |
| Organizer validator | **PASS** | required |
| Candidates excluded | **251** | honeypot triple-guard |
| Honeypots in top 110 | **0** hard | DQ if >10% in top 100 |
| Teacher labels | **9,989** | high-recall pool |
| Tests | **119 passed** | `pytest -q` |

**Harness composite** (teacher tiers, pool = 338): rules baseline **0.713**,
lambdarank **0.811**, pointwise **0.852** (deployed: lambdarank).

**Output (`submission.csv`):** 100 rows, score range **5.285–7.382**. Top ranks
skew toward product/fintech/edtech employers (Meesho, Paytm, CRED, Razorpay,
Zomato) with retrieval/vector/IR evidence grounded in their profiles.

| Rank | candidate_id | score |
|------|--------------|-------|
| 1 | CAND_0005883 | 7.382 |
| 2 | CAND_0038716 | 7.326 |
| 3 | CAND_0044804 | 7.294 |
| 4 | CAND_0063585 | 7.285 |
| 5 | CAND_0014209 | 7.281 |

### Honest limitations (stated for the deck / interview)
1. Harness relevance = pseudo-teacher tiers; high NDCG vs the teacher doesn't
   guarantee agreement with the hidden judges.
2. No frontier LLM at build time — the rubric is implemented as transparent
   features + a composer, not a hosted model.
3. Lexical TF-IDF substitutes for Qwen embeddings in the no-API build (upgrade path
   is `offline/02 --dense-backend hf`).
4. The organizer composite on hidden labels isn't available pre-submission — the
   internal harness is for **relative** model selection only.

---

## 12. How to run it (cheat sheet)

```bash
# 0. Set up
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

# 1. Place organizer inputs in data/ (candidates.jsonl, spec files, schema)

# 2a. Produce the submission from existing artifacts (the real reproduce command)
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

# 2b. OR rebuild everything offline (no API) then rank
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv

# 3. Validate + honeypot sanity check
python validate_submission.py submission.csv
python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110

# 4. Tests
python -m pytest -q
```

### Portal deliverables (what gets submitted)
1. **GitHub repo** — this project, reproducible.
2. **Deck (PPT → PDF)** — what/why/how.
3. **Ranked CSV** — 100 rows + header, passes `validate_submission.py`.
4. **`submission_metadata.yaml`** (fill the TODOs: team name, contacts, GitHub URL,
   sandbox link) + a **sandbox link** that runs the ranker on ≤100 candidates.

---

### One-paragraph summary
This project ranks 100,000 candidates for one fixed Senior AI Engineer job and
returns a trustworthy, explained top-100. Because the JD and the pool are fixed,
all the intelligence is **precomputed offline** (feature store, hybrid text scores,
a teacher-labeled LightGBM ranker, honeypot guards, verified reasoning) and the
submission command `rank.py` is a **thin, deterministic, CPU-only, network-free
replay** that streams the file, excludes honeypots with a strict triple-guard,
predicts, sorts, attaches grounded reasoning, and self-validates — finishing in
~52 seconds with 0 honeypots in the top 110 and 119 passing tests.
