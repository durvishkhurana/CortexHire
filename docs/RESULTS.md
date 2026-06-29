# Results & benchmarks (reference)

Canonical record of **historical reproducible runs** on the full **100,000** candidate pool.
**Environment:** macOS local sandbox, Python 3.12.13, CPU-only replay, **no hosted LLM API**.
**Data:** `data/candidates.jsonl`, reference now = **2026-05-27** (`max(last_active_date)`).

Note: these results are not re-runnable from a fresh clone unless the private organizer data and generated artifacts are present or rebuilt.

**Reproduce:**

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

---

## Latest full pipeline run (2026-06-29)

End-to-end command above (includes `offline/01` feature rebuild). Approximate wall times from logs:

| Step | Script | Duration (approx.) | Notes |
|------|--------|-------------------|--------|
| Features | `offline/01_build_features.py` | ~8 s | 100K × 49 columns → `features.parquet` |
| Text step5 | `offline/02_text_scores.py` | ~22 s | BM25 + **lexical TF-IDF** dense + fusion |
| Text step6 | `offline/02_text_scores.py` | ~8 s | Proxy **reranker** = BM25 vs full JD text |
| Labels | `offline/03_teacher_label.py` | ~2 s | Pool **9,989** ids; **9,989** labels kept (`--keep-inconsistent`) |
| Train | `offline/04_train_student.py` | ~22 s | Pointwise + lambdarank; harness on pooled top-100s |
| Audit | `offline/05_head_audit_retrain.py` | ~14 s | **38** pairwise adjustments; **19** audit flags |
| Rank | `rank.py` | **14.64 s** | 100K streamed; **215** excluded |
| Reasoning | `offline/06_reasoning.py` | ~2 s | 100 verified composer strings → `reasoning.json` |
| Validate | `validate_submission.py` | instant | **Submission is valid.** |

**Total pipeline:** 107.03 s observed for the full no-API command; replay is 14.64 s.

---

## Replay (`rank.py`) vs organizer limits

| Metric | Observed | Limit |
|--------|----------|--------|
| Wall-clock (100K) | **14.64 s** | ≤ 5 min |
| GPU at rank time | **No** | CPU only |
| Network at rank time | **No** | Off |
| CSV rows | **100** + header | 100 |
| Organizer validator | **PASS** | required |
| Honeypot hard check (top 110) | **0** | DQ if >10% in top 100 |

```bash
python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110
# checked top 110 ranks; hard honeypots in set: 0
```

---

## Feature store & text scores (`artifacts/features.parquet`)

| Column | Non-null rows | Distinct values | Min | Max |
|--------|---------------|-----------------|-----|-----|
| `bm25_score` | 100,000 | 19,828 | 6.13 | 34.93 |
| `dense_score` (TF-IDF) | 100,000 | 70,885 | 0.004 | 0.095 |
| `fusion_score` | 100,000 | 89,357 | 0.001 | 1.000 |
| `reranker_score` (proxy) | 100,000 | 30,908 | 13.55 | 69.04 |

**Stack (no API):** career-text BM25 + sklearn TF-IDF cosine (JD query variants) + rank-normalized convex fusion (α=0.5) + BM25-vs-JD proxy reranker.

---

## Labeling & teacher

| Metric | Value |
|--------|--------|
| Labeling pool size | **9,989** unique candidates |
| Labels written | **9,989** (0 dropped with `--keep-inconsistent`) |
| Pilot consistency (400 ids) | **31.75%** strict primary/strict tier agreement |
| Mean abs score delta (pilot) | **7.99** (0–100 scale) |
| Teacher type | Deterministic **pseudo-teacher** (`src/pseudo_teacher.py`) |

**Pseudo-teacher tier distribution (labeling pool):**

| Tier | Count |
|------|------:|
| 0 | 3,013 |
| 1 | 126 |
| 2 | 575 |
| 3 | 914 |
| 4 | 2,871 |
| 5 | 2,490 |

---

## Student model & harness

**Training:** LightGBM **pointwise** + **lambdarank** (5K query groups), 5-seed ensemble, monotone constraints on behavioral features.

**Pooled harness** (`artifacts/harness_results.json`): union of each variant’s top-100 + random floor → **278** judged ids; relevance = pseudo-teacher tiers.

| Variant | NDCG@10 | NDCG@50 | MAP | P@10 | Composite* |
|---------|---------|---------|-----|------|------------|
| rules_v0 | 0.892 | 0.671 | 0.0490 | 1.000 | **0.705** |
| **pointwise** (harness winner) | **1.000** | **0.973** | 0.0698 | 1.000 | **0.852** |
| lambdarank | 0.936 | 0.744 | 0.0498 | 0.900 | 0.744 |

\*Composite = `0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10` (organizer formula).

**Deployed for `rank.py`:** **`pointwise`** ensemble in `artifacts/model/` (see `selection.json`). This follows the pooled harness winner instead of forcing lambdarank.

**Top pointwise feature importances (gain):** `production_ownership_gap_flag`, `months_since_last_ic`, `n_soft_flags`, `n_hard_flags`, `cv_speech_without_ir_flag`, `is_product_current`.

---

## Safety & audit

| Item | Value |
|------|--------|
| Candidates excluded at rank time (latest run) | **215** |
| `audit_flags.json` entries | **19** |
| Hard honeypots in ranks 1–110 | **0** |
| Calibrated hard-rule rate (pool-wide, prior audit) | ~**0.18%** (see `docs/decisions.md`) |

---

## Submission output (`submission.csv`)

| Field | Value |
|-------|--------|
| Rows | 100 |
| Score range | **70.000** – **100.000** |
| Reasoning | Cached `artifacts/reasoning.json` (composer + verifier) |

**Rank 1–5 (candidate_id):**

| Rank | ID | Score |
|------|-----|------:|
| 1 | CAND_0027801 | 100.000 |
| 2 | CAND_0041669 | 98.945 |
| 3 | CAND_0061257 | 98.840 |
| 4 | CAND_0006567 | 98.813 |
| 5 | CAND_0018549 | 98.614 |

Top ranks skew toward **product/fintech/edtech** employers (e.g. Meesho, Paytm, CRED, Razorpay, Zomato) with retrieval/vector/IR skill evidence in reasoning (grounded in profiles).

---

## Tests

```bash
python -m pytest -q
```

**122 tests** passed (last verified with full artifacts present).

---

## Honest limitations (for deck / interview)

1. **Harness relevance = pseudo-teacher tiers** — high NDCG vs teacher does not guarantee hidden judge agreement; pointwise “wins” harness partly by fitting teacher scores.
2. **No frontier LLM** for labeling or reasoning at build time — rubric implemented as transparent features + composer text.
3. **Lexical dense** substitutes for Qwen embeddings when no GPU/HF download — good for no-API reproducibility; upgrade path is `offline/02` with `--dense-backend hf`.
4. **Organizer composite on hidden labels** — not available pre-submission; treat internal harness as **relative** model selection only.

---

## Related files

| File | Contents |
|------|----------|
| `artifacts/harness_results.json` | Harness metrics table |
| `artifacts/teacher_pilot_stats.json` | Pilot consistency |
| `artifacts/model/selection.json` | Deployed head + note |
| `artifacts/audit_flags.json` | Audit exclusion set |
| `artifacts/submission_final.csv` | Rank output from pipeline mid-step |
| `submission.csv` | Portal-ready CSV (repo root) |
