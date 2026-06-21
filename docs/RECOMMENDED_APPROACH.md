# Recommended approach (target architecture)

This is the **best-fit solution** for the official challenge given the constraints in [`CHALLENGE_BRIEF.md`](./CHALLENGE_BRIEF.md). **Latest measured output:** [`RESULTS.md`](./RESULTS.md).

---

## Implementation status (2026-06-21)

| Item | Status |
|------|--------|
| `scripts/run_pipeline_no_api.py` | ✅ Full offline + replay |
| BM25 + lexical TF-IDF dense + fusion | ✅ `offline/02` `--dense-backend lexical` |
| Proxy reranker (BM25 vs JD) | ✅ `step6_proxy` |
| Parquet score merge (pandas) | ✅ Fixed NaN join bug |
| Pseudo-teacher + `--keep-inconsistent` | ✅ ~9,989 labels |
| LightGBM lambdarank (5K query groups) | ✅ Deployed in `artifacts/model/` |
| `audit_flags.json` | ✅ Written in `offline/05` |
| `submission.csv` + validator | ✅ |
| Hosted LLM teacher / reasoning | ⏳ Optional upgrade |
| HF Qwen embeddings / reranker | ⏳ Optional (`--dense-backend hf`) |
| Git LFS artifacts + sandbox + PDF deck | ⏳ Portal |

---

## Recommended design (one sentence)

**Offline teacher + hybrid retrieval + rich features → LightGBM lambdarank student → CPU replay with live honeypot guards and verified reasoning cache.**

---

## Why this beats alternatives

| Approach | Verdict |
|----------|---------|
| **Keyword / skill-count only** | Fails stuffers and misses plain-language Tier 5s; honeypots in top 10 hurt NDCG and P@10. |
| **Online LLM per candidate at rank time** | Violates 5 min / no-network; fails Stage 3. |
| **Pure vector search top-100** | Good recall, weak on behavioral twins and disqualifiers; needs learning-to-rank on top. |
| **Pure rules baseline** | Defensible and fast but caps NDCG@10 vs learned ranker on rubric labels. |
| **Offline distill to GBT (this repo)** | Matches production latency story; explainable; fits organizer compute box. |

---

## Target pipeline (phases)

### Phase 0 — Spec fidelity

- Treat `job_description.docx`, `submission_spec.docx`, `redrob_signals_doc.docx` as source of truth (`data/_txt_*.txt`).
- Rubric: **0–5** tiers, honeypot = 0, P@10 relevant = tier ≥ 3 → [`teacher_rubric.md`](./teacher_rubric.md).

### Phase 1 — Feature store ✅

- Stream 100K → `features.parquet` (career evidence, company lexicon, verification gap, disqualifier flags, 23 signals).
- **Next:** extend skill ontology from `artifacts/skill_counts.csv`.

### Phase 2 — Text intelligence ✅ (no-API path)

- Career document = descriptions + summary + headline (**exclude skills list**).
- **Shipped:** BM25 + **TF-IDF** dense + fusion + BM25-JD proxy reranker.
- **Upgrade:** Qwen3 embed/rerank via `offline/02` + GPU (`requirements-offline-gpu.txt`).

### Phase 3 — Labels ⚠️ partial

- High-recall pool ~10K ✅
- Pseudo-teacher ✅ (no API)
- **Next:** LLM or human anchors for non-circular harness

### Phase 4 — Student ✅

- LightGBM lambdarank + pointwise, monotone constraints, harness selection.
- **Deployed:** lambdarank for replay (see `RESULTS.md`).

### Phase 5 — Safety ✅

- Hard rules live in `rank.py`; `audit_flags.json`; honeypot check script.

### Phase 6 — Reasoning ✅ (composer)

- Verifier-gated composer → `reasoning.json`; LLM optional offline.

### Phase 7 — Replay ✅

- `rank.py` ~52 s / 100K, self-validates.

---

## What to ship for judges

1. Repo + `docs/RESULTS.md` + reproduce command.
2. **PDF deck** (use RESULTS tables).
3. `submission.csv`.
4. `submission_metadata.yaml` + sandbox.

---

## Priority fixes (remaining ROI)

1. **LLM labels** on labeling pool (biggest quality lift).
2. **Harness relevance** from anchors, not teacher-only.
3. **Skill ontology** expansion.
4. **Git LFS** for `features.parquet`, `model/`, `reasoning.json`.
5. Portal: team info, sandbox, deck PDF.

---

## Honest positioning in the deck

- Strength: production-shaped ranker, **~52 s** replay, trap-aware, documented benchmarks.
- Limitation: pseudo-teacher ceiling; harness NDCG vs teacher ≠ hidden judge score.
- Narrative: AI recruiter = JD + semantic fit + signals, not keyword filter.
