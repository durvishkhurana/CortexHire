# Stage 5 — One-pager (defend your work)

## What we built
Offline-first ranking for one fixed Senior AI Engineer JD over 100K profiles: heavy work (features, BM25/dense/reranker scores, pseudo-teacher labels, LightGBM student, reasoning cache) lives in `offline/*.py` → `artifacts/`. `rank.py` is a **deterministic replay**: two-pass streaming `orjson`, live honeypot checks, parquet join, ensemble predict, sort, reasoning, self-validate — CPU-only, no network.

## Why this architecture
JD + pool are fixed ⇒ every score is precomputable. Matches organizer constraints (≤5 min, ≤16 GB, no GPU/API at rank time) and the LANTERN / ConFit-v3 distillation pattern (teacher → small student).

## Teacher (offline, no paid API)
Frontier LLM labeling was replaced with a **deterministic pseudo-teacher** (rubric + feature evidence) on an ~11K labeling pool; inconsistent pairs dropped. Documented in `DECISIONS.md` — honest limitation vs a true frontier teacher.

## Student
LightGBM **lambdarank** + pointwise heads, monotone constraints on response rate, interview completion, notice period, recency; 3–5 seed ensemble. **Pooled harness** on rules + both heads picked **lambdarank** (see `artifacts/harness_results.json`).

## Honeypots
Single hard rule violation excludes; triple-guard = hard ∨ audit ∨ (anomaly ∧ soft corroboration). Founding-year table for tenure vs company age.

## Validation
TREC-style pooled judging over variant top-100s + random floor — not a spectrum sample. `validate_submission.py` matches organizer verbatim.

## Fairness / rubric
0–5 tiers reconstructed from organizer docx; instructed to weight career evidence and signals, not keyword stuffing; institution prestige only via `education.tier` field.

## AI tools
Cursor / IDE assistance during implementation; see submission_metadata.yaml.

## Reproduce
```bash
pip install -r requirements.txt
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```
Offline rebuild: see `README.md` and `offline/01`–`06`.
