# Stage 5 — One-pager (defend your work)

## What we built
Offline-first ranking for one fixed Senior AI Engineer JD over **100K** profiles: features, **BM25 + TF-IDF lexical dense + proxy reranker** (no hosted API), pseudo-teacher labels, LightGBM **pointwise** student, verified reasoning cache → `artifacts/`. `rank.py` **replays** in **14.64 s** on CPU: two-pass streaming, live honeypot checks, parquet join, predict, sort, CSV + self-validate.

## Why this architecture
JD + pool are fixed ⇒ scores are precomputable. Fits **≤5 min / 16 GB / CPU / no network** at rank time. Teacher → small student matches common distillation practice and is defensible in interview.

## Teacher (offline, no paid API)
**Deterministic pseudo-teacher** on a **~10K** high-recall labeling pool (**9,989** labels in latest run; `--keep-inconsistent`). Pilot self-consistency **~32%** (primary vs strict) — documented in [`decisions.md`](./decisions.md). Honest ceiling vs frontier LLM; see [`RESULTS.md`](./RESULTS.md).

## Student
LightGBM **pointwise** (deployed) + lambdarank (harness comparison), monotone behavioral constraints, 5-seed ensemble. Pooled harness **winner = pointwise** on teacher tiers (composite **0.852**), so `artifacts/model/selection.json` now follows that winner.

## Honeypots
Single hard violation excludes; triple-guard = hard ∨ **audit_flags** ∨ (anomaly ∧ soft). Latest run: **0** hard honeypots in top **110**; **215** total exclusions at rank time.

## Validation
TREC-style pooled harness (**278** ids in latest run). `validate_submission.py` byte-identical to organizer bundle. **122** pytest tests green.

## Fairness / rubric
0–5 tiers from organizer docx; career evidence over keywords; prestige only via `education.tier`.

## Reproduce
```bash
pip install -r requirements.txt
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```
Numbers: [`RESULTS.md`](./RESULTS.md).
