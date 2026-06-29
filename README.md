# CortexHire — Intelligent Candidate Discovery & Ranking

Redrob / India Runs **Data & AI Challenge** solution for ranking **100,000** candidates against the fixed **Senior AI Engineer — Founding Team** JD.

The goal is not keyword matching. This repo builds a deterministic, CPU-only ranking system that favors recruiter-grade evidence: shipped retrieval/ranking/recommendation work, product-company context, seniority fit, availability signals, and honeypot safety.

## Submission Status

Latest local full run: **2026-06-29** on the organizer `data/candidates.jsonl`.

| Check | Result |
|---|---:|
| Ranked CSV | `submission.csv`, 100 rows + header |
| Organizer validator | **PASS** |
| Replay time | **14.64 s** on 100K candidates |
| Replay constraints | CPU-only, no network, <16 GB target |
| Honeypot hard check | **0** hard honeypots in top 110 |
| Deployed model | LightGBM **pointwise** ensemble |
| Tests | **122 passed** |

Top-20 sanity after the final run: all top 20 candidates are in the JD’s 5-9 YoE band, no weak “transitioning/still building depth” profiles, and 0 hard honeypots in the top-110 margin.

## Reproduce

Install pinned dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # macOS/Linux
# .venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

After cloning the repo, pull LFS artifacts if they are not already present:

```bash
git lfs install
git lfs pull
```

Fast replay, when `data/candidates.jsonl` and `artifacts/` are present:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --artifacts ./artifacts
python validate_submission.py submission.csv
python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110
```

Full no-API rebuild from organizer data:

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

Do **not** use `sample_submission.csv` as a ranking source. It is format-only.

## Required Local Inputs

The organizer bundle is private/large and is not committed. Place these under `data/`:

- `candidates.jsonl` — 100,000 candidate records
- `candidate_schema.json`
- `job_description.docx`
- `submission_spec.docx`
- `redrob_signals_doc.docx`
- `sample_candidates.json`

For full-quality replay, keep generated artifacts under `artifacts/`:

- `features.parquet`
- `model/`
- `reasoning.json`
- `audit_flags.json`
- `anomaly/`
- `founding_years.csv`

Large replay artifacts are configured for Git LFS via `.gitattributes`.

## How It Works

| Phase | Files | Purpose |
|---|---|---|
| Feature build | `offline/01_build_features.py`, `src/features.py` | Career evidence, product-company context, seniority fit, behavioral signals, honeypot/coherence flags |
| Text scoring | `offline/02_text_scores.py` | BM25 + TF-IDF dense retrieval + proxy reranker against the fixed JD |
| Teacher labels | `offline/03_teacher_label.py`, `src/pseudo_teacher.py` | Deterministic 0-5 rubric labels over a high-recall pool |
| Student model | `offline/04_train_student.py`, `src/model.py` | LightGBM pointwise/lambdarank comparison; deploys harness winner |
| Audit | `offline/05_head_audit_retrain.py` | Pairwise head audit, audit flags, retrain |
| Reasoning | `offline/06_reasoning.py`, `src/reasoning.py` | Grounded reasoning cache for final 100 |
| Replay | `rank.py` | Stream candidates, join artifacts, exclude honeypots, score, sort, write and self-validate CSV |

## Recruiter-Fit Signals

The model is designed around the challenge brief:

- Rewards shipped search, retrieval, recommendation, ranking, vector DB, evaluation, and strong Python evidence.
- Prefers product-company career evidence over bare skill-list stuffing.
- Down-ranks services-only, title-chasing, stale/no-recent-IC, CV/speech-without-IR, junior/extreme-seniority mismatch, and “still transitioning” profiles.
- Uses Redrob behavioral signals for availability and responsiveness.
- Hard-excludes internally impossible profiles through live honeypot checks.

## Results

See [docs/RESULTS.md](docs/RESULTS.md) for full timings, harness metrics, feature importance, score range, and final top-5 IDs.

Current harness winner:

| Variant | Composite |
|---|---:|
| rules baseline | 0.705 |
| lambdarank | 0.744 |
| **pointwise** | **0.852** |

The harness uses pseudo-teacher labels, so it is a relative model-selection signal, not a claim about hidden judge labels.

## Repository Layout

```text
rank.py                 # deterministic replay entry point
validate_submission.py  # organizer CSV validator
submission_metadata.yaml
requirements.txt
src/                    # parser, features, model, honeypot, reasoning
offline/                # artifact build and training pipeline
scripts/                # full pipeline + honeypot top-110 check
tests/
docs/
data/                   # local organizer inputs, gitignored
artifacts/              # generated replay artifacts, LFS/gitignored
```

## Submission Files

- `submission.csv` — final ranked output
- `submission_metadata.yaml` — fill team/contact/sandbox placeholders before upload
- PDF deck — explain approach, architecture, results, and limitations
- GitHub repo — include code plus required LFS artifacts or rebuild instructions

## Transparency

`rank.py` is deterministic and offline at replay time: no hosted LLMs, no hosted APIs, no network calls. AI assistance used during development is declared in `submission_metadata.yaml`; it is not used by the replay command.
