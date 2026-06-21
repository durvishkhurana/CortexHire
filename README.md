# Intelligent Candidate Discovery & Ranking

**Data & AI Challenge** — Redrob / India Runs: rank **100,000** candidates for a fixed **Senior AI Engineer** role and deliver a recruiter-trustworthy **top-100** shortlist (scores + grounded reasoning).

Organizers ask for systems that **understand the JD**, judge **semantic fit beyond keywords**, and integrate **behavioral signals**—with **no prescribed architecture**. At submission time, ranking must replay on **CPU only**, **no network**, **≤5 minutes**, **≤16 GB RAM** (see [`docs/CHALLENGE_BRIEF.md`](docs/CHALLENGE_BRIEF.md)).

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

This runs the replay pipeline and validates with the organizer’s `validate_submission.py`.

---

## What to submit (portal)

| Deliverable | Notes |
|-------------|--------|
| **GitHub repo** | This project; reproducible `submission.csv` |
| **Deck (PPT → PDF)** | What you built, why, how it works |
| **Ranked CSV** | 100 rows + header; `validate_submission.py` clean |

Also: team metadata, **sandbox URL** (≤100 candidates) — mirror [`submission_metadata.yaml`](submission_metadata.yaml) to the bundle’s `submission_metadata_template.yaml`.

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- Unzip the organizer bundle; copy inputs into `data/` (see [`data/README.md`](data/README.md)):
  - `candidates.jsonl` (~465 MB; or `.jsonl.gz` per some bundles)
  - docx spec files + `candidate_schema.json`, `sample_candidates.json`
- **Pre-built artifacts** under `artifacts/` for full quality (feature store, model, reasoning). See [Building artifacts](#building-artifacts-offline).

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Quick start (submission CSV)

1. Copy `candidates.jsonl` (and spec files) into `data/`.
2. Ensure `artifacts/` contains trained model and supporting files (or run the offline pipeline below).
3. Generate the submission:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --artifacts ./artifacts
```

4. Validate and honeypot sanity check:

```bash
python validate_submission.py submission.csv
python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110
```

> `sample_submission.csv` in the organizer bundle is **format-only** (often bad rankings on purpose). Do not copy it.

---

## How it works

| Phase | Where | What |
|--------|--------|------|
| **Offline** | `offline/*.py` | Stream-parse pool → features; BM25 / embeddings / reranker; teacher labels; LightGBM student; reasoning cache |
| **Replay** | `rank.py` | Two-pass streaming JSONL, live honeypot checks, join features, ensemble predict, sort, reasoning, write CSV |

- Challenge & rules: [`docs/CHALLENGE_BRIEF.md`](docs/CHALLENGE_BRIEF.md)  
- Design: [`docs/architecture.md`](docs/architecture.md)  
- Engineering log: [`docs/decisions.md`](docs/decisions.md)  
- **Results (latest run):** [`docs/RESULTS.md`](docs/RESULTS.md)  
- Target / gaps: [`docs/RECOMMENDED_APPROACH.md`](docs/RECOMMENDED_APPROACH.md)

---

## Latest results (summary)

Full tables and reproduce notes: **[`docs/RESULTS.md`](docs/RESULTS.md)**.

| Check | Result |
|--------|--------|
| Full pipeline | `python scripts/run_pipeline_no_api.py` → **valid** `submission.csv` |
| Replay on 100K | **~52 s**, **251** excluded, CPU / no network |
| Honeypots (top 110) | **0** hard |
| Labeling pool | **9,989** pseudo-teacher labels |
| Deployed ranker | LightGBM **lambdarank** (`artifacts/model/`) |
| Tests | **119** passed (`pytest -q`) |

---

## Building artifacts (offline)

Requires `data/candidates.jsonl`. **No hosted LLM API** — default text scores use BM25 + TF-IDF (`--dense-backend lexical`).

**One command (full rebuild + `submission.csv`):**

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

**Manual steps** (optional GPU/HuggingFace for stronger embeddings — not required):

```bash
python offline/01_build_features.py --candidates data/candidates.jsonl --artifacts artifacts
python offline/02_text_scores.py --mode step5 --candidates data/candidates.jsonl --artifacts artifacts --dense-backend lexical --device cpu
python offline/02_text_scores.py --mode step6_proxy --candidates data/candidates.jsonl --artifacts artifacts
python offline/03_teacher_label.py --artifacts artifacts --keep-inconsistent
python offline/04_train_student.py --artifacts artifacts
python offline/05_head_audit_retrain.py --artifacts artifacts
python rank.py --candidates data/candidates.jsonl --out artifacts/submission_final.csv
python offline/06_reasoning.py --candidates data/candidates.jsonl --artifacts artifacts --ranking artifacts/submission_final.csv
```

Set `HF_HOME=./artifacts/hf_cache` when downloading Hugging Face models.

Large binaries (`features.parquet`, `model/`, `reasoning.json`) should ship via **Git LFS** for Stage 3 (no network in sandbox).

---

## Repository layout

```
├── rank.py                 # Entry point (replay)
├── validate_submission.py  # Organizer CSV validator (matches bundle)
├── submission_metadata.yaml
├── requirements.txt
├── src/                    # Parsing, features, model, honeypot, reasoning
├── offline/                # Offline training and feature pipelines
├── tests/
├── scripts/                # e.g. honeypot check
├── docs/                   # Brief, architecture, rubric, recommendations
├── data/                   # Organizer inputs (local only)
└── artifacts/              # Precomputed outputs (LFS)
```

---

## Tests

```bash
python -m pytest -q
```

---

## Portal checklist

- [ ] `submission.csv` — 100 rows + header, passes `validate_submission.py` (**done** — see [`docs/RESULTS.md`](docs/RESULTS.md))
- [ ] **PDF deck** — approach, architecture, results
- [ ] GitHub URL — this repository
- [ ] `submission_metadata.yaml` — aligned with portal + template
- [ ] **Sandbox** — run ranker on ≤100 candidates (HF Spaces, Streamlit, Docker, etc.)

---

## License

MIT — see repository license file if present; update for your team as needed.
