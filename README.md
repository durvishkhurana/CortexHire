# Intelligent Candidate Ranking (India Runs · Track 1)

Rank **100,000** candidate profiles against a fixed **Senior AI Engineer** job description and produce a **top-100 CSV** with scores and short, grounded reasoning. The ranking step is a **deterministic offline replay**: CPU-only, no network, within organizer limits (≤5 minutes, ≤16 GB RAM).

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

The command above validates the output with the organizer’s `validate_submission.py` before exit.

---

## Prerequisites

- **Python 3.10+** (3.11 recommended)
- Place organizer files under `data/` (see [`data/README.md`](data/README.md)):
  - `candidates.jsonl` (~465 MB, not committed)
- **Pre-built artifacts** under `artifacts/` (feature store, trained model, reasoning cache). See [Building artifacts](#building-artifacts-offline).

```bash
python -m venv .venv
.venv\Scripts\activate   # Windows
pip install -r requirements.txt
```

---

## Quick start (submission)

1. Copy `candidates.jsonl` into `data/`.
2. Ensure `artifacts/` contains the trained model and supporting files (or run the offline pipeline below).
3. Generate the submission:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv --artifacts ./artifacts
```

4. Optional sanity check:

```bash
python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110
```

---

## How it works

| Phase | Where | What |
|--------|--------|------|
| **Offline** | `offline/*.py` | Stream-parse pool → features; BM25 / embeddings / reranker scores; pseudo-teacher labels; LightGBM student; reasoning cache |
| **Replay** | `rank.py` | Two-pass streaming JSONL, live honeypot checks, join features, ensemble predict, sort, attach reasoning, write CSV |

Design details: [`docs/architecture.md`](docs/architecture.md). Engineering log: [`docs/decisions.md`](docs/decisions.md).

---

## Building artifacts (offline)

Requires `data/candidates.jsonl`. GPU optional for text embeddings (`requirements-offline-gpu.txt`).

```bash
python offline/01_build_features.py --candidates data/candidates.jsonl --artifacts artifacts
python offline/02_text_scores.py --mode step5 --candidates data/candidates.jsonl --artifacts artifacts --device cuda
python offline/02_text_scores.py --mode step6_reranker --candidates data/candidates.jsonl --artifacts artifacts --device cuda
python offline/03_teacher_label.py --artifacts artifacts
python offline/04_train_student.py --artifacts artifacts
python offline/05_head_audit_retrain.py --artifacts artifacts
python rank.py --candidates data/candidates.jsonl --out artifacts/submission_final.csv
python offline/06_reasoning.py --candidates data/candidates.jsonl --artifacts artifacts --ranking artifacts/submission_final.csv
```

Set `HF_HOME=./artifacts/hf_cache` when downloading Hugging Face models so weights stay on disk with the project.

Large binaries (`features.parquet`, `model/`, `reasoning.json`) are intended for **Git LFS** at submission time.

---

## Repository layout

```
├── rank.py                 # Entry point (replay)
├── validate_submission.py  # Organizer CSV validator
├── submission_metadata.yaml
├── requirements.txt
├── src/                    # Parsing, features, model, honeypot, reasoning
├── offline/                # Offline training and feature pipelines
├── tests/
├── scripts/                # Utilities (e.g. honeypot check)
├── docs/                   # Architecture, decisions, rubric
├── data/                   # candidates.jsonl (local only)
└── artifacts/              # Precomputed outputs (LFS)
```

---

## Tests

```bash
python -m pytest -q
```

---

## Portal checklist

- [ ] `submission.csv` — exactly 100 rows + header, passes `validate_submission.py`
- [ ] GitHub URL — this repository
- [ ] `submission_metadata.yaml` — team, sandbox URL, AI-tools declaration
- [ ] Sandbox — hosted demo on ≤100 candidates (Hugging Face Spaces, Streamlit, or Docker)

---

## License

MIT — see repository license file if present; update for your team as needed.
