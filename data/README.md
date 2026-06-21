# data/ — organizer bundle (local only)

This directory holds files from **`[PUB] India_runs_data_and_ai_challenge`** (unzipped). Large files are **gitignored**; only this README is tracked.

## Expected files

Copy from `India_runs_data_and_ai_challenge/` inside the zip:

| File | Description |
|------|-------------|
| `candidates.jsonl` | 100,000 profiles (~465 MB). Some bundles use `candidates.jsonl.gz` instead—gunzip or open with `gzip` in Python. |
| `sample_candidates.json` | First 50 candidates (pretty JSON) |
| `candidate_schema.json` | JSON Schema for one record |
| `job_description.docx` | Senior AI Engineer JD + **hackathon participant note** (read the closing section) |
| `submission_spec.docx` | CSV format, compute limits, scoring, stages, sandbox rules |
| `redrob_signals_doc.docx` | 23 behavioral signals in `redrob_signals` |
| `README.docx` | Bundle getting-started |
| `sample_submission.csv` | **CSV shape only**—not a model to imitate |
| `validate_submission.py` | Optional copy; repo root validator is **byte-identical** to the bundle |

Text extracts (for grep/diff without python-docx) may live here as:

- `_txt_job_description.txt`
- `_txt_submission_spec.txt`
- `_txt_redrob_signals_doc.txt`
- `_txt_README.txt`

## Getting started

1. Read JD → submission spec → signals doc → schema → skim `sample_candidates.json`.
2. Place `candidates.jsonl` in this folder (or symlink from the unzip path).
3. Run ranker from repo root:

   ```bash
   python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
   ```

4. Validate:

   ```bash
   python validate_submission.py submission.csv
   ```

Official problem statement and portal deliverables: [`docs/CHALLENGE_BRIEF.md`](../docs/CHALLENGE_BRIEF.md).

**Full rebuild + submission (no hosted API):**

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

Latest metrics: [`docs/RESULTS.md`](../docs/RESULTS.md).

## Schema in code

Field names are centralized in `src/parse.py` (`F`, `CareerF`, `SkillF`, `EduF`). Confirm against `candidate_schema.json` after any organizer update.

Tests use `tests/fixtures/synthetic_candidates.jsonl` when the full pool is absent.
