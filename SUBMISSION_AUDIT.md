# Submission Audit

Audit date: 2026-06-29

## Readiness

Status: **Ready except manual fields**

Reason: the organizer data was downloaded from the provided public Drive link, full artifacts were rebuilt locally, `submission.csv` validates, and the top-110 honeypot check is clean. Manual team/contact/sandbox metadata is still required.

## Files Present

- Present: `README.md`, `docs/CHALLENGE_BRIEF.md`, `docs/RESULTS.md`, `rank.py`, `validate_submission.py`, `submission_metadata.yaml`, `requirements.txt`, `tests/`, `scripts/`, `offline/`, `src/`
- Present data: `data/candidates.jsonl`, organizer docx files, `candidate_schema.json`, `sample_candidates.json`, `sample_submission.csv`, `submission_metadata_template.yaml`, `validate_submission.py`
- Present artifacts: `artifacts/features.parquet`, `artifacts/model/`, `artifacts/reasoning.json`, `artifacts/audit_flags.json`, `artifacts/anomaly/`, `artifacts/founding_years.csv`, labels/harness/audit outputs
- Present scripts: `scripts/run_pipeline_no_api.py`, `scripts/honeypot_top100_check.py`

## Critical Artifact Status

| Path | Status |
|---|---|
| `data/candidates.jsonl` | Present: 100,000 rows, 465 MB |
| `artifacts/features.parquet` | Present |
| `artifacts/model/meta.json` | Present |
| `artifacts/model/` required model files | Present |
| `artifacts/reasoning.json` | Present |
| `artifacts/audit_flags.json` | Present |
| `artifacts/founding_years.csv` | Present |

`rank.py` uses full artifacts. Evidence: replay logs show `joined OFFLINE features for 100000 ids (full store)` and `loading model ensemble from artifacts/model`.

## Commands Run

| Command | Result |
|---|---|
| `python --version` | Failed: `python` command not found |
| `python3 --version` | `Python 3.9.6` |
| `python3 -m venv .venv` | Created venv, but with unsupported Python 3.9.6 |
| `.venv/bin/python -m pip install -r requirements.txt` | Failed in sandbox before escalation: `Failed to establish a new connection: [Errno 8] nodename nor servname provided, or not known` |
| `.venv/bin/python -m pip install -r requirements.txt` | Failed under Python 3.9.6: `orjson==3.11.9` unavailable for that interpreter |
| `/Users/gurnoor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 --version` | `Python 3.12.13` |
| `/Users/gurnoor/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m venv --clear .venv` | Rebuilt `.venv` with Python 3.12.13 |
| `.venv/bin/python -m pip install -r requirements.txt` | Passed |
| `.venv/bin/python -m pytest -q` | Passed: 122 tests, 1 pytest config warning |
| `.venv/bin/python -m pytest --collect-only -q` | Collected 122 tests |
| `git lfs version` | `git-lfs/3.7.1` |
| `curl ... Drive download` | Downloaded `[PUB] India_runs_data_and_ai_challenge.zip` from the provided Drive link |
| `wc -l data/candidates.jsonl` | 100,000 rows |
| `.venv/bin/python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv` | Passed: valid CSV; runtime 107.03s |
| `.venv/bin/python rank.py --candidates data/candidates.jsonl --out submission.csv --artifacts artifacts` | Passed; runtime 14.64s; 100 rows; 215 excluded |
| `.venv/bin/python validate_submission.py submission.csv` | Passed: `Submission is valid.` |
| `.venv/bin/python scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110` | Passed: `checked top 110 ranks; hard honeypots in set: 0`; runtime 3.32s |

## Ranking / Validation / Honeypot

- Ranking command was run with full artifacts and no fallback.
- `validate_submission.py submission.csv` passes.
- `scripts/honeypot_top100_check.py submission.csv data/candidates.jsonl 110` reports 0 hard honeypots.
- Output has exactly 101 lines including header.

## Rebuild Path

To rebuild from scratch:

```bash
python scripts/run_pipeline_no_api.py --candidates data/candidates.jsonl --out submission.csv
```

## Git LFS

`.gitattributes` now tracks the required large replay artifacts with Git LFS patterns:

- `artifacts/features.parquet`
- `artifacts/model/**`
- `artifacts/reasoning.json`
- `artifacts/audit_flags.json`
- `artifacts/anomaly/**`

`git lfs version` reports `git-lfs/3.7.1`, so the local Git LFS command is installed.

## Remaining Manual Actions

- Fill `submission_metadata.yaml` placeholders: `TODO_TEAM_NAME`, `TODO_PRIMARY_CONTACT_NAME`, `TODO_PRIMARY_CONTACT_EMAIL`, `TODO_PHONE`, `TODO_TEAM_MEMBER_NAME`, `TODO_TEAM_MEMBER_EMAIL`, `TODO_SANDBOX_LINK`.
- Confirm portal upload filename/team ID and sandbox link.
- Convert the deck/PPT to PDF and upload it with this repo and `submission.csv`.

## Risk Assessment

- Medium: hidden labels are unavailable; internal harness is pseudo-teacher based.
- Medium: reasoning is deterministic/composer-based, not a frontier LLM review.
- Low: replay is fast, validates, and has 0 hard honeypots in top 110.
- Low: tests pass under Python 3.12.13 with pinned requirements.

Recommended next step: fill manual metadata and produce the PDF deck; keep `submission.csv` from the pointwise-selected run unless a stronger validated experiment beats it.
