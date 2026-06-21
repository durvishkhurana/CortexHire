# Challenge brief (official alignment)

This document mirrors the **organizer bundle** (`[PUB] India_runs_data_and_ai_challenge`) and portal copy for **Intelligent Candidate Discovery & Ranking** (Data & AI Challenge). Use it to keep README, deck, and `submission_metadata.yaml` consistent with what judges expect.

---

## Problem (in plain language)

Recruiters review hundreds of profiles and still miss strong fits because **keyword filters cannot see what matters**. You must build an AI system that ranks candidates the way a **strong recruiter** would: understanding the role, reading full career context, and integrating behavioral signals—not matching buzzwords.

**Fixed task:** rank **100,000** candidates in `candidates.jsonl` for the released **Senior AI Engineer — Founding Team** JD (`job_description.docx` / `data/_txt_job_description.txt`).

**Dataset traps (organizer-documented):** keyword stuffers, plain-language Tier-5 fits without AI buzzwords, behavioral twins, and **~80 honeypots** with subtly impossible profiles. Honeypot rate **>10% in your top 100** → Stage 3 disqualification.

---

## What your solution must do

| Requirement | Detail |
|-------------|--------|
| **Deep job understanding** | Interpret nuanced JD intent (shipper vs researcher, production retrieval, disqualifiers)—not keyword overlap. |
| **Contextual relevance** | Career evidence and role fit over skill-list stuffing (see JD participant note). |
| **Signal integration** | Profile fields + career metadata + **23 `redrob_signals`** (activity, response rates, notice period, etc.). |
| **Output** | **Top 100** ranked CSV: `candidate_id`, `rank`, `score`, `reasoning` (reasoning strongly recommended for Stage 4). |
| **Architecture** | **No prescribed stack**—semantic search, LLM ranking, embeddings, hybrid scoring are all allowed if you meet replay constraints. |

---

## What to submit (portal + repo)

1. **GitHub repo** — complete, reproducible code; single command produces `submission.csv` (see `submission_spec.docx` §10.3).
2. **Deck (PPT → PDF)** — approach: what you built, why, how it works (for human review; not validated by `validate_submission.py`).
3. **Ranked CSV** — exactly **100** data rows + header; filename per team ID on portal (format in `submission_spec.docx` §2).

Also required at upload (see `submission_metadata_template.yaml` in the bundle):

- Team name, contacts, member list  
- **GitHub URL**  
- **Sandbox link** (≤100 candidates; HF Spaces, Streamlit, Docker, Colab, Binder, etc.—§10.5)  
- AI tools declaration (honest; not penalized)  
- Compute summary  
- Optional: ≤200-word methodology summary  

**Repo root:** copy template → `submission_metadata.yaml` and fill fields to match the portal.

---

## Ranking step constraints (Stage 3 reproduction)

From `submission_spec.docx` §3 (must hold when producing the CSV):

| Constraint | Limit |
|------------|--------|
| Wall-clock | ≤ **5 minutes** |
| RAM | ≤ **16 GB** |
| Compute | **CPU only** (no GPU during ranking) |
| Network | **Off** (no hosted LLM/API calls during ranking) |
| Disk (intermediate) | ≤ **5 GB** |

Pre-computation (embeddings, training, labeling) may take longer offline; the **ranking command** must finish inside the box above.

**Reproduce command (this repo):**

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv
```

---

## How submissions are scored (after close)

Composite (hidden ground truth):

`0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10`

Tiebreaks: P@5 → P@10 → earlier submission time.

**P@10 “relevant”** = tier **≥ 3** on the organizer **0–5** relevance scale (honeypots = tier **0**). Rubric language is reconstructed in [`teacher_rubric.md`](./teacher_rubric.md) from organizer docx.

There is **no live leaderboard**; at most **3** submissions; last valid submission counts.

---

## Evaluation stages (summary)

| Stage | Focus |
|-------|--------|
| 1 | CSV format (`validate_submission.py`) |
| 2 | Composite vs hidden labels |
| 3 | Reproduce ranking in Docker; honeypot rate in top 100 |
| 4 | Reasoning quality (6 checks), methodology, git history |
| 5 | Defend-your-work interview (top finalists) |

---

## Bundle contents (unzipped)

Path inside zip: `India_runs_data_and_ai_challenge/`

| File | Purpose |
|------|---------|
| `candidates.jsonl` | 100,000 profiles (~465 MB; some bundles ship `.jsonl.gz` instead) |
| `sample_candidates.json` | First 50 candidates (schema inspection) |
| `candidate_schema.json` | JSON Schema for records |
| `job_description.docx` | JD + hackathon participant note |
| `submission_spec.docx` | Rules, metrics, stages, sandbox |
| `redrob_signals_doc.docx` | 23 behavioral signals |
| `README.docx` | Getting started |
| `sample_submission.csv` | **Format only**—not a good ranking (often keyword-stuffer examples) |
| `validate_submission.py` | Local format check (byte-identical to repo root copy) |
| `submission_metadata_template.yaml` | Portal/repo metadata template |

Place working copies under `data/` (gitignored). Text extracts for docx live in `data/_txt_*.txt` for search/diff.

---

## Reasoning column (Stage 4)

Sampled rows are checked for: specific facts, JD connection, honest concerns, no hallucination, variation (not templated), rank-consistent tone. See `submission_spec.docx` §3.

---

## Relation to this codebase

Implementation design: [`architecture.md`](./architecture.md). Decisions log: [`decisions.md`](./decisions.md). Recommended evolution: [`RECOMMENDED_APPROACH.md`](./RECOMMENDED_APPROACH.md). **Measured runs:** [`RESULTS.md`](./RESULTS.md).

---

## Our solution status (2026-06-21)

| Portal requirement | Status |
|--------------------|--------|
| Reproducible `submission.csv` | ✅ `validate_submission.py` passes |
| CPU / no-network replay | ✅ ~52 s on 100K |
| Honeypot sanity (top 110) | ✅ 0 hard honeypots |
| GitHub + deck + sandbox | ⏳ fill `submission_metadata.yaml` |

Details: [`RESULTS.md`](./RESULTS.md).
