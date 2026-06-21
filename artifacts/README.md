# artifacts/ — precomputed outputs of the OFFLINE pipeline

`rank.py` reads from here at replay time; it never writes intelligence here.
Built by `offline/*.py` or **`scripts/run_pipeline_no_api.py`**.

**Benchmarks for the latest full build:** [`../docs/RESULTS.md`](../docs/RESULTS.md).

Large binaries are gitignored and intended to ship via **Git LFS** (Stage-3 Docker has no network).

| Artifact | Produced by | Typical size / notes |
|---|---|---|
| `founding_years.csv` | manual / forensics | small; tracked in git |
| `features.parquet` | `offline/01` + `offline/02` | ~3 MB; 100K × 46 cols; BM25/TF-IDF/reranker columns |
| `labeling_pool.parquet` | `offline/03` step 7 | ~10K ids |
| `teacher_labels.parquet` | `offline/03` step 9 | ~10K rows |
| `teacher_labels_audit.parquet` | `offline/05` | augmented labels |
| `teacher_pilot_stats.json` | `offline/03` step 8 | consistency metrics |
| `model/`, `model_lambdarank/` | `offline/04`–`05` | LightGBM ensembles |
| `model/selection.json` | `offline/04` | deployed head (**lambdarank**) |
| `harness_results.json` | `offline/04` | pooled NDCG/MAP/composite |
| `anomaly/` | `offline/04` | IsolationForest |
| `audit_flags.json` | `offline/05` | e.g. **56** ids (latest run) |
| `audit_disagreements.json` | `offline/05` | pairwise audit sample |
| `reasoning.json` | `offline/06` | 100 verified strings |
| `submission_final.csv` | `rank.py` (pipeline) | ranked top 100 |
| `profiling_notes.md`, `*_counts.csv` | `offline/00b` | pool stats |

`rank.py` degrades when artifacts are missing (rules baseline + composer reasoning) so tests and sandbox samples still run.

**Optional (not required for no-API path):**

| `hf_cache/`, `bge-small-int8/` | HuggingFace / fallback embedder | only if using `--dense-backend hf` |
