# artifacts/ — precomputed outputs of the OFFLINE pipeline

`rank.py` reads from here at replay time; it never writes intelligence here.
Built by `offline/*.py`. Large binaries are gitignored and intended to ship via
**Git LFS** (Stage-3 Docker may have no network — confirm in forensics).

| Artifact | Produced by | Tracked in git? |
|---|---|---|
| `founding_years.csv` | `offline/00_docx_forensics.py` / `offline/02_text_scores.py` | **yes** (small text, reviewable) |
| `features.parquet` | `offline/01_build_features.py` | no → Git LFS |
| `model/` (LightGBM seed ensemble + feature names) | `offline/04_train_student.py` | no → Git LFS |
| `reasoning.json` | `offline/06_reasoning.py` | no → Git LFS |
| `bge-small-int8/` (bundled fallback embedder) | downloaded offline (TODO) | no → Git LFS |

`rank.py` degrades gracefully when any of these are absent: it falls back to a
documented deterministic placeholder scorer and the composer reasoning fallback,
so the end-to-end pipeline runs on the synthetic fixture even before the offline
artifacts exist.
