# data/ — raw inputs (gitignored)

This directory holds the organizer-provided inputs. **Everything here except this
README and `.gitkeep` is gitignored** because the real dataset is ~465 MB and the
`.docx` files carry the rubric/spec language.

Expected contents (provided by the organizers — **not** committed):

- `candidates.jsonl` — the 100,000 candidate profiles (~465 MB).
- `sample_candidates.json` — the 50-candidate sample for profiling.
- `job_description.docx` — the fixed Senior AI Engineer JD (+ hackathon notes).
- `redrob_signals_doc.docx` — behavioral-signal documentation.
- `submission_spec.docx` — submission format + relevance tier scale.

> 🛑 **STOP point:** reading/interpreting anything in this folder is owned by the
> human (it is the authoritative spec). The scaffolding is built against a
> SYNTHETIC fixture at `tests/fixtures/synthetic_candidates.jsonl` and an
> assumed schema centralized in `src/parse.py`. Confirm the real field names
> against these files before the offline run.
