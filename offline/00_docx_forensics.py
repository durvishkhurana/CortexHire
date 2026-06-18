"""OFFLINE Phase 0 — Docx forensics & rubric reconstruction (WORKFLOW STEP 1).

🛑 STOP POINT: this step reads the organizer ``.docx`` files under ``data/`` and
is owned by the human (they are the authoritative spec). This skeleton wires the
structure; it does NOT read ``data/`` or make any network call.

What this step must produce (exit gate):
  * ``teacher_rubric.md`` — the 0-5 rubric reconstructed near-verbatim from
    ``job_description.docx`` / ``redrob_signals_doc.docx`` / ``submission_spec.docx``
    + the JD's three example judgments, instructed to ignore names/prestige.
  * the resolved open items recorded in ``DECISIONS.md``: exact tier scale,
    Stage-3 Docker network access, salary band, signal-weighting hints.
  * (seed for STEP 2) the distinct ``company`` values from the 50-sample, to
    curate ``artifacts/founding_years.csv`` (left as a documented TODO here).

TODO (blocked on the human-provided docx/data):
  - parse the three docx files (python-docx) and diff normative sentences vs the
    brief summary;
  - hand-label ~50 calibration anchors fixed to the 3 example judgments.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [00_forensics] %(message)s")
log = logging.getLogger(__name__)


def reconstruct_rubric(docx_dir: str, out_path: str) -> None:
    """Reconstruct the teacher rubric from the docx files. TODO (needs data)."""
    raise NotImplementedError(
        "STOP: reading the .docx spec files is a human-owned step. "
        "Confirm access, then implement docx parsing here (python-docx)."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--docx-dir", default="data", help="dir with the .docx spec files")
    ap.add_argument("--out", default="teacher_rubric.md", help="rubric output path")
    args = ap.parse_args()

    log.info(
        "STEP 1 forensics is a 🛑 STOP point — needs the human-provided docx files."
    )
    log.info("expected docx dir: %s ; would write: %s", args.docx_dir, args.out)
    log.info(
        "Resolve in DECISIONS.md: tier scale, Docker network, salary band, signals."
    )
    # Intentionally not reading data/ or the network. Implement once unblocked.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
