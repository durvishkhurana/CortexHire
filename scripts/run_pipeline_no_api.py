#!/usr/bin/env python3
"""End-to-end offline + replay pipeline (no hosted LLM APIs).

Uses lexical TF-IDF dense scores + BM25 + proxy reranker. Requires only
``requirements.txt`` (CPU). Optional HF models are not used.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=ROOT)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default="data/candidates.jsonl")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--out", default="submission.csv")
    ap.add_argument("--skip-features", action="store_true")
    args = ap.parse_args()

    py = sys.executable
    cand = args.candidates
    art = args.artifacts

    if not args.skip_features:
        run([py, "offline/01_build_features.py", "--candidates", cand, "--artifacts", art])

    run(
        [
            py,
            "offline/02_text_scores.py",
            "--candidates",
            cand,
            "--artifacts",
            art,
            "--mode",
            "step5",
            "--dense-backend",
            "lexical",
            "--device",
            "cpu",
        ]
    )
    run(
        [
            py,
            "offline/02_text_scores.py",
            "--candidates",
            cand,
            "--artifacts",
            art,
            "--mode",
            "step6_proxy",
            "--device",
            "cpu",
        ]
    )
    run([py, "offline/03_teacher_label.py", "--artifacts", art, "--keep-inconsistent"])
    run([py, "offline/04_train_student.py", "--artifacts", art])
    run([py, "offline/05_head_audit_retrain.py", "--artifacts", art])

    sub_final = str(Path(art) / "submission_final.csv")
    run([py, "rank.py", "--candidates", cand, "--out", sub_final, "--artifacts", art])
    run(
        [
            py,
            "offline/06_reasoning.py",
            "--candidates",
            cand,
            "--artifacts",
            art,
            "--ranking",
            sub_final,
        ]
    )
    run([py, "rank.py", "--candidates", cand, "--out", args.out, "--artifacts", art])
    run([py, "validate_submission.py", args.out])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
