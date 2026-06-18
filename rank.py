"""Deterministic ranking replay (CPU, no network).

Usage:
    python rank.py --candidates ./data/candidates.jsonl --out ./submission.csv

Streams the candidate pool twice: compute reference date (max last_active_date),
then featurize, score, exclude honeypots, select top 100, attach reasoning, and
validate the CSV. Falls back to a rules baseline if model artifacts are missing.
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import heapq
from typing import Any

import validate_submission as vs  # organizer's authoritative validator
from src import features as ft
from src import honeypot as hp
from src import internal_validate as iv
from src import lexicon, parse
from src import reasoning as rz
from src import top20 as t20
from src.parse import F, candidate_sort_key

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [rank] %(levelname)s %(message)s"
)
log = logging.getLogger("rank")

MAX_ROWS = 100

# Artifact paths (relative to --artifacts).
MODEL_SUBDIR = "model"
FEATURES_PARQUET = "features.parquet"
REASONING_JSON = "reasoning.json"
FOUNDING_CSV = "founding_years.csv"
AUDIT_JSON = "audit_flags.json"
ANOMALY_SUBDIR = "anomaly"


# ---------------------------------------------------------------------------
# Artifact loaders (all optional — replay degrades gracefully)
# ---------------------------------------------------------------------------
def _load_model(artifacts_dir: str):
    path = os.path.join(artifacts_dir, MODEL_SUBDIR, "meta.json")
    if not os.path.exists(path):
        log.info("no model artifact; using rules baseline scorer")
        return None
    from src.model import Ensemble

    log.info("loading model ensemble from %s", os.path.dirname(path))
    return Ensemble.load(os.path.join(artifacts_dir, MODEL_SUBDIR))


def _load_offline_features(
    artifacts_dir: str, ids: list[Any] | None
) -> dict[Any, dict[str, float]]:
    """Lazy/memory-mapped parquet join of the OFFLINE feature columns by id.

    Returns ``{candidate_id: {offline_col: value}}`` for ids present in the
    store. Missing ids fall back to NaN (the fallback featurizer)."""
    path = os.path.join(artifacts_dir, FEATURES_PARQUET)
    if not os.path.exists(path):
        log.info("no features.parquet -> OFFLINE columns default to NaN")
        return {}
    import polars as pl

    cols = [F.CANDIDATE_ID] + ft.OFFLINE_FEATURES
    id_set = set(map(str, ids)) if ids is not None else None
    lf = pl.scan_parquet(path).select(cols)
    df = lf.collect()
    out: dict[Any, dict[str, float]] = {}
    for rec in df.iter_rows(named=True):
        cid = str(rec[F.CANDIDATE_ID])
        if id_set is None or cid in id_set:
            out[cid] = {c: rec.get(c) for c in ft.OFFLINE_FEATURES}
    if ids is None:
        log.info("joined OFFLINE features for %d ids (full store)", len(out))
    else:
        log.info("joined OFFLINE features for %d/%d ids", len(out), len(ids))
    return out


def _load_anomaly(artifacts_dir: str):
    path = os.path.join(artifacts_dir, ANOMALY_SUBDIR, "meta.json")
    if not os.path.exists(path):
        return None, None
    from src import anomaly as anom

    clf, names = anom.load_model(os.path.join(artifacts_dir, ANOMALY_SUBDIR))
    return clf, names


def _load_audit_flags(artifacts_dir: str) -> set:
    path = os.path.join(artifacts_dir, AUDIT_JSON)
    if not os.path.exists(path):
        return set()
    import json

    with open(path, encoding="utf-8") as fh:
        return set(map(str, json.load(fh)))


# ---------------------------------------------------------------------------
# Helpers for streaming top-K
# ---------------------------------------------------------------------------
def _neg_cid_key(candidate_id: Any) -> tuple[int, ...]:
    """Key that sorts larger ids *smaller* (for tie-breaking worst-first heaps)."""
    b = str(candidate_id).encode("utf-8", errors="replace")
    return tuple(-x for x in b)


def _push_topk(
    heap: list[tuple[float, tuple[int, ...], str, dict[str, Any]]],
    *,
    k: int,
    score6: float,
    cid: Any,
    rec: dict[str, Any],
) -> None:
    """Maintain a size-k heap of best candidates by (score desc, id asc)."""
    cid_s = str(cid)
    item = (score6, _neg_cid_key(cid_s), cid_s, rec)
    if len(heap) < k:
        heapq.heappush(heap, item)
        return
    # heap[0] is the WORST item under our ordering (smallest score, or tie + larger id)
    if item > heap[0]:
        heapq.heapreplace(heap, item)


# ---------------------------------------------------------------------------
# Scoring — Rules baseline v0 (Solution 1)
# ---------------------------------------------------------------------------
# A transparent, deterministic weighted-sum over the schema features. This is the
# legitimate Submission-1 scorer (PANEL_REVIEW S1: "right Submission-1 insurance
# + source of hard gates"), NOT a trained model. The trained LightGBM ensemble
# auto-replaces it once ``artifacts/model/`` exists (STEP 9+). Weights encode the
# JD's stated priorities: career evidence > skill names; product over services;
# fraud-resistant assessments; behavioral availability; explicit disqualifier
# penalties. NaN features contribute 0 (LightGBM will route NaN; the baseline
# simply ignores absent values — absence is not a penalty, rule #5).
_RULES_V0_WEIGHTS = {
    # --- positive: career evidence & fit ---
    "evidence_density": 1.0,  # JD-cluster hits in CAREER TEXT (anti-stuffer #1)
    "is_product_current": 1.0,  # product company now (JD: product over services)
    "product_tenure_months": 0.004,  # per-month product experience
    "yoe_fit": 1.0,  # soft 5-9 window (ideal 6-8)
    "jd_skill_coverage": 0.4,  # # JD clusters covered by skills
    "assessment_coverage": 0.8,  # JD skills backed by platform assessments
    "max_assessment_score": 0.004,  # best fraud-resistant test score
    "location_fit": 0.4,  # Pune/Noida pref; relocation; no visa sponsor
    "open_to_work": 0.2,
    "has_assessments": 0.1,
    # --- positive: behavioral availability (learned direction) ---
    "recruiter_response_rate": 0.6,
    "interview_completion_rate": 0.4,
    "last_active_recency": 0.01,  # -(months since active); recent => higher
    # --- negative: JD disqualifiers / traps ---
    "claimed_unverified_ratio": -1.2,  # keyword-stuffer tell
    "cv_speech_without_ir_flag": -1.2,  # CV/speech/robotics w/o NLP/IR
    "services_only_flag": -1.0,  # consulting-services-only career
    "no_recent_ic_flag": -1.0,  # hasn't written code in 18+ months
    "title_chasing_flag": -0.8,  # ~1.5y company-hopping for titles
    "yoe_junior_flag": -1.0,  # junior extreme
    "notice_period_days": -0.002,  # long notice => bar gets higher
    "n_hard_flags": -3.0,  # coherence (honeypots already excluded)
    "n_soft_flags": -0.3,
}


def _rules_v0_score(row: dict[str, float]) -> float:
    """Deterministic rules-baseline score from schema features (NaN -> 0)."""
    s = 0.0
    for feat, w in _RULES_V0_WEIGHTS.items():
        v = row.get(feat, math.nan)
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            s += w * float(v)
    return s


def _score_candidates(
    feature_rows: list[dict[str, float]],
    ensemble,
) -> list[float]:
    if ensemble is None:
        return [_rules_v0_score(r) for r in feature_rows]
    import numpy as np

    X = np.array(
        [[r.get(c, math.nan) for c in ensemble.feature_names] for r in feature_rows],
        dtype=float,
    )
    return list(map(float, ensemble.predict(X)))


# ---------------------------------------------------------------------------
# Main replay
# ---------------------------------------------------------------------------
def _stream_input(path: str):
    """Pick the streaming reader by extension.

    The 487 MB pool is JSONL -> :func:`parse.stream_candidates` (line-by-line,
    never loaded whole). The small sandbox sample is a JSON *array*
    (``sample_candidates.json``) -> :func:`parse.stream_json_array`.
    """
    if path.lower().endswith(".jsonl"):
        return parse.stream_candidates(path)
    if path.lower().endswith(".json"):
        return parse.stream_json_array(path)
    return parse.stream_candidates(path)


def run(candidates_path: str, out_path: str, artifacts_dir: str = "artifacts") -> int:
    # PASS 1/2: reference "now" = max(last_active_date) (deterministic).
    log.info("pass 1/2: streaming for reference now: %s", candidates_path)
    now = parse.reference_now(
        rec.get(F.LAST_ACTIVE_DATE) for rec in _stream_input(candidates_path)
    )
    log.info("reference now = %s", now)

    founding = lexicon.load_founding_years(os.path.join(artifacts_dir, FOUNDING_CSV))
    audit_flags = _load_audit_flags(artifacts_dir)
    anomaly_clf, anomaly_feat_names = _load_anomaly(artifacts_dir)

    # Offline features: load mapping once (small: only 4 floats per candidate).
    # We do not need to pre-collect ids; the store already covers the universe.
    offline_feats = _load_offline_features(artifacts_dir, None)

    # PASS 2/2: stream, featurize, score in chunks, keep top-K.
    log.info("pass 2/2: streaming featurize + score + top-100 selection")
    ensemble = _load_model(artifacts_dir)
    cache = rz.load_reasoning_cache(os.path.join(artifacts_dir, REASONING_JSON))

    # We keep a top-K heap slightly larger than 100 only for safety under ties,
    # then apply the exact sort+slice to 100 at the end.
    TOPK_BUFFER = 250
    heap: list[tuple[float, tuple[int, ...], str, dict[str, Any]]] = []

    # Chunked scoring to avoid materializing 100K rows.
    chunk_ids: list[Any] = []
    chunk_recs: list[dict[str, Any]] = []
    chunk_rows: list[dict[str, float]] = []
    chunk_assessments: list[Any] = []

    def _flush_chunk() -> None:
        nonlocal chunk_ids, chunk_recs, chunk_rows, chunk_assessments, heap, n_excluded
        if not chunk_rows:
            return
        from src import anomaly as anom

        if anomaly_clf is not None and anomaly_feat_names is not None:
            anomaly_flags_batch = anom.anomaly_flags(
                anomaly_clf, chunk_rows, anomaly_feat_names
            )
        else:
            anomaly_flags_batch = [False] * len(chunk_rows)

        scores = _score_candidates(chunk_rows, ensemble)
        for cid, rec, row, sc, assessment, anomaly_flag in zip(
            chunk_ids,
            chunk_recs,
            chunk_rows,
            scores,
            chunk_assessments,
            anomaly_flags_batch,
            strict=True,
        ):
            ex = hp.triple_guard(
                False,
                anomaly_flag=anomaly_flag,
                audit_flag=False,
                soft_count=assessment.soft_count,
            )
            if ex:
                n_excluded += 1
                continue
            score6 = round(float(sc), 6)
            _push_topk(heap, k=TOPK_BUFFER, score6=score6, cid=cid, rec=rec)
        chunk_ids = []
        chunk_recs = []
        chunk_rows = []
        chunk_assessments = []

    n_total = 0
    n_excluded = 0
    for rec in _stream_input(candidates_path):
        n_total += 1
        cid = rec.get(F.CANDIDATE_ID)
        row = ft.build_feature_row(rec, now=now, founding_years=founding)
        off = offline_feats.get(str(cid))
        if off:
            row.update({k: off.get(k) for k in ft.OFFLINE_FEATURES})

        assessment = hp.run_consistency_suite(rec, founding_years=founding, now=now)
        if assessment.hard_violation or str(cid) in audit_flags:
            n_excluded += 1
            continue

        chunk_ids.append(cid)
        chunk_recs.append(rec)
        chunk_rows.append(row)
        chunk_assessments.append(assessment)
        if len(chunk_rows) >= 4096:
            _flush_chunk()
    _flush_chunk()

    log.info("processed %d candidates; excluded=%d", n_total, n_excluded)

    # Final exact sort to top 100 (+ top-20 uncertainty penalty).
    feature_by_id: dict[str, dict[str, float]] = {}
    winners = []
    for sc, _neg, cid, rec in heap:
        row = ft.build_feature_row(rec, now=now, founding_years=founding)
        off = offline_feats.get(cid)
        if off:
            row.update({k: off.get(k) for k in ft.OFFLINE_FEATURES})
        feature_by_id[cid] = row
        winners.append((cid, sc))
    winners.sort(key=lambda kv: (-kv[1], candidate_sort_key(kv[0])))
    ranked = t20.apply_top20_penalty(winners[:MAX_ROWS], feature_by_id)
    ranked.sort(key=lambda kv: (-kv[1], candidate_sort_key(kv[0])))
    for i in range(1, len(ranked)):
        cid, sc = ranked[i]
        prev_sc = ranked[i - 1][1]
        if sc > prev_sc:
            ranked[i] = (cid, prev_sc)

    # Build rec_by_id for only the winners (for reasoning + CSV).
    winner_ids = {cid for cid, _ in ranked}
    rec_by_id: dict[Any, dict[str, Any]] = {}
    for rec in _stream_input(candidates_path):
        cid = str(rec.get(F.CANDIDATE_ID))
        if cid in winner_ids:
            rec_by_id[cid] = rec
            if len(rec_by_id) >= len(winner_ids):
                break

    # write CSV
    rows_written = _write_csv(out_path, ranked, rec_by_id, cache, now)
    log.info("wrote %d data rows to %s", rows_written, out_path)

    # 8) self-validate with the ORGANIZER's authoritative validator.
    #    It requires *exactly* 100 data rows (the real submission). For the
    #    <100-candidate sandbox we additionally run our internal validator with
    #    allow_fewer and treat the organizer's row-count complaint as expected.
    org_errors = vs.validate_submission(out_path)
    if rows_written == MAX_ROWS:
        if org_errors:
            for e in org_errors:
                log.error("organizer validator: %s", e)
            log.error("self-validation FAILED (organizer)")
            return 1
        log.info("self-validation PASSED (organizer validate_submission)")
    else:
        log.warning(
            "sandbox: %d (<100) candidates -> organizer's exact-100-row rule "
            "applies only to the full submission; running internal validator",
            rows_written,
        )
        try:
            iv.validate(out_path, expected_rows=MAX_ROWS, allow_fewer=True)
        except iv.ValidationError as exc:
            log.error("self-validation FAILED (internal): %s", exc)
            return 1
        # Surface any organizer issues that are NOT row-count artifacts.
        rowcount_markers = ("data rows", "must appear exactly once")
        non_rowcount = [
            e for e in org_errors if not any(m in e for m in rowcount_markers)
        ]
        for e in non_rowcount:
            log.warning("organizer validator (sandbox note): %s", e)
        log.info("self-validation PASSED (internal allow_fewer; sandbox)")
    return 0


def _rank_eligible(
    ids: list[Any], scores: list[float], excluded: dict[Any, bool]
) -> list[tuple[Any, float]]:
    """Drop excluded ids, sort by rounded score desc + id asc, cap at 100.

    Falls back to including excluded ids only if exclusion would empty the
    output (keeps a valid CSV in the degenerate all-honeypot case)."""
    eligible = [
        (cid, round(sc, 6)) for cid, sc in zip(ids, scores) if not excluded.get(cid)
    ]
    if not eligible:
        log.warning("all candidates excluded; falling back to full set for a valid CSV")
        eligible = [(cid, round(sc, 6)) for cid, sc in zip(ids, scores)]
    eligible.sort(key=lambda kv: (-kv[1], candidate_sort_key(kv[0])))
    return eligible[:MAX_ROWS]


def _write_csv(
    out_path: str,
    ranked: list[tuple[Any, float]],
    rec_by_id: dict[Any, dict[str, Any]],
    cache: dict[str, str],
    now,
) -> int:
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate_id", "rank", "score", "reasoning"])
        for i, (cid, score) in enumerate(ranked, start=1):
            rec = rec_by_id.get(cid, {F.CANDIDATE_ID: cid})
            reasoning = rz.reasoning_for(rec, i, cache, now=now)
            writer.writerow([cid, i, f"{score:.6f}", reasoning])
    return len(ranked)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Deterministic candidate-ranking replay."
    )
    parser.add_argument("--candidates", required=True, help="path to candidates.jsonl")
    parser.add_argument("--out", required=True, help="output submission CSV path")
    parser.add_argument("--artifacts", default="artifacts", help="artifacts directory")
    args = parser.parse_args(argv)
    return run(args.candidates, args.out, args.artifacts)


if __name__ == "__main__":
    raise SystemExit(main())
