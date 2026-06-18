"""OFFLINE Phase 2 — Text scores as features (WORKFLOW STEPS 5-6).

Fills the OFFLINE columns of ``features.parquet``:
  * ``bm25_score``   — BM25 over the per-candidate career-text document
    (career_history descriptions + summary + headline; NEVER the skills list).
    This part needs NO network/GPU.
  * ``dense_score``  — Qwen3-Embedding-0.6B cosine vs 3-5 JD-intent query
    variants. Offline-only; downloads model weights once (internet ok now), then
    runs GPU/CPU depending on ``--device``. `rank.py` never imports transformers.
  * ``reranker_score`` — Qwen3-Reranker score vs the fixed JD over all 100K.
    Offline-only; GPU strongly recommended (STEP 6).
  * ``fusion_score`` — tuned convex/RRF fusion of dense + bm25 (tune on harness).

This script writes the computed columns back into ``artifacts/features.parquet``
by joining on ``candidate_id`` (overwriting prior values, if any).

Repro notes:
- Stage-3 replay runs with **no network**. This file is never imported at replay
  time; only the produced scores in ``features.parquet`` matter.
- If the model is not present locally, transformers will download it (use
  ``--hf-home`` to pin a cache location). Run once while network is available.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any
from typing import Iterable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import parse  # noqa: E402
from src.parse import CareerF, F  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [02_text] %(message)s")
log = logging.getLogger(__name__)

# HuggingFace model ids (offline-only). Keep in this file so replay deps stay slim.
DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

# JD-intent query variants ("what the JD means", buzzword-free). Tune/extend.
JD_QUERY_VARIANTS: list[str] = [
    "production experience building embeddings-based retrieval and semantic search",
    "ranking and recommendation systems with relevance evaluation (NDCG, MAP)",
    "vector databases and hybrid search at a product company",
    "learning-to-rank and model fine-tuning for search/recommendations",
    "strong Python engineer shipping ML systems to production",
]

# A fixed JD text used by the reranker (query side). Keep it stable and local.
# We derive it from the reconstructed rubric so we don't depend on docx parsing.
JD_TEXT = (
    "Senior AI Engineer (Founding Team). Own retrieval, ranking, and matching systems. "
    "Strong production experience with embeddings-based retrieval, hybrid/vector search, "
    "ranking evaluation (NDCG/MAP), and strong Python shipping ML systems to production. "
    "Prefer product-company shipped search/recsys; down-rank keyword stuffers and "
    "profiles with no production evidence."
)

# Document hygiene: cap extremely long free-text to keep tokenization bounded.
MAX_DOC_CHARS = 6000


def career_document(record: dict[str, Any]) -> str:
    """Per-candidate career text: descriptions + summary + headline.

    Deliberately excludes the skills *list* (anti-keyword-stuffer)."""
    parts: list[str] = []
    for role in record.get(F.CAREER_HISTORY) or []:
        if isinstance(role, dict) and role.get(CareerF.DESCRIPTION):
            parts.append(str(role[CareerF.DESCRIPTION]))
    for key in (F.SUMMARY, F.HEADLINE):
        if record.get(key):
            parts.append(str(record[key]))
    doc = " ".join(parts).strip()
    if len(doc) > MAX_DOC_CHARS:
        doc = doc[:MAX_DOC_CHARS]
    return doc


def _simple_tokenize(text: str) -> list[str]:
    # rank-bm25 ships no tokenizer; we use a deterministic whitespace tokenization
    # with mild normalization. It’s crude but stable and fast for 100K docs.
    return text.lower().split()


def compute_bm25(documents: list[str], queries: list[str]) -> list[float]:
    """BM25 score of each document vs the union of JD-intent queries (offline).

    Real implementation (no network): uses rank-bm25 over tokenized career docs.
    Returns the max query score per document (a simple, tunable fusion)."""
    from rank_bm25 import BM25Okapi

    tokenized = [_simple_tokenize(d) for d in documents]
    bm25 = BM25Okapi(tokenized)
    per_doc_max = [0.0] * len(documents)
    for q in queries:
        scores = bm25.get_scores(_simple_tokenize(q))
        for i, s in enumerate(scores):
            per_doc_max[i] = max(per_doc_max[i], float(s))
    return per_doc_max


def _embed_texts(
    texts: list[str],
    *,
    model_id: str,
    device: str,
    batch_size: int,
    max_length: int,
    hf_home: str | None,
) -> "Any":
    """Return L2-normalized embeddings for a list of texts (offline-only).

    Uses a standard mean-pooling over last_hidden_state masked by attention.
    """
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    model.to(device)
    model.eval()

    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            last = model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).to(last.dtype)  # [B,T,1]
            pooled = (last * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            out.append(pooled.detach().cpu().numpy())
    return np.concatenate(out, axis=0)


def compute_dense_scores(
    documents: list[str],
    queries: list[str],
    *,
    model_id: str,
    device: str,
    batch_size: int,
    max_length: int,
    hf_home: str | None,
) -> list[float]:
    """Dense cosine: max_{q in variants} cos(emb(q), emb(doc))."""
    import numpy as np

    q_emb = _embed_texts(
        queries,
        model_id=model_id,
        device=device,
        batch_size=max(1, batch_size),
        max_length=max_length,
        hf_home=hf_home,
    )  # [Q,H]
    d_emb = _embed_texts(
        documents,
        model_id=model_id,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        hf_home=hf_home,
    )  # [N,H]
    sims = d_emb @ q_emb.T  # cosine since both normalized
    return np.max(sims, axis=1).astype("float32").tolist()


def compute_reranker_scores(
    documents: list[str],
    jd_text: str,
    *,
    model_id: str,
    device: str,
    batch_size: int,
    max_length: int,
    hf_home: str | None,
) -> list[float]:
    """Reranker: score(jd_text, doc) for each doc (offline-only).

    Returns the raw model score (higher = better). Model-specific calibration is
    deferred to the student model + harness.
    """
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if hf_home:
        os.environ["HF_HOME"] = hf_home

    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    scores: list[float] = []
    with torch.no_grad():
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i : i + batch_size]
            enc = tok(
                [jd_text] * len(batch_docs),
                batch_docs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            if logits.ndim == 2 and logits.shape[1] > 1:
                # If it is a 2-class classifier, take the "relevant" logit (last).
                vals = logits[:, -1]
            else:
                vals = logits.view(-1)
            scores.extend(vals.detach().cpu().float().numpy().tolist())
    return np.asarray(scores, dtype="float32").tolist()


def _rank_norm(values: list[float]) -> list[float]:
    """Deterministic rank-based normalization to [0,1]."""
    import numpy as np

    v = np.asarray(values, dtype="float32")
    n = v.shape[0]
    if n == 0:
        return []
    # Stable order: value desc then index asc.
    order = np.lexsort((np.arange(n, dtype="int64"), -v))
    ranks = np.empty(n, dtype="int64")
    ranks[order] = np.arange(n, dtype="int64")
    if n == 1:
        return [1.0]
    return (1.0 - (ranks / (n - 1))).astype("float32").tolist()


def _convex_fusion(a: list[float], b: list[float], alpha: float) -> list[float]:
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b, strict=True)]


def _write_scores_into_features(
    *,
    features_path: str,
    out_path: str,
    candidate_ids: list[str],
    cols: dict[str, list[float]],
) -> None:
    import polars as pl

    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"{features_path} missing. Run offline/01_build_features.py first."
        )

    df_scores = pl.DataFrame({F.CANDIDATE_ID: candidate_ids, **cols})
    base = pl.scan_parquet(features_path)
    merged = base.join(df_scores.lazy(), on=F.CANDIDATE_ID, how="left")
    # Write deterministically (single file parquet).
    merged.collect(streaming=True).write_parquet(out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", default=os.path.join("data", "candidates.jsonl"))
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--features", default=None, help="path to artifacts/features.parquet")
    ap.add_argument("--out", default=None, help="output parquet path (default overwrites)")
    ap.add_argument("--device", default="cuda", help="cuda|cpu (offline only)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--hf-home", default=None, help="HF cache dir for one-time downloads")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--reranker-model", default=DEFAULT_RERANKER_MODEL)
    ap.add_argument(
        "--mode",
        default="step5",
        choices=("step5", "bm25", "dense", "step6_reranker"),
        help="step5 writes bm25+dense+fusion; step6_reranker writes reranker_score",
    )
    ap.add_argument("--alpha", type=float, default=0.5, help="fusion alpha for dense")
    args = ap.parse_args()

    if not os.path.exists(args.candidates):
        log.warning(
            "candidates file not found: %s", args.candidates
        )
        return 0

    features_path = args.features or os.path.join(args.artifacts, "features.parquet")
    out_path = args.out or features_path

    candidate_ids: list[str] = []
    docs: list[str] = []
    for rec in parse.stream_candidates(args.candidates):
        candidate_ids.append(str(rec.get(F.CANDIDATE_ID)))
        docs.append(career_document(rec))
    log.info("built %d career documents", len(docs))

    cols: dict[str, list[float]] = {}

    if args.mode in {"bm25", "step5"}:
        bm25 = compute_bm25(docs, JD_QUERY_VARIANTS)
        cols["bm25_score"] = bm25
        log.info(
            "computed BM25 for %d docs (max=%.3f)",
            len(bm25),
            max(bm25) if bm25 else 0,
        )

    if args.mode in {"dense", "step5"}:
        dense = compute_dense_scores(
            docs,
            JD_QUERY_VARIANTS,
            model_id=args.embedding_model,
            device=args.device,
            batch_size=args.batch_size,
            max_length=args.max_length,
            hf_home=args.hf_home,
        )
        cols["dense_score"] = dense
        log.info(
            "computed dense scores for %d docs (max=%.3f)",
            len(dense),
            max(dense) if dense else 0,
        )

    if args.mode == "step5":
        if "bm25_score" not in cols or "dense_score" not in cols:
            raise RuntimeError("internal: step5 requires bm25 and dense scores computed")
        fusion = _convex_fusion(
            _rank_norm(cols["dense_score"]), _rank_norm(cols["bm25_score"]), args.alpha
        )
        cols["fusion_score"] = fusion
        log.info("computed fusion_score (alpha=%.2f)", args.alpha)

    if args.mode == "step6_reranker":
        rr = compute_reranker_scores(
            docs,
            JD_TEXT,
            model_id=args.reranker_model,
            device=args.device,
            batch_size=max(1, args.batch_size),
            max_length=args.max_length,
            hf_home=args.hf_home,
        )
        cols["reranker_score"] = rr
        log.info(
            "computed reranker_score for %d docs (max=%.3f)",
            len(rr),
            max(rr) if rr else 0,
        )

    if not cols:
        log.warning("nothing to do for mode=%s", args.mode)
        return 0

    log.info("writing %s columns into %s", ", ".join(sorted(cols)), out_path)
    _write_scores_into_features(
        features_path=features_path,
        out_path=out_path,
        candidate_ids=candidate_ids,
        cols=cols,
    )
    log.info("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
