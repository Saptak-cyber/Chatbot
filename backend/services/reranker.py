"""
Reranker service using BAAI/bge-reranker-base via the HuggingFace Inference API.

bge-reranker-base is a cross-encoder that scores (query, passage) pairs and
returns a relevance logit.  Higher logit = more relevant.

We call the HF serverless text-classification endpoint and parse the returned
score to re-rank the candidate pool produced by RRF fusion.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List

import httpx

logger = logging.getLogger(__name__)

HF_RERANKER_MODEL = "BAAI/bge-reranker-base"
_HF_API_BASE = "https://api-inference.huggingface.co/models"

# Shared async HTTP client (created once, reused across requests)
_http_client: httpx.AsyncClient | None = None


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        token = os.getenv("HF_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        # 30 s timeout — HF serverless can be slow on first call (cold start)
        _http_client = httpx.AsyncClient(headers=headers, timeout=30.0)
        logger.info("Reranker HTTP client initialised.")
    return _http_client


# ── Internal API call ─────────────────────────────────────────────────────────

async def _call_reranker_api(pairs: List[List[str]]) -> List[float]:
    """
    POST (query, passage) pairs to HF Inference API for bge-reranker-base.

    The endpoint accepts:
        {"inputs": [["query", "passage1"], ["query", "passage2"], ...]}

    It returns a list of dicts:
        [{"label": "LABEL_0", "score": 0.92}, ...]

    bge-reranker-base only has one label (LABEL_0); the score is the
    sigmoid-activated relevance probability [0, 1].
    """
    url = f"{_HF_API_BASE}/{HF_RERANKER_MODEL}"
    payload = {"inputs": pairs}

    client = _get_http_client()

    try:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        data = response.json()

        # Unwrap: HF returns [[{label, score}]] — one list per pair
        scores: List[float] = []
        for item in data:
            if isinstance(item, list):
                # Each item is [{label, score}, ...] — take the first (and only) score
                scores.append(float(item[0]["score"]))
            elif isinstance(item, dict):
                scores.append(float(item.get("score", 0.0)))
            else:
                scores.append(0.0)

        return scores

    except httpx.HTTPStatusError as e:
        logger.error(
            f"Reranker API HTTP error {e.response.status_code}: {e.response.text[:300]}"
        )
        raise
    except Exception as e:
        logger.error(f"Reranker API call failed: {e}")
        raise


# ── Public API ────────────────────────────────────────────────────────────────

async def rerank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Rerank a pool of chunks using bge-reranker-base and return the top_n.

    Each chunk gets a "reranker_score" field (float, higher = more relevant).
    The "score" field is preserved from the RRF fusion stage.

    Falls back to the original ordering if the API call fails so the
    pipeline always returns something.

    Args:
        query:   The (possibly rewritten) user query.
        chunks:  Candidate pool from RRF fusion (vector + BM25 merged).
        top_n:   How many chunks to return after reranking.

    Returns:
        List of up to top_n chunks, sorted by reranker_score descending.
    """
    if not chunks:
        return []

    # If the pool is already smaller than top_n, skip the API call
    if len(chunks) <= 1:
        for chunk in chunks:
            chunk["reranker_score"] = chunk.get("score", 0.0)
        return chunks[:top_n]

    pairs = [[query, c["text"]] for c in chunks]

    try:
        scores = await asyncio.wait_for(
            _call_reranker_api(pairs),
            timeout=25.0,  # Hard timeout — don't block the whole pipeline
        )
    except asyncio.TimeoutError:
        logger.warning("Reranker API timed out — returning RRF-ranked chunks as fallback.")
        for chunk in chunks:
            chunk["reranker_score"] = chunk.get("score", 0.0)
        return sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)[:top_n]
    except Exception as e:
        logger.warning(f"Reranker failed ({e}) — returning RRF-ranked chunks as fallback.")
        for chunk in chunks:
            chunk["reranker_score"] = chunk.get("score", 0.0)
        return sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)[:top_n]

    # Attach reranker scores
    for chunk, score in zip(chunks, scores):
        chunk["reranker_score"] = score

    ranked = sorted(chunks, key=lambda c: c["reranker_score"], reverse=True)

    if ranked:
        logger.info(
            f"Reranker selected top {min(top_n, len(ranked))} of {len(chunks)} chunks. "
            f"Top reranker score: {ranked[0]['reranker_score']:.4f}"
        )

    return ranked[:top_n]
