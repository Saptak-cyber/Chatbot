"""
BM25 retrieval service.

Maintains an in-memory BM25 index per pdf_id, built lazily on the first
query for that document and cached for subsequent queries.  The cache is
invalidated when a PDF is deleted (call invalidate(pdf_id) from
delete_pdf_chunks).

The corpus for each pdf_id is populated by scrolling all matching points
from Qdrant, so no duplicate storage is needed.
"""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# { pdf_id: (BM25Okapi index, [chunk_dict, ...]) }
_bm25_cache: Dict[str, Tuple[BM25Okapi, List[Dict[str, Any]]]] = {}

# Max chunks scrolled per pdf_id when building the index.
# Generous limit — real PDFs rarely exceed 2 000 semantic chunks.
_MAX_SCROLL_CHUNKS = 2_000


def _tokenize(text: str) -> List[str]:
    """Simple, fast tokeniser: lowercase + split on non-word characters."""
    return re.split(r"\W+", text.lower())


# ── Cache management ──────────────────────────────────────────────────────────

def invalidate(pdf_id: str) -> None:
    """Remove the cached BM25 index for a pdf_id (call after deletion)."""
    removed = _bm25_cache.pop(pdf_id, None)
    if removed:
        logger.info(f"BM25 cache invalidated for pdf_id='{pdf_id}'")


def invalidate_all() -> None:
    """Wipe the entire BM25 cache (useful on startup or for testing)."""
    _bm25_cache.clear()
    logger.info("BM25 cache fully cleared.")


# ── Index building ────────────────────────────────────────────────────────────

async def _build_index(pdf_id: str) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """
    Scroll Qdrant for all chunks belonging to pdf_id, build BM25Okapi index,
    and return (index, chunk_list).  The chunk list mirrors the structure
    returned by query_chunks:  {text, metadata, score}.
    """
    from services.vector_store import get_client, COLLECTION_NAME
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    client = await get_client()

    chunks: List[Dict[str, Any]] = []
    offset = None

    logger.info(f"Building BM25 index for pdf_id='{pdf_id}' — scrolling Qdrant...")

    while True:
        points, offset = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            ),
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for pt in points:
            payload = pt.payload.copy()
            text = payload.pop("text", "")
            payload.pop("parent_text", None)
            payload.pop("parent_id", None)
            chunks.append({"text": text, "metadata": payload, "score": 0.0})

        if offset is None or len(chunks) >= _MAX_SCROLL_CHUNKS:
            break

    if not chunks:
        logger.warning(f"No chunks found in Qdrant for pdf_id='{pdf_id}'")
        return BM25Okapi([[""]]), []

    tokenized_corpus = [_tokenize(c["text"]) for c in chunks]
    index = BM25Okapi(tokenized_corpus)
    logger.info(f"BM25 index built: {len(chunks)} chunks for pdf_id='{pdf_id}'")
    return index, chunks


async def _get_index(pdf_id: str) -> Tuple[BM25Okapi, List[Dict[str, Any]]]:
    """Return cached index, building it on first access."""
    if pdf_id not in _bm25_cache:
        _bm25_cache[pdf_id] = await _build_index(pdf_id)
    return _bm25_cache[pdf_id]


# ── Public retrieval API ──────────────────────────────────────────────────────

async def bm25_search(
    query: str,
    pdf_ids: List[str],
    top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run BM25 retrieval across all active pdf_ids.

    Each pdf_id's index is queried independently; results are merged and
    the global top_k chunks (by BM25 score) are returned, sorted descending.

    Returns chunks in the same dict shape as query_chunks:
      {"text": str, "metadata": dict, "score": float}
    where "score" here is the normalised BM25 score [0, 1].
    """
    if not pdf_ids:
        return []

    tokenized_query = _tokenize(query)

    # Build all indices concurrently (each is a single coroutine but the
    # Qdrant scrolls inside can overlap)
    indices = await asyncio.gather(*[_get_index(pid) for pid in pdf_ids])

    all_results: List[Dict[str, Any]] = []

    for index, chunks in indices:
        if not chunks:
            continue

        raw_scores = index.get_scores(tokenized_query)
        max_score = max(raw_scores) if raw_scores.size else 1.0

        # Pair each chunk with its normalised BM25 score
        scored = [
            (float(raw_scores[i]) / max(max_score, 1e-9), chunks[i])
            for i in range(len(chunks))
        ]
        # Keep only positively-scored chunks (BM25 can return 0 for no match)
        scored = [(s, c) for s, c in scored if s > 0]
        scored.sort(key=lambda x: x[0], reverse=True)

        for score, chunk in scored[:top_k]:
            entry = chunk.copy()
            entry["score"] = score
            all_results.append(entry)

    # Global sort + top_k
    all_results.sort(key=lambda c: c["score"], reverse=True)
    result = all_results[:top_k]

    if result:
        logger.info(
            f"BM25 search returned {len(result)} chunks "
            f"(top score: {result[0]['score']:.3f})"
        )
    else:
        logger.info("BM25 search returned 0 chunks.")

    return result
