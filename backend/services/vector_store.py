"""
Qdrant Cloud vector store client for storing and querying PDF chunks.

Provides two retrieval modes:
  - query_chunks:        Pure vector (cosine) retrieval with dynamic-k cutoff.
  - query_chunks_hybrid: BM25 + vector → RRF fusion → BGE-reranker-base rerank.
"""
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    PayloadSchemaType,
)
import uuid
import os
from typing import List, Dict, Any, Optional
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "pdf_chunks"
VECTOR_SIZE = 384  # BGE-small-en-v1.5 embedding dimension (same as all-MiniLM-L6-v2)

_client: Optional[AsyncQdrantClient] = None


async def get_client():
    """Return a singleton Qdrant client and ensure collection exists."""
    global _client
    if _client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in environment variables"
            )
        
        logger.info(f"Initializing Qdrant client at '{QDRANT_URL}'...")
        _client = AsyncQdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Create collection if it doesn't exist
        collections = (await _client.get_collections()).collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection '{COLLECTION_NAME}'...")
            await _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            
            # Create payload index for pdf_id to enable efficient filtering
            from qdrant_client.models import PayloadSchemaType
            await _client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="pdf_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created payload index for 'pdf_id' field")
        
        collection_info = await _client.get_collection(COLLECTION_NAME)
        logger.info(
            f"Qdrant collection '{COLLECTION_NAME}' ready. "
            f"Count: {collection_info.points_count}"
        )
    
    return _client


async def add_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and store chunks in Qdrant. Returns number of chunks stored."""
    from services.embedder import embed_texts
    import asyncio

    client = await get_client()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch embed
    embeddings = await asyncio.to_thread(embed_texts, texts)

    # Create points for Qdrant
    points = []
    for text, embedding, metadata in zip(texts, embeddings, metadatas):
        point_id = str(uuid.uuid4())
        points.append(
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "text": text,
                    # Hierarchical fields (parent_id, parent_text) are included
                    # via **metadata if present; legacy chunks without them work fine.
                    **metadata,
                },
            )
        )

    # Upload to Qdrant
    await client.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.info(f"Stored {len(chunks)} chunks in Qdrant.")
    return len(chunks)


@traceable(name="query_chunks", run_type="retriever")
async def query_chunks(
    query: str,
    pdf_ids: List[str],
    top_k: int = 10,
    min_score: float = 0.20,
    dynamic_k: bool = True,
) -> List[Dict[str, Any]]:
    """
    Query Qdrant for the most relevant chunks filtered to the given pdf_ids.

    Chunks below `min_score` (cosine similarity) are discarded so the LLM only
    receives genuinely relevant context. If nothing clears the threshold the
    caller should treat this as an out-of-scope query and refuse immediately
    without calling the LLM.

    Returns chunks sorted by cosine similarity (highest first).
    """
    from services.embedder import embed_query
    import asyncio

    client = await get_client()

    collection_info = await client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        return []

    query_embedding = await asyncio.to_thread(embed_query, query)

    # Build filter for pdf_ids
    query_filter = None
    if pdf_ids:
        if len(pdf_ids) == 1:
            query_filter = Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_ids[0]))]
            )
        else:
            query_filter = Filter(
                must=[FieldCondition(key="pdf_id", match=MatchAny(any=pdf_ids))]
            )

    try:
        search_result = (await client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
        )).points

        logger.info(f"Qdrant search returned {len(search_result)} results")
        if search_result:
            logger.info(f"Top result score: {search_result[0].score:.4f}")
            
            if dynamic_k:
                top_score = search_result[0].score
                filtered_result = [search_result[0]]
                for i in range(1, len(search_result)):
                    curr_score = search_result[i].score
                    prev_score = search_result[i-1].score
                    
                    # Cutoff heuristics:
                    # 1. Sudden drop > 0.05 between consecutive chunks
                    # 2. Score drops below 80% of the top score
                    score_drop = prev_score - curr_score
                    relative_score = curr_score / top_score if top_score > 0 else 0
                    
                    # if score_drop > 0.1 or relative_score < 0.80:
                    if relative_score < 0.80:
                        logger.info(
                            f"Dynamic k cutoff triggered at index {i}. "
                            f"Drop: {score_drop:.3f}, Rel: {relative_score:.2f} "
                            f"(Top: {top_score:.3f})"
                        )
                        break
                    filtered_result.append(search_result[i])
                
                search_result = filtered_result
                logger.info(f"After dynamic k cutoff, retained {len(search_result)} results")

    except Exception as e:
        logger.error(f"Qdrant query error: {e}")
        return []

    chunks = []
    for scored_point in search_result:
        payload = scored_point.payload.copy()
        text = payload.pop("text", "")
        # Drop any leftover hierarchical fields so they don't pollute metadata
        payload.pop("parent_text", None)
        payload.pop("parent_id", None)

        chunks.append({
            "text": text,
            "metadata": payload,
            "score": float(scored_point.score),
        })

    if chunks:
        logger.info(
            f"Returning {len(chunks)} chunks above min_score={min_score} "
            f"(top score: {chunks[0]['score']:.3f})"
        )
    else:
        logger.info(f"No chunks cleared min_score={min_score} — query is out of scope.")

    return chunks


async def delete_pdf_chunks(pdf_id: str) -> None:
    """Delete all chunks belonging to a given pdf_id and bust the BM25 cache."""
    from services.bm25_store import invalidate as bm25_invalidate

    client = await get_client()

    # Delete points matching the pdf_id filter
    await client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
        ),
    )

    # Bust the in-memory BM25 index so stale data is never served
    bm25_invalidate(pdf_id)

    logger.info(f"Deleted all chunks for pdf_id='{pdf_id}' (BM25 cache invalidated).")


# ── Hybrid retrieval ──────────────────────────────────────────────────────────

def _rrf_merge(
    list_a: List[Dict[str, Any]],
    list_b: List[Dict[str, Any]],
    k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion of two ranked chunk lists.

    Deduplication key: first 120 characters of the chunk text (robust to
    minor payload differences between the vector and BM25 paths).

    Formula (original RRF paper, Cormack et al. 2009):
        score(d) = Σ  1 / (k + rank_i(d))   for each list i that contains d

    Args:
        list_a: Vector retrieval results (sorted by cosine score desc).
        list_b: BM25 retrieval results (sorted by BM25 score desc).
        k:      RRF constant — higher k smooths rank differences.

    Returns:
        Merged list sorted by RRF score descending.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map: Dict[str, Dict[str, Any]] = {}

    for ranked_list in (list_a, list_b):
        for rank, chunk in enumerate(ranked_list, start=1):
            key = chunk["text"][:120]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunk_map:
                chunk_map[key] = chunk

    merged = sorted(chunk_map.keys(), key=lambda key: rrf_scores[key], reverse=True)
    result = []
    for key in merged:
        entry = chunk_map[key].copy()
        entry["rrf_score"] = rrf_scores[key]
        result.append(entry)

    logger.info(
        f"RRF merge: {len(list_a)} vector + {len(list_b)} BM25 → {len(result)} unique chunks"
    )
    return result


@traceable(name="query_chunks_hybrid", run_type="retriever")
async def query_chunks_hybrid(
    query: str,
    pdf_ids: List[str],
    vector_k: int = 20,
    bm25_k: int = 20,
    rerank_n: int = 10,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Hybrid retrieval pipeline:
        1. Vector search (Qdrant cosine, top vector_k)
        2. BM25 search (in-memory rank-bm25, top bm25_k)
        3. Reciprocal Rank Fusion → union pool
        4. BGE-reranker-base reranking → top rerank_n chunks
        5. Dynamic-k cutoff on reranker scores (relative threshold 0.80)

    Falls back to pure vector results if BM25 or the reranker fails, so
    the pipeline always returns something useful.

    Returns chunks in the same dict shape as query_chunks:
        {"text": str, "metadata": dict, "score": float, "reranker_score": float}
    """
    import asyncio
    from services.bm25_store import bm25_search
    from services.reranker import rerank_chunks

    # ── Stage 1: parallel retrieval ───────────────────────────────────────────
    vector_task = query_chunks(
        query, pdf_ids, top_k=vector_k, min_score=min_score, dynamic_k=False
    )
    bm25_task = bm25_search(query, pdf_ids, top_k=bm25_k)

    try:
        vector_chunks, bm25_chunks = await asyncio.gather(vector_task, bm25_task)
    except Exception as e:
        logger.error(f"Parallel retrieval error: {e} — falling back to vector-only")
        vector_chunks = await query_chunks(
            query, pdf_ids, top_k=vector_k, min_score=min_score, dynamic_k=False
        )
        bm25_chunks = []

    logger.info(
        f"Hybrid retrieval: {len(vector_chunks)} vector chunks, "
        f"{len(bm25_chunks)} BM25 chunks"
    )

    if not vector_chunks and not bm25_chunks:
        logger.info("Both retrievers returned 0 results — out of scope.")
        return []

    # ── Stage 2: RRF fusion ───────────────────────────────────────────────────
    pool = _rrf_merge(vector_chunks, bm25_chunks)

    # ── Stage 3: Reranking ────────────────────────────────────────────────────
    try:
        final = await rerank_chunks(query, pool, top_n=rerank_n)
    except Exception as e:
        logger.warning(f"Reranker failed ({e}) — using top {rerank_n} RRF chunks.")
        final = pool[:rerank_n]
        for chunk in final:
            chunk["reranker_score"] = chunk.get("rrf_score", 0.0)

    # ── Stage 4: Dynamic-k cutoff on reranker scores ────────────────────────────
    # Discard chunks whose reranker score falls below 80% of the top score.
    # This mirrors the cosine dynamic-k but operates on the higher-quality
    # reranker signal, giving a more precise tail prune.
    if len(final) > 1:
        top_reranker_score = final[0]["reranker_score"]
        cutoff_idx = len(final)  # default: keep all
        for i in range(1, len(final)):
            relative = (
                final[i]["reranker_score"] / top_reranker_score
                if top_reranker_score > 0
                else 0.0
            )
            if relative < 0.80:
                cutoff_idx = i
                logger.info(
                    f"Dynamic-k cutoff on reranker scores at index {i}: "
                    f"score={final[i]['reranker_score']:.4f}, "
                    f"rel={relative:.2f} (top={top_reranker_score:.4f})"
                )
                break
        final = final[:cutoff_idx]

    logger.info(
        f"query_chunks_hybrid → {len(final)} final chunks "
        + (f"(top reranker score: {final[0]['reranker_score']:.4f})" if final else "(empty)")
    )
    return final


async def get_pdf_chunk_count(pdf_id: str) -> int:
    """Count chunks stored for a given pdf_id."""
    client = await get_client()
    
    # Use scroll to count points with the given pdf_id
    result = await client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
        ),
        limit=1,  # We only need the count, not the actual points
        with_payload=False,
        with_vectors=False,
    )
    
    # Scroll returns (points, next_page_offset)
    # For accurate count, we'd need to scroll through all pages
    # But for a quick count, we can use count API if available
    # Let's use a more accurate approach
    count = 0
    offset = None
    
    while True:
        points, offset = await client.scroll(
            collection_name=COLLECTION_NAME,
            scroll_filter=Filter(
                must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
            ),
            limit=100,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        count += len(points)
        if offset is None:
            break
    
    return count
