"""
Multi-query retrieval — native implementation (no LangChain dependency).

Uses the existing Groq client to generate query variants and the existing
ChromaDB client to search for each variant, then deduplicates results.

Same public API as before:
    multi_query_retrieve(query, pdf_ids, top_k, min_score) → List[Dict]
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any, Optional

from langsmith import traceable

logger = logging.getLogger(__name__)

# Prompt sent to the LLM to generate query variants
_VARIANT_PROMPT = """\
You are a query expansion assistant. Given a user question, generate {n} \
alternative phrasings of the same question that would help retrieve relevant \
passages from a document. Each variant should approach the question from a \
slightly different angle or use different vocabulary. Output ONLY the variants, \
one per line, with no numbering, bullets, or extra commentary.

Original question: {query}"""


@traceable(name="generate_query_variants", run_type="llm")
def _generate_variants(query: str, n: int = 3) -> List[str]:
    """
    Use Groq Llama 3.3 70B to generate `n` alternative phrasings of `query`.
    Returns the variants as a list of strings (original query always included).
    Falls back gracefully to just [query] if generation fails.
    """
    from services.llm import get_client

    try:
        client = get_client()
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": _VARIANT_PROMPT.format(n=n, query=query),
                }
            ],
            temperature=0,
            max_tokens=256,
        )
        raw = response.choices[0].message.content or ""
        variants = [line.strip() for line in raw.splitlines() if line.strip()]
        # Always include the original query
        all_queries = [query] + [v for v in variants if v.lower() != query.lower()]
        logger.info(f"Generated {len(all_queries)} queries (1 original + {len(all_queries)-1} variants)")
        return all_queries[:n + 1]  # cap at n variants + original
    except Exception as e:
        logger.warning(f"Query variant generation failed, using original only: {e}")
        return [query]


@traceable(name="multi_query_retrieve", run_type="retriever")
def multi_query_retrieve(
    query: str,
    pdf_ids: List[str],
    top_k: int = 8,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Retrieve relevant chunks using multi-query expansion.

    1. LLM generates 3 alternative phrasings of `query`.
    2. Each phrasing is embedded and searched against ChromaDB (filtered to
       the given pdf_ids).
    3. Results from all searches are deduplicated by document content.
    4. Chunks below `min_score` are filtered out.
    5. Returns sorted list of chunk dicts (same format as query_chunks()).
    """
    from services.vector_store import query_chunks

    # Generate query variants (falls back to [query] on failure)
    queries = _generate_variants(query, n=3)

    # Run a ChromaDB search for each query variant, collect all chunks
    seen_texts: set = set()
    all_chunks: List[Dict[str, Any]] = []

    for q in queries:
        try:
            chunks = query_chunks(q, pdf_ids, top_k=top_k, min_score=min_score)
        except Exception as e:
            logger.warning(f"query_chunks failed for variant {q!r}: {e}")
            continue

        for chunk in chunks:
            text = chunk["text"]
            if text not in seen_texts:
                seen_texts.add(text)
                all_chunks.append(chunk)

    if not all_chunks:
        logger.info("Multi-query: no chunks found across all variants.")
        return []

    # Re-score every unique chunk against the *original* query so the ranking
    # and confidence display reflects relevance to what the user actually asked.
    from services.embedder import embed_query as _embed_query
    from services.vector_store import get_collection

    try:
        query_embedding = _embed_query(query)
        collection = get_collection()
        where: Optional[Dict] = None
        if pdf_ids:
            where = {"pdf_id": {"$eq": pdf_ids[0]}} if len(pdf_ids) == 1 else {"pdf_id": {"$in": pdf_ids}}

        scored = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(len(all_chunks) + top_k, collection.count()),
            where=where,
            include=["documents", "distances"],
        )
        score_map: Dict[str, float] = {}
        if scored["documents"] and scored["documents"][0]:
            for doc_text, dist in zip(scored["documents"][0], scored["distances"][0]):
                score_map[doc_text] = float(1 - dist)

        for chunk in all_chunks:
            if chunk["text"] in score_map:
                chunk["score"] = score_map[chunk["text"]]
    except Exception as e:
        logger.warning(f"Re-scoring failed, keeping variant scores: {e}")

    # Final filter and sort by score against the original query
    result = [c for c in all_chunks if c.get("score", 0.0) >= min_score]
    result.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    if result:
        logger.info(
            f"Multi-query: {len(result)} unique chunks above min_score={min_score} "
            f"(top score: {result[0]['score']:.3f})"
        )
    return result
