"""
ChromaDB vector store client for storing and querying PDF chunks.
Uses cosine similarity with metadata filtering by pdf_id.
"""
import chromadb
import uuid
import os
from typing import List, Dict, Any, Optional
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
COLLECTION_NAME = "pdf_chunks"

_client: chromadb.PersistentClient | None = None
_collection = None


def get_collection():
    """Return a singleton ChromaDB collection."""
    global _client, _collection
    if _client is None:
        logger.info(f"Initializing ChromaDB at '{CHROMA_PATH}'...")
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(f"ChromaDB collection '{COLLECTION_NAME}' ready. Count: {_collection.count()}")
    return _collection


def add_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and store chunks in ChromaDB. Returns number of chunks stored."""
    from services.embedder import embed_texts

    collection = get_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]
    ids = [str(uuid.uuid4()) for _ in chunks]

    # Batch embed
    embeddings = embed_texts(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )

    logger.info(f"Stored {len(chunks)} chunks in ChromaDB.")
    return len(chunks)


@traceable(name="query_chunks", run_type="retriever")
def query_chunks(
    query: str,
    pdf_ids: List[str],
    top_k: int = 8,
    min_score: float = 0.20,
) -> List[Dict[str, Any]]:
    """
    Query ChromaDB for the most relevant chunks filtered to the given pdf_ids.

    Chunks below `min_score` (cosine similarity) are discarded so the LLM only
    receives genuinely relevant context. If nothing clears the threshold the
    caller should treat this as an out-of-scope query and refuse immediately
    without calling the LLM.

    Returns chunks sorted by cosine similarity (highest first).
    """
    from services.embedder import embed_query

    collection = get_collection()
    total_count = collection.count()

    if total_count == 0:
        return []

    n_results = min(top_k, total_count)

    query_embedding = embed_query(query)

    # Build where filter
    where: Optional[Dict] = None
    if pdf_ids:
        if len(pdf_ids) == 1:
            where = {"pdf_id": {"$eq": pdf_ids[0]}}
        else:
            where = {"pdf_id": {"$in": pdf_ids}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where,
            include=["documents", "metadatas", "distances"],
        )
    except Exception as e:
        logger.error(f"ChromaDB query error: {e}")
        return []

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            score = float(1 - dist)  # cosine similarity
            if score < min_score:
                continue
            chunks.append({"text": doc, "metadata": meta, "score": score})

    # Sort by score descending
    chunks.sort(key=lambda x: x["score"], reverse=True)

    if chunks:
        logger.info(
            f"Returning {len(chunks)} chunks above min_score={min_score} "
            f"(top score: {chunks[0]['score']:.3f})"
        )
    else:
        logger.info(f"No chunks cleared min_score={min_score} — query is likely out of scope.")

    return chunks


def delete_pdf_chunks(pdf_id: str) -> None:
    """Delete all chunks belonging to a given pdf_id."""
    collection = get_collection()
    collection.delete(where={"pdf_id": {"$eq": pdf_id}})
    logger.info(f"Deleted all chunks for pdf_id='{pdf_id}'.")


def get_pdf_chunk_count(pdf_id: str) -> int:
    """Count chunks stored for a given pdf_id."""
    collection = get_collection()
    result = collection.get(where={"pdf_id": {"$eq": pdf_id}})
    return len(result["ids"])
