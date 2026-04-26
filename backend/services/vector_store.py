"""
Qdrant Cloud vector store client for storing and querying PDF chunks.
Uses cosine similarity with metadata filtering by pdf_id.
"""
from qdrant_client import QdrantClient
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
VECTOR_SIZE = 384  # all-MiniLM-L6-v2 embedding dimension

_client: Optional[QdrantClient] = None


def get_client():
    """Return a singleton Qdrant client and ensure collection exists."""
    global _client
    if _client is None:
        if not QDRANT_URL or not QDRANT_API_KEY:
            raise ValueError(
                "QDRANT_URL and QDRANT_API_KEY must be set in environment variables"
            )
        
        logger.info(f"Initializing Qdrant client at '{QDRANT_URL}'...")
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        # Create collection if it doesn't exist
        collections = _client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if COLLECTION_NAME not in collection_names:
            logger.info(f"Creating collection '{COLLECTION_NAME}'...")
            _client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
            )
            
            # Create payload index for pdf_id to enable efficient filtering
            from qdrant_client.models import PayloadSchemaType
            _client.create_payload_index(
                collection_name=COLLECTION_NAME,
                field_name="pdf_id",
                field_schema=PayloadSchemaType.KEYWORD,
            )
            logger.info(f"Created payload index for 'pdf_id' field")
        
        collection_info = _client.get_collection(COLLECTION_NAME)
        logger.info(
            f"Qdrant collection '{COLLECTION_NAME}' ready. "
            f"Count: {collection_info.points_count}"
        )
    
    return _client


def add_chunks(chunks: List[Dict[str, Any]]) -> int:
    """Embed and store chunks in Qdrant. Returns number of chunks stored."""
    from services.embedder import embed_texts

    client = get_client()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Batch embed
    embeddings = embed_texts(texts)

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
                    **metadata,  # pdf_id, pdf_name, page_number, chunk_index, section
                },
            )
        )

    # Upload to Qdrant
    client.upsert(collection_name=COLLECTION_NAME, points=points)

    logger.info(f"Stored {len(chunks)} chunks in Qdrant.")
    return len(chunks)


@traceable(name="query_chunks", run_type="retriever")
def query_chunks(
    query: str,
    pdf_ids: List[str],
    top_k: int = 8,
    min_score: float = 0.20,
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

    client = get_client()
    
    # Check if collection has any points
    collection_info = client.get_collection(COLLECTION_NAME)
    if collection_info.points_count == 0:
        return []

    query_embedding = embed_query(query)

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
        # Use query_points which is the correct method in qdrant-client
        search_result = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_embedding,
            query_filter=query_filter,
            limit=top_k,
            score_threshold=min_score,
        ).points
        
        # Debug logging
        logger.info(f"Qdrant search returned {len(search_result)} results")
        if search_result:
            logger.info(f"Top result score: {search_result[0].score:.4f}")
            logger.info(f"Top result payload keys: {list(search_result[0].payload.keys())}")
    except Exception as e:
        logger.error(f"Qdrant query error: {e}")
        return []

    chunks = []
    for scored_point in search_result:
        payload = scored_point.payload.copy()  # Make a copy to avoid modifying original
        text = payload.pop("text", "")  # Extract text from payload
        
        chunks.append({
            "text": text,
            "metadata": payload,  # Remaining fields are metadata
            "score": float(scored_point.score),
        })

    # Already sorted by score descending from Qdrant
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
    client = get_client()
    
    # Delete points matching the pdf_id filter
    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[FieldCondition(key="pdf_id", match=MatchValue(value=pdf_id))]
        ),
    )
    
    logger.info(f"Deleted all chunks for pdf_id='{pdf_id}'.")


def get_pdf_chunk_count(pdf_id: str) -> int:
    """Count chunks stored for a given pdf_id."""
    client = get_client()
    
    # Use scroll to count points with the given pdf_id
    result = client.scroll(
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
        points, offset = client.scroll(
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
