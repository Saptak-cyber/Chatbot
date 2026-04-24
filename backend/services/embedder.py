"""
Unified embedding service using LlamaIndex HuggingFaceEmbedding.
Loaded once as a singleton to avoid reloading the model on every call.
This is shared by both the semantic chunker and the vector store.
"""
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from typing import List
import logging

logger = logging.getLogger(__name__)

_embed_model: HuggingFaceEmbedding | None = None


def get_embed_model() -> HuggingFaceEmbedding:
    """Return a singleton HuggingFace embedding model."""
    global _embed_model
    if _embed_model is None:
        logger.info("Loading HuggingFace embedding model (all-MiniLM-L6-v2)...")
        _embed_model = HuggingFaceEmbedding(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            embed_batch_size=32,
        )
        logger.info("Embedding model loaded successfully.")
    return _embed_model


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of texts using the singleton model (batch)."""
    model = get_embed_model()
    return model.get_text_embedding_batch(texts, show_progress=False)


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    model = get_embed_model()
    return model.get_query_embedding(query)
