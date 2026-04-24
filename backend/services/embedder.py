"""
Embedding service using the HuggingFace Inference API (remote inference).
No model is loaded locally — all embedding computation happens on HF servers.

Provides:
  - embed_texts(texts)  → List[List[float]]   for ChromaDB inserts
  - embed_query(query)  → List[float]          for ChromaDB queries
  - get_embed_model()   → LlamaIndex-compatible BaseEmbedding
                          used by SemanticSplitterNodeParser
"""
from __future__ import annotations

import os
import logging
import numpy as np
from typing import Any, List, Optional

from huggingface_hub import InferenceClient
from llama_index.core.embeddings import BaseEmbedding
from pydantic import Field

logger = logging.getLogger(__name__)

HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ─── HF Inference API client (singleton) ─────────────────────────────────────

_hf_client: Optional[InferenceClient] = None


def _get_client() -> InferenceClient:
    global _hf_client
    if _hf_client is None:
        token = os.getenv("HF_TOKEN")
        if not token:
            logger.warning("HF_TOKEN not set — embedding calls may be rate-limited.")
        _hf_client = InferenceClient(token=token)
        logger.info("HuggingFace InferenceClient initialised.")
    return _hf_client


def _call_inference_api(texts: List[str]) -> List[List[float]]:
    """Call HF Inference API and return embeddings as a list of float lists."""
    client = _get_client()
    result = client.feature_extraction(texts, model=HF_MODEL)
    # result is a numpy ndarray of shape (n_texts, embedding_dim)
    if isinstance(result, np.ndarray):
        return result.tolist()
    # Some versions return a nested list already
    return list(result)


# ─── Public helpers used by vector_store.py ──────────────────────────────────

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of documents via HF Inference API."""
    logger.debug(f"Embedding {len(texts)} texts via HF Inference API.")
    return _call_inference_api(texts)


def embed_query(query: str) -> List[float]:
    """Embed a single query string via HF Inference API."""
    return _call_inference_api([query])[0]


# ─── LlamaIndex-compatible adapter (used by SemanticSplitterNodeParser) ──────

class HFInferenceAPIEmbedding(BaseEmbedding):
    """
    Thin LlamaIndex BaseEmbedding wrapper around HuggingFace Inference API.
    Passes 'model_name' to BaseEmbedding so LlamaIndex internals are happy.
    """

    hf_token: Optional[str] = Field(default=None, exclude=True)

    def __init__(self, **kwargs: Any) -> None:
        # Ensure the LlamaIndex model_name field is set
        kwargs.setdefault("model_name", HF_MODEL)
        super().__init__(**kwargs)

    def _get_text_embedding(self, text: str) -> List[float]:
        return _call_inference_api([text])[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return _call_inference_api([query])[0]

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._get_query_embedding(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._get_text_embedding(text)

    def get_text_embedding_batch(
        self, texts: List[str], show_progress: bool = False, **kwargs: Any
    ) -> List[List[float]]:
        return _call_inference_api(texts)


# ─── Singleton embed_model for the chunker ───────────────────────────────────

_embed_model: Optional[HFInferenceAPIEmbedding] = None


def get_embed_model() -> HFInferenceAPIEmbedding:
    """Return a singleton HFInferenceAPIEmbedding instance."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HFInferenceAPIEmbedding()
        logger.info("HFInferenceAPIEmbedding (LlamaIndex adapter) initialised.")
    return _embed_model
