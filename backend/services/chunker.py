"""
Semantic chunking using LlamaIndex's SemanticSplitterNodeParser.
Chunks are created per-page so each chunk retains exact page metadata for citations.
Uses the shared embedding singleton from services/embedder.py.
"""
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
import fitz  # PyMuPDF
from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)

_splitter: SemanticSplitterNodeParser | None = None


def get_splitter() -> SemanticSplitterNodeParser:
    """Return a singleton SemanticSplitterNodeParser."""
    global _splitter
    if _splitter is None:
        from services.embedder import get_embed_model
        embed_model = get_embed_model()
        _splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        logger.info("SemanticSplitterNodeParser initialized.")
    return _splitter


def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    pdf_id: str,
    pdf_name: str,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page-by-page (preserving page metadata),
    then apply semantic chunking within each page.

    Returns a flat list of chunk dicts:
    {
        "text": str,
        "metadata": {
            "pdf_id": str,
            "pdf_name": str,
            "page_number": int,   # 1-indexed
            "chunk_index": int,
        }
    }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    splitter = get_splitter()

    all_chunks: List[Dict[str, Any]] = []

    logger.info(f"Processing PDF '{pdf_name}' ({len(doc)} pages)...")

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text or len(text) < 30:
            # Skip effectively blank pages
            continue

        # Create a LlamaIndex Document for this page
        llama_doc = Document(
            text=text,
            metadata={
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "page_number": page_num + 1,  # 1-indexed
            },
        )

        # Apply semantic chunking — splits based on embedding similarity
        nodes = splitter.get_nodes_from_documents([llama_doc])

        for chunk_idx, node in enumerate(nodes):
            chunk_text = node.text.strip()
            if not chunk_text:
                continue
            all_chunks.append(
                {
                    "text": chunk_text,
                    "metadata": {
                        "pdf_id": pdf_id,
                        "pdf_name": pdf_name,
                        "page_number": page_num + 1,
                        "chunk_index": chunk_idx,
                    },
                }
            )

    doc.close()
    logger.info(f"Extracted {len(all_chunks)} semantic chunks from '{pdf_name}'.")
    return all_chunks
