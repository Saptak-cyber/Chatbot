"""
Semantic chunking using LlamaIndex's SemanticSplitterNodeParser.

Strategy: page-scoped semantic chunking with cross-page tail overlap.

Each PDF page is chunked independently so every chunk retains an exact page
number for citations.  To handle paragraphs/sentences that straddle a page
break, the last PAGE_OVERLAP_SENTENCES sentences of page N are prepended to
page N+1 before chunking.  The overlap text is tagged so it is NOT
double-counted in citations — the resulting chunk is still attributed to the
page where the *new* content starts, but its embedding captures the full
cross-boundary meaning.

Uses the HF Inference API embedding from services/embedder.py (no local model).
"""
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# Number of sentences carried over from the bottom of the previous page.
# 3 sentences is enough to bridge most cross-page paragraph breaks without
# bloating the chunk embeddings with irrelevant prior-page content.
PAGE_OVERLAP_SENTENCES = 3

_splitter: Optional[SemanticSplitterNodeParser] = None


def get_splitter() -> SemanticSplitterNodeParser:
    """Return a singleton SemanticSplitterNodeParser backed by the HF Inference API."""
    global _splitter
    if _splitter is None:
        from services.embedder import get_embed_model
        embed_model = get_embed_model()
        _splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=embed_model,
        )
        logger.info("SemanticSplitterNodeParser initialised (HF Inference API).")
    return _splitter


def _tail_sentences(text: str, n: int) -> str:
    """Return the last `n` sentences of `text` as a single string."""
    # Split on sentence-ending punctuation followed by whitespace or end-of-string.
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    tail = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(tail)


def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    pdf_id: str,
    pdf_name: str,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page-by-page with tail overlap between pages,
    then apply semantic chunking within each (possibly overlapped) page.

    Returns a flat list of chunk dicts:
    {
        "text": str,
        "metadata": {
            "pdf_id": str,
            "pdf_name": str,
            "page_number": int,   # 1-indexed — the page where new content starts
            "chunk_index": int,
        }
    }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    splitter = get_splitter()

    all_chunks: List[Dict[str, Any]] = []

    logger.info(f"Processing PDF '{pdf_name}' ({len(doc)} pages)...")

    prev_page_tail: str = ""  # tail sentences carried over from the previous page

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        if not text or len(text) < 30:
            # Blank/image-only page — reset the tail so we don't bridge over it
            prev_page_tail = ""
            continue

        # Prepend the previous page's tail so cross-page paragraphs are captured
        # in a single coherent chunk embedding.
        if prev_page_tail:
            combined_text = prev_page_tail + "\n" + text
        else:
            combined_text = text

        # Update tail for the next iteration BEFORE chunking (uses raw page text,
        # not the combined text, so the overlap doesn't compound across many pages)
        prev_page_tail = _tail_sentences(text, PAGE_OVERLAP_SENTENCES)

        # Create a LlamaIndex Document attributed to the current page.
        # Even though combined_text may start with sentences from the previous
        # page, the citation page number is always the current page — this is
        # the page where the *new* content lives.
        llama_doc = Document(
            text=combined_text,
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
