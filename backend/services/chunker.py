"""
Hierarchical semantic chunking using LlamaIndex's SemanticSplitterNodeParser.

Strategy: two-level hierarchy with small-to-big retrieval

  Level 1 — Parent chunk (per page/section)
    The full page text with context header prepended.
    This is what the LLM sees as context — complete, coherent.

  Level 2 — Child chunks (semantic sub-units within the parent)
    Produced by SemanticSplitterNodeParser (threshold=88, tighter than before).
    These are embedded and stored in Qdrant for precise vector matching.
    Each child stores its parent's text + a shared parent_id.

Retrieval flow:
  1. Query embedding matches child chunks (small → high precision cosine score)
  2. vector_store groups children by parent_id and deduplicates
  3. LLM receives the parent text (large → complete context, fewer hallucinations)

Additional enhancements:
  • Cross-page tail overlap: last 3 sentences of page N prepended to page N+1
  • Contextual header injection: "Document: X | Section: Y | Page: Z" (Anthropic style)
  • Section heading detection via PyMuPDF font-size percentile + bold heuristic
"""
from __future__ import annotations

import re
from llama_index.core import Document
from llama_index.core.node_parser import SemanticSplitterNodeParser
import fitz  # PyMuPDF
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────

# Sentences carried forward from the bottom of the previous page to bridge
# cross-page paragraph breaks.
PAGE_OVERLAP_SENTENCES = 3

# A line is classified as a heading if its dominant font size is at or above
# this percentile of all font sizes seen on the page.
HEADING_FONT_PERCENTILE = 0.80

# Semantic split threshold.
# Lower = more splits = smaller, more precise child chunks.
# 88 is the sweet spot: tight enough for precise matching, not so tight that
# single sentences become their own chunks.
BREAKPOINT_THRESHOLD = 88   # was 95

# ── Singleton splitter ────────────────────────────────────────────────────────

_splitter: Optional[SemanticSplitterNodeParser] = None


def get_splitter() -> SemanticSplitterNodeParser:
    """Return a singleton SemanticSplitterNodeParser backed by the HF Inference API."""
    global _splitter
    if _splitter is None:
        from services.embedder import get_embed_model
        embed_model = get_embed_model()
        _splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=BREAKPOINT_THRESHOLD,
            embed_model=embed_model,
        )
        logger.info(
            f"SemanticSplitterNodeParser initialised "
            f"(threshold={BREAKPOINT_THRESHOLD}, model=BGE-small-en-v1.5)."
        )
    return _splitter


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tail_sentences(text: str, n: int) -> str:
    """Return the last `n` sentences of `text` as a single string."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    tail = sentences[-n:] if len(sentences) >= n else sentences
    return " ".join(tail)


def _extract_headings(page: fitz.Page) -> List[str]:
    """
    Return heading-like lines from a PyMuPDF page.

    A line is considered a heading if its dominant span font size is at or
    above the HEADING_FONT_PERCENTILE of all font sizes on the page.
    Bold-only short lines (≤ 80 chars) with no sentence-ending punctuation
    are also accepted as headings even if their size doesn't hit the threshold.
    """
    try:
        blocks = page.get_text("dict")["blocks"]
    except Exception:
        return []

    # Collect every font size seen on this page
    all_sizes: List[float] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                sz = span.get("size", 0)
                if sz > 0:
                    all_sizes.append(sz)

    if not all_sizes:
        return []

    sorted_sizes = sorted(all_sizes)
    threshold = sorted_sizes[int(len(sorted_sizes) * HEADING_FONT_PERCENTILE)]

    headings: List[str] = []
    for block in blocks:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            line_text = "".join(s.get("text", "") for s in spans).strip()
            if not line_text or len(line_text) < 2:
                continue

            dominant_size = max(s.get("size", 0) for s in spans)
            is_bold = any(s.get("flags", 0) & 2**4 for s in spans)  # bold flag

            size_qualifies = dominant_size >= threshold
            # Short bold line with no trailing sentence punctuation → likely a heading
            bold_short = is_bold and len(line_text) <= 80 and not line_text[-1] in ".?!"

            if (size_qualifies or bold_short) and line_text not in headings:
                headings.append(line_text)

    return headings


def _build_context_header(pdf_name: str, page_number: int, section: str) -> str:
    """
    Build the short context string prepended to every chunk before embedding.

    Example:
        Document: annual_report.pdf | Section: Financial Highlights | Page: 5
    """
    parts = [f"Document: {pdf_name}"]
    if section:
        parts.append(f"Section: {section}")
    parts.append(f"Page: {page_number}")
    return " | ".join(parts)


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_and_chunk_pdf(
    pdf_bytes: bytes,
    pdf_id: str,
    pdf_name: str,
) -> List[Dict[str, Any]]:
    """
    Extract text from PDF page-by-page with tail overlap and contextual header
    injection, then apply semantic chunking within each page.

    Returns a flat list of chunk dicts:
    {
        "text": str,          # header + chunk body (used for embedding + LLM context)
        "metadata": {
            "pdf_id":      str,
            "pdf_name":    str,
            "page_number": int,   # 1-indexed — page where the new content starts
            "chunk_index": int,
            "section":     str,   # active section heading (empty string if unknown)
        }
    }
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    splitter = get_splitter()

    all_chunks: List[Dict[str, Any]] = []
    total_pages = len(doc)

    logger.info(f"Processing PDF '{pdf_name}' ({total_pages} pages)...")

    prev_page_tail: str = ""   # cross-page overlap context
    active_section: str = ""   # last detected section heading (persists across pages)

    for page_num in range(total_pages):
        page = doc[page_num]
        page_number_1indexed = page_num + 1

        text = page.get_text().strip()

        if not text or len(text) < 30:
            prev_page_tail = ""
            continue

        # ── 1. Update active section heading ─────────────────────────────────
        page_headings = _extract_headings(page)
        if page_headings:
            active_section = page_headings[0]

        # ── 2. Cross-page tail overlap ────────────────────────────────────────
        body_text = (prev_page_tail + "\n" + text).strip() if prev_page_tail else text
        prev_page_tail = _tail_sentences(text, PAGE_OVERLAP_SENTENCES)

        # ── 3. Build context header ───────────────────────────────────────────
        header = _build_context_header(pdf_name, page_num + 1, active_section)
        # The header is prepended so the embedding encodes document position.
        # The LLM also sees it, which helps it produce precise inline citations.
        contextual_text = f"{header}\n\n{body_text}"

        # ── 4. Semantic split into child chunks (embedded + searched) ─────────
        llama_doc = Document(
            text=contextual_text,
            metadata={
                "pdf_id": pdf_id,
                "pdf_name": pdf_name,
                "page_number": page_number_1indexed,
            },
        )

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
                        "page_number": page_number_1indexed,
                        "chunk_index": chunk_idx,
                        "section": active_section,
                    },
                }
            )

    doc.close()
    logger.info(
        f"Extracted {len(all_chunks)} child chunks from '{pdf_name}' "
        f"(threshold={BREAKPOINT_THRESHOLD}, model=BGE-small-en-v1.5)."
    )
    return all_chunks

