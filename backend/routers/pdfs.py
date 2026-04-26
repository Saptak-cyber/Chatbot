"""
PDF management endpoints: upload, list, and delete PDFs.
Maintains a JSON-backed registry of uploaded PDFs alongside Qdrant.
"""
import json
import os
import uuid
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from models.schemas import PDFInfo, UploadResponse
from services.chunker import extract_and_chunk_pdf
from services.vector_store import add_chunks, delete_pdf_chunks
from typing import List

logger = logging.getLogger(__name__)
router = APIRouter(tags=["PDFs"])

PDF_REGISTRY_PATH = os.getenv("PDF_REGISTRY_PATH", "./pdf_registry.json")

# In-memory PDF registry: pdf_id -> {name, page_count, chunk_count}
_pdf_registry: dict = {}


def _load_registry() -> None:
    global _pdf_registry
    if os.path.exists(PDF_REGISTRY_PATH):
        try:
            with open(PDF_REGISTRY_PATH, "r") as f:
                _pdf_registry = json.load(f)
            logger.info(f"Loaded {len(_pdf_registry)} PDFs from registry.")
        except Exception as e:
            logger.warning(f"Could not load PDF registry: {e}")
            _pdf_registry = {}


def _save_registry() -> None:
    try:
        with open(PDF_REGISTRY_PATH, "w") as f:
            json.dump(_pdf_registry, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save PDF registry: {e}")


# Load registry on module import
_load_registry()


@router.post("/upload", response_model=UploadResponse, summary="Upload and index a PDF")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF, semantically chunk it, embed it, and store in Qdrant."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files (.pdf) are accepted.")

    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    pdf_id = str(uuid.uuid4())
    pdf_name = file.filename

    logger.info(f"Processing upload: '{pdf_name}' (pdf_id={pdf_id})")

    try:
        chunks = extract_and_chunk_pdf(pdf_bytes, pdf_id, pdf_name)
    except Exception as e:
        logger.error(f"Chunking failed: {e}")
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {str(e)}")

    if not chunks:
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from this PDF. It may be a scanned image without OCR.",
        )

    try:
        chunk_count = add_chunks(chunks)
    except Exception as e:
        logger.error(f"Vector store insert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to store embeddings: {str(e)}")

    page_count = max(c["metadata"]["page_number"] for c in chunks)

    _pdf_registry[pdf_id] = {
        "name": pdf_name,
        "page_count": page_count,
        "chunk_count": chunk_count,
    }
    _save_registry()

    return UploadResponse(
        id=pdf_id,
        name=pdf_name,
        page_count=page_count,
        chunk_count=chunk_count,
        message=f"Successfully processed '{pdf_name}' into {chunk_count} semantic chunks across {page_count} pages.",
    )


@router.get("/pdfs", response_model=List[PDFInfo], summary="List all uploaded PDFs")
async def list_pdfs():
    """Return metadata for all currently indexed PDFs."""
    return [
        PDFInfo(id=pdf_id, **info)
        for pdf_id, info in _pdf_registry.items()
    ]


@router.delete("/pdfs/{pdf_id}", summary="Delete a PDF and its embeddings")
async def delete_pdf(pdf_id: str):
    """Remove a PDF and all its chunks from Qdrant and the registry."""
    if pdf_id not in _pdf_registry:
        raise HTTPException(status_code=404, detail="PDF not found.")

    pdf_name = _pdf_registry[pdf_id]["name"]

    try:
        delete_pdf_chunks(pdf_id)
    except Exception as e:
        logger.error(f"Failed to delete chunks for pdf_id={pdf_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")

    del _pdf_registry[pdf_id]
    _save_registry()

    logger.info(f"Deleted PDF '{pdf_name}' (pdf_id={pdf_id}).")
    return {"message": f"'{pdf_name}' has been deleted successfully."}
