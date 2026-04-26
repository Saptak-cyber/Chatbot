"""
PDF management endpoints: upload, list, and delete PDFs.
Maintains a Neon-backed registry of uploaded PDFs alongside Qdrant.
"""
import os
import uuid
import logging
from contextlib import contextmanager
from typing import List, Generator

import psycopg
from fastapi import APIRouter, UploadFile, File, HTTPException

from models.schemas import PDFInfo, UploadResponse
from services.chunker import extract_and_chunk_pdf
from services.vector_store import add_chunks, delete_pdf_chunks

logger = logging.getLogger(__name__)
router = APIRouter(tags=["PDFs"])

NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
PDF_REGISTRY_TABLE = "pdf_registry"


@contextmanager
def _db_conn() -> Generator[psycopg.Connection, None, None]:
    """Open a short-lived psycopg connection to Neon."""
    if not NEON_DATABASE_URL:
        raise ValueError("NEON_DATABASE_URL environment variable is not set")
    conn = psycopg.connect(NEON_DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


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

    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {PDF_REGISTRY_TABLE} (id, name, page_count, chunk_count)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (id) DO UPDATE
                SET name = EXCLUDED.name,
                    page_count = EXCLUDED.page_count,
                    chunk_count = EXCLUDED.chunk_count
                """,
                (pdf_id, pdf_name, page_count, chunk_count),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"PDF registry insert failed (rolling back vectors): {e}")
        try:
            delete_pdf_chunks(pdf_id)
        except Exception as rollback_err:
            logger.error(f"Vector rollback failed for pdf_id={pdf_id}: {rollback_err}")
        raise HTTPException(status_code=500, detail="Failed to persist PDF metadata.")

    return UploadResponse(
        id=pdf_id,
        name=pdf_name,
        page_count=page_count,
        chunk_count=chunk_count,
        message=f"Successfully processed '{pdf_name}' into {chunk_count} semantic chunks across {page_count} pages.",
    )


@router.get("/pdfs", response_model=List[PDFInfo], summary="List all uploaded PDFs")
async def list_pdfs():
    """Return metadata for all currently indexed PDFs from Neon."""
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT id, name, page_count, chunk_count
                FROM {PDF_REGISTRY_TABLE}
                ORDER BY created_at DESC
                """
            )
            rows = cur.fetchall()
    except Exception as e:
        logger.error(f"Failed to list PDFs from registry: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch PDFs.")

    return [
        PDFInfo(id=row[0], name=row[1], page_count=row[2], chunk_count=row[3])
        for row in rows
    ]


@router.delete("/pdfs/{pdf_id}", summary="Delete a PDF and its embeddings")
async def delete_pdf(pdf_id: str):
    """Remove a PDF and all its chunks from Qdrant and Neon registry."""
    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT name FROM {PDF_REGISTRY_TABLE} WHERE id = %s",
                (pdf_id,),
            )
            row = cur.fetchone()
    except Exception as e:
        logger.error(f"Failed to check PDF in registry: {e}")
        raise HTTPException(status_code=500, detail="Failed to access PDF registry.")

    if not row:
        raise HTTPException(status_code=404, detail="PDF not found.")

    pdf_name = row[0]

    try:
        delete_pdf_chunks(pdf_id)
    except Exception as e:
        logger.error(f"Failed to delete chunks for pdf_id={pdf_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete embeddings: {str(e)}")

    try:
        with _db_conn() as conn, conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {PDF_REGISTRY_TABLE} WHERE id = %s",
                (pdf_id,),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to delete PDF from registry: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete PDF metadata.")

    logger.info(f"Deleted PDF '{pdf_name}' (pdf_id={pdf_id}).")
    return {"message": f"'{pdf_name}' has been deleted successfully."}
