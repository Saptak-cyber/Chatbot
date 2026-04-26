from pydantic import BaseModel
from typing import List, Optional


class PDFInfo(BaseModel):
    id: str
    name: str
    page_count: int
    chunk_count: int


class UploadResponse(BaseModel):
    id: str
    name: str
    page_count: int
    chunk_count: int
    message: str


class ChatRequest(BaseModel):
    session_id: str
    message: str
    active_pdf_ids: List[str]


class Citation(BaseModel):
    pdf_name: str
    page_number: int
    section: Optional[str] = None  # active section heading when the chunk was created
    score: Optional[float] = None  # cosine similarity of the best chunk on this page


class ChatResponse(BaseModel):
    response: str
    session_id: str
    sources_used: List[Citation]
    is_grounded: bool = True          # False when the query is out of scope / refused
    retrieval_score: Optional[float] = None  # max cosine similarity among retrieved chunks
    confidence_level: Optional[str] = None  # "high", "medium", "low" based on scores
    num_sources: Optional[int] = None  # number of unique pages cited
