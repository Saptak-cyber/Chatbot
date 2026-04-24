"""
Chat endpoint with full RAG pipeline:
- Retrieves relevant chunks from ChromaDB filtered by active PDF IDs
- Maintains per-session conversation history
- Calls Groq Llama 3.3 70B with strict grounding system prompt
- Returns response with page citations
"""
import logging
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, Citation
from services.vector_store import query_chunks
from services.llm import generate_response
from typing import Dict, List

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])

# In-memory conversation history keyed by session_id
# Each value is a list of clean {"role": str, "content": str} dicts
# (without the injected context — that is re-added per-turn)
_conversation_history: Dict[str, List[Dict]] = {}

MAX_HISTORY_TURNS = 5  # Keep last 5 user/assistant exchanges (10 messages)


@router.post("/chat", response_model=ChatResponse, summary="Send a message and get a grounded response")
async def chat(request: ChatRequest):
    """
    RAG chat endpoint:
    1. Takes a user message + list of active PDF IDs
    2. Retrieves top-k semantically similar chunks from ChromaDB (filtered by PDF IDs)
    3. Constructs a grounded prompt with conversation history
    4. Calls Groq Llama 3.3 70B
    5. Returns the response with page-level citations
    """
    if not request.active_pdf_ids:
        raise HTTPException(
            status_code=400,
            detail="Please select at least one PDF before sending a message.",
        )

    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    if len(request.message) > 2000:
        raise HTTPException(status_code=400, detail="Message is too long (max 2000 characters).")

    # Retrieve conversation history for this session
    history = _conversation_history.get(request.session_id, [])
    # Limit to last MAX_HISTORY_TURNS exchanges
    recent_history = history[-(MAX_HISTORY_TURNS * 2):]

    # Query ChromaDB for relevant chunks
    try:
        chunks = query_chunks(
            query=request.message,
            pdf_ids=request.active_pdf_ids,
            top_k=6,
        )
    except Exception as e:
        logger.error(f"Vector store query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant context.")

    if not chunks:
        response_text = (
            "I cannot find relevant information in the selected PDF(s). "
            "Please ensure you have selected the correct PDFs and that they have been loaded."
        )
        _update_history(request.session_id, history, request.message, response_text)
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            sources_used=[],
        )

    # Generate grounded response via Groq
    try:
        response_text = generate_response(
            query=request.message,
            context_chunks=chunks,
            history=recent_history,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

    # Update history with CLEAN messages (no context injected)
    _update_history(request.session_id, history, request.message, response_text)

    # Build deduplicated, sorted citations from chunks actually used
    citations = _build_citations(chunks)

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        sources_used=citations,
    )


@router.delete("/chat/{session_id}", summary="Clear conversation history for a session")
async def clear_history(session_id: str):
    """Clear the conversation memory for a given session_id."""
    _conversation_history.pop(session_id, None)
    return {"message": "Conversation history cleared."}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _update_history(session_id: str, current_history: List[Dict], user_msg: str, assistant_msg: str) -> None:
    """Append the new user/assistant turn to conversation history."""
    updated = current_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    _conversation_history[session_id] = updated


def _build_citations(chunks: List[Dict]) -> List[Citation]:
    """Build a deduplicated, sorted list of Citations from retrieved chunks."""
    seen = set()
    citations = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["pdf_name"], meta["page_number"])
        if key not in seen:
            seen.add(key)
            citations.append(
                Citation(
                    pdf_name=meta["pdf_name"],
                    page_number=meta["page_number"],
                )
            )
    citations.sort(key=lambda c: (c.pdf_name, c.page_number))
    return citations
