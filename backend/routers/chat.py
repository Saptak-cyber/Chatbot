"""
Chat endpoint with full RAG pipeline:
- Retrieves relevant chunks from ChromaDB filtered by active PDF IDs
- Maintains per-session conversation history, persisted to disk so it survives
  server restarts and stays in sync with the frontend's localStorage log
- Calls Groq Llama 3.3 70B with strict grounding system prompt
- Returns response with page citations
"""
import json
import logging
import os
from fastapi import APIRouter, HTTPException
from models.schemas import ChatRequest, ChatResponse, Citation
from services.vector_store import query_chunks
from services.llm import generate_response
from typing import Dict, List

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])

MAX_HISTORY_TURNS = 5  # Keep last 5 user/assistant exchanges (10 messages)

HISTORY_PATH = os.getenv("HISTORY_PATH", "./chat_history.json")

# ── Disk-backed conversation history ─────────────────────────────────────────
# Keyed by session_id → list of clean {"role", "content"} dicts.
# Persisted to HISTORY_PATH so the backend survives restarts without losing
# context that the frontend already has in localStorage.

_conversation_history: Dict[str, List[Dict]] = {}


def _load_history() -> None:
    global _conversation_history
    if os.path.exists(HISTORY_PATH):
        try:
            with open(HISTORY_PATH, "r") as f:
                _conversation_history = json.load(f)
            logger.info(f"Loaded conversation history for {len(_conversation_history)} session(s).")
        except Exception as e:
            logger.warning(f"Could not load chat history: {e}")
            _conversation_history = {}


def _save_history() -> None:
    try:
        with open(HISTORY_PATH, "w") as f:
            json.dump(_conversation_history, f, indent=2)
    except Exception as e:
        logger.error(f"Could not save chat history: {e}")


# Load persisted history when the module is first imported
_load_history()


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

    # Query ChromaDB for relevant chunks (min_score threshold applied inside)
    try:
        chunks = query_chunks(
            query=request.message,
            pdf_ids=request.active_pdf_ids,
            top_k=8,
            min_score=0.20,
        )
    except Exception as e:
        logger.error(f"Vector store query failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve relevant context.")

    # ── Deterministic out-of-scope refusal ────────────────────────────────────
    # If no chunk cleared the similarity threshold the query is almost certainly
    # outside the PDF's content. Refuse immediately without calling the LLM.
    if not chunks:
        response_text = (
            "I'm sorry, but this question does not appear to be covered by the "
            "uploaded PDF(s). I can only answer questions based on the content of "
            "the documents you have provided. Please ask something that is addressed "
            "within those documents."
        )
        _update_history(request.session_id, history, request.message, response_text)
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            sources_used=[],
            is_grounded=False,
            retrieval_score=None,
        )

    retrieval_score = round(chunks[0]["score"], 4)

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

    # ── Detect LLM-level refusals ─────────────────────────────────────────────
    # Even when chunks pass the threshold, the LLM may still decide the context
    # doesn't actually answer the question. Detect the standard refusal prefix.
    is_grounded = "cannot find an answer" not in response_text.lower()

    # Update history with CLEAN messages (no context injected)
    _update_history(request.session_id, history, request.message, response_text)

    # Build deduplicated, sorted citations from chunks actually used
    citations = _build_citations(chunks)

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        sources_used=citations if is_grounded else [],
        is_grounded=is_grounded,
        retrieval_score=retrieval_score,
    )


@router.delete("/chat/{session_id}", summary="Clear conversation history for a session")
async def clear_history(session_id: str):
    """Clear the conversation memory for a given session_id."""
    _conversation_history.pop(session_id, None)
    _save_history()
    return {"message": "Conversation history cleared."}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _update_history(session_id: str, current_history: List[Dict], user_msg: str, assistant_msg: str) -> None:
    """Append the new user/assistant turn to conversation history and persist to disk."""
    updated = current_history + [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    _conversation_history[session_id] = updated
    _save_history()


def _build_citations(chunks: List[Dict]) -> List[Citation]:
    """Build a deduplicated, sorted list of Citations from retrieved chunks.

    For each unique (pdf_name, page_number) pair the highest similarity score
    among all chunks on that page is recorded, along with the section heading
    that was active when those chunks were created.
    """
    best: dict = {}  # key → {"score": float, "section": str}
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["pdf_name"], meta["page_number"])
        score = chunk.get("score", 0.0)
        if key not in best or score > best[key]["score"]:
            best[key] = {
                "score": score,
                "section": meta.get("section", ""),
            }

    citations = [
        Citation(
            pdf_name=key[0],
            page_number=key[1],
            section=info["section"] or None,
            score=round(info["score"], 3),
        )
        for key, info in best.items()
    ]
    citations.sort(key=lambda c: (c.pdf_name, c.page_number))
    return citations
