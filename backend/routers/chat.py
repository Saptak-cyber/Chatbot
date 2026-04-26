"""
Chat endpoint with full RAG pipeline:
- Retrieves relevant chunks from ChromaDB filtered by active PDF IDs
- Maintains per-session conversation history in Neon PostgreSQL via
  LangChain's PostgresChatMessageHistory (langchain-postgres)
- Calls Groq Llama 3.3 70B with strict grounding system prompt
- Returns response with page citations
"""
import logging
import os
import psycopg
from contextlib import contextmanager
from fastapi import APIRouter, HTTPException
from langchain_postgres import PostgresChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from models.schemas import ChatRequest, ChatResponse, Citation
from services.vector_store import query_chunks
from services.llm import generate_response
from typing import Dict, Generator, List

logger = logging.getLogger(__name__)
router = APIRouter(tags=["Chat"])

MAX_HISTORY_TURNS = 5  # Keep last 5 user/assistant exchanges (10 messages)

NEON_DATABASE_URL = os.getenv("NEON_DATABASE_URL")
TABLE_NAME = "chat_messages"


# ── DB connection helper ──────────────────────────────────────────────────────

@contextmanager
def _db_conn() -> Generator[psycopg.Connection, None, None]:
    """Open a short-lived psycopg connection to Neon, close on exit.

    Neon's pooler endpoint handles the actual connection pooling at the
    infrastructure level, so one connection per request is fine.
    """
    if not NEON_DATABASE_URL:
        raise ValueError("NEON_DATABASE_URL environment variable is not set")
    conn = psycopg.connect(NEON_DATABASE_URL)
    try:
        yield conn
    finally:
        conn.close()


# ── LangChain history helpers ─────────────────────────────────────────────────

def _history_to_dicts(lc_history: PostgresChatMessageHistory) -> List[Dict]:
    """Convert stored LangChain BaseMessages to plain role/content dicts.

    Only the most recent MAX_HISTORY_TURNS exchanges are returned so the
    Groq context window stays manageable.
    """
    result = []
    for msg in lc_history.messages[-(MAX_HISTORY_TURNS * 2):]:
        if isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


def _rewrite_query(query: str, history: List[Dict]) -> str:
    """Rewrite a follow-up query into a standalone query using conversation history.

    Converts ambiguous follow-ups like "repeat again", "elaborate on that", or
    "what about the second point?" into specific, self-contained queries that
    the vector store can match against PDF content.

    If there is no history, or the LLM call fails, the original query is returned
    unchanged so retrieval always proceeds.
    """
    if not history:
        return query

    # Use the last 2 turns (4 messages) as context — enough without bloating the prompt
    context_lines = "\n".join(
        f"{m['role'].upper()}: {m['content'][:400]}" for m in history[-4:]
    )
    prompt = (
        "Given the conversation history below and a follow-up question, rewrite the "
        "follow-up as a standalone, specific question that can be understood and "
        "answered without any prior context. If the question is already self-contained "
        "and specific, return it unchanged.\n"
        "Return ONLY the rewritten question — no explanation, no quotes.\n\n"
        f"Conversation history:\n{context_lines}\n\n"
        f"Follow-up question: {query}\n\n"
        "Standalone question:"
    )

    try:
        from services.llm import get_client
        client = get_client()
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=150,
        )
        rewritten = resp.choices[0].message.content.strip()
        if rewritten and rewritten.lower() != query.lower():
            logger.info(f"Query rewritten for retrieval: '{query}' → '{rewritten}'")
        return rewritten or query
    except Exception as e:
        logger.warning(f"Query rewrite failed, using original: {e}")
        return query


# ── Chat endpoint ──────────────────────────────────────────────────────────────

@router.post("/chat", response_model=ChatResponse, summary="Send a message and get a grounded response")
async def chat(request: ChatRequest):
    """
    RAG chat endpoint:
    1. Takes a user message + list of active PDF IDs
    2. Retrieves top-k semantically similar chunks from ChromaDB (filtered by PDF IDs)
    3. Constructs a grounded prompt with the session's conversation history from Neon
    4. Calls Groq Llama 3.3 70B
    5. Persists the new turn to Neon and returns the response with page-level citations
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

    # Load session history from Neon
    recent_history: List[Dict] = []
    try:
        with _db_conn() as conn:
            lc_history = PostgresChatMessageHistory(
                TABLE_NAME, request.session_id, sync_connection=conn
            )
            recent_history = _history_to_dicts(lc_history)
    except Exception as e:
        logger.error(f"Failed to load conversation history: {e}")
        # Non-fatal: continue without history

    # Rewrite ambiguous follow-up queries into standalone queries for better retrieval.
    # The original message is still passed to the LLM so the response feels natural.
    retrieval_query = _rewrite_query(request.message, recent_history)

    try:
        chunks = query_chunks(
            query=retrieval_query,
            pdf_ids=request.active_pdf_ids,
            top_k=8,
            min_score=0.20,
        )
    except Exception as e:
        logger.error(f"Retrieval failed: {e}")
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
        _persist_turn(request.session_id, request.message, response_text)
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
    is_grounded = "cannot find an answer" not in response_text.lower()

    # Persist the new turn to Neon
    _persist_turn(request.session_id, request.message, response_text)

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
    """Clear all stored messages for a given session_id from Neon."""
    try:
        with _db_conn() as conn:
            lc_history = PostgresChatMessageHistory(
                TABLE_NAME, session_id, sync_connection=conn
            )
            lc_history.clear()
    except Exception as e:
        logger.error(f"Failed to clear history for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear conversation history.")
    return {"message": "Conversation history cleared."}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _persist_turn(session_id: str, user_msg: str, assistant_msg: str) -> None:
    """Append the user/assistant turn to Neon via LangChain history object."""
    try:
        with _db_conn() as conn:
            lc_history = PostgresChatMessageHistory(
                TABLE_NAME, session_id, sync_connection=conn
            )
            lc_history.add_user_message(user_msg)
            lc_history.add_ai_message(assistant_msg)
    except Exception as e:
        logger.error(f"Failed to persist conversation turn: {e}")


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
