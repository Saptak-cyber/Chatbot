"""
Chat endpoint with full RAG pipeline:
- Retrieves relevant chunks from Qdrant filtered by active PDF IDs
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
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langsmith import traceable
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

    After summarization, Neon stores at most one SystemMessage (the running
    summary) followed by the last MAX_HISTORY_TURNS exchanges.  The summary is
    passed as a synthetic user/assistant pair so every LLM API accepts it.
    """
    result = []
    for msg in lc_history.messages:
        if isinstance(msg, SystemMessage):
            # Inject the summary as a user/assistant exchange that all LLMs accept
            result.append({"role": "user", "content": f"[Summary of our earlier conversation]\n{msg.content}"})
            result.append({"role": "assistant", "content": "Understood. I have the full context from our earlier conversation."})
        elif isinstance(msg, HumanMessage):
            result.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            result.append({"role": "assistant", "content": msg.content})
    return result


@traceable(name="summarize_conversation", run_type="llm")
def _summarize_messages(messages: list, existing_summary: str = "") -> str:
    """Progressively summarize a list of LangChain BaseMessages using Groq.

    If there is an existing running summary it is extended rather than replaced,
    so no context is ever permanently lost.
    """
    from services.llm import get_client

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )

    prompt = (
        "Progressively summarize the lines of conversation provided, adding onto "
        "the previous summary and returning a new, concise summary that captures "
        "the key topics, facts discussed, and any important context.\n\n"
    )
    if existing_summary:
        prompt += f"Current summary:\n{existing_summary}\n\n"
    prompt += f"New lines of conversation:\n{conversation_text}\n\nNew summary:"

    client = get_client()
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=512,
    )
    return resp.choices[0].message.content.strip()


def _maybe_summarize(session_id: str) -> None:
    """After MAX_HISTORY_TURNS exchanges accumulate, fold the oldest messages
    into a running summary stored as a SystemMessage in Neon.

    Storage after summarization:
        [SystemMessage(summary)]  ← rolling summary of everything before the window
        [HumanMessage, AIMessage] × MAX_HISTORY_TURNS  ← verbatim recent window
    """
    try:
        with _db_conn() as conn:
            lc_history = PostgresChatMessageHistory(
                TABLE_NAME, session_id, sync_connection=conn
            )
            all_msgs = lc_history.messages
            if not all_msgs:
                return

            # Separate an existing summary from the conversation messages
            has_summary = isinstance(all_msgs[0], SystemMessage)
            existing_summary = all_msgs[0].content if has_summary else ""
            conv_msgs = all_msgs[1:] if has_summary else all_msgs

            # Only compress when we exceed the verbatim window
            if len(conv_msgs) <= MAX_HISTORY_TURNS * 2:
                return

            to_summarize = conv_msgs[:-(MAX_HISTORY_TURNS * 2)]
            to_keep = conv_msgs[-(MAX_HISTORY_TURNS * 2):]

            new_summary = _summarize_messages(to_summarize, existing_summary)

            # Rebuild Neon history: [summary] + recent verbatim window
            lc_history.clear()
            lc_history.add_message(SystemMessage(content=new_summary))
            for msg in to_keep:
                lc_history.add_message(msg)

            logger.info(
                f"Session {session_id}: summarized {len(to_summarize)} messages "
                f"into rolling summary ({len(to_keep)} messages kept verbatim)."
            )
    except Exception as e:
        logger.error(f"Conversation summarization failed for session {session_id}: {e}")


@traceable(name="rewrite_query", run_type="llm")
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
    2. Retrieves top-k semantically similar chunks from Qdrant (filtered by PDF IDs)
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
            top_k=10,
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
    # Check for multiple refusal patterns
    refusal_patterns = [
        "cannot find an answer",
        "does not contain",
        "not addressed in",
        "no information about",
        "not covered in",
        "outside the scope",
    ]
    is_grounded = not any(pattern in response_text.lower() for pattern in refusal_patterns)

    # Persist the new turn to Neon
    _persist_turn(request.session_id, request.message, response_text)

    citations = _build_citations(chunks)
    
    # Calculate confidence level based on retrieval scores
    confidence_level = "low"
    if retrieval_score >= 0.40:
        confidence_level = "high"
    elif retrieval_score >= 0.28:
        confidence_level = "medium"

    return ChatResponse(
        response=response_text,
        session_id=request.session_id,
        sources_used=citations if is_grounded else [],
        is_grounded=is_grounded,
        retrieval_score=retrieval_score,
        confidence_level=confidence_level if is_grounded else None,
        num_sources=len(citations) if is_grounded else 0,
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
    """Append the user/assistant turn to Neon, then compress if needed."""
    try:
        with _db_conn() as conn:
            lc_history = PostgresChatMessageHistory(
                TABLE_NAME, session_id, sync_connection=conn
            )
            lc_history.add_user_message(user_msg)
            lc_history.add_ai_message(assistant_msg)
    except Exception as e:
        logger.error(f"Failed to persist conversation turn: {e}")
        return

    # Summarize old messages once the window is exceeded (non-fatal if it fails)
    _maybe_summarize(session_id)


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
