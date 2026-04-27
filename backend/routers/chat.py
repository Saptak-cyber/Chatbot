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
SUMMARY_INTERVAL_TURNS = 5  # Summarize every 5 exchanges: Q5, Q10, Q15, ...
REWRITE_QUERY_MAX_MESSAGES = 4  # Last 2 user/assistant turns passed into query rewrite only

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
    
    If the existing summary is very long (>800 tokens), it will be compressed first
    to prevent unbounded growth.
    """
    from services.llm import get_client

    conversation_text = "\n".join(
        f"{'User' if isinstance(m, HumanMessage) else 'Assistant'}: {m.content}"
        for m in messages
    )

    # Estimate token count (rough: 1 token ≈ 4 characters)
    existing_summary_tokens = len(existing_summary) // 4 if existing_summary else 0
    
    # If existing summary is too long, compress it first
    if existing_summary_tokens > 800:
        logger.info(f"Compressing long summary (~{existing_summary_tokens} tokens)")
        compress_prompt = (
            "The following is a conversation summary that has become too long. "
            "Create a more concise version that preserves all key facts, topics, "
            "and important context, but removes redundancy and verbose descriptions.\n\n"
            f"Long summary:\n{existing_summary}\n\n"
            "Concise summary:"
        )
        
        client = get_client()
        compress_resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": compress_prompt}],
            temperature=0.0,
            max_tokens=512,
        )
        existing_summary = compress_resp.choices[0].message.content.strip()
        logger.info(f"Compressed summary to ~{len(existing_summary) // 4} tokens")

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
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1024,  # Increased from 512 to 1024 for longer conversations
    )
    return resp.choices[0].message.content.strip()


@traceable(name="generate_from_history", run_type="llm")
def _generate_from_history(query: str, history: List[Dict], query_type: str) -> tuple[str, bool]:
    """Generate a response based only on conversation history, without retrieving new chunks.
    
    Used for greetings, confirmations, clarifications, and history-based questions.
    Uses the same context as generate_response: full history (including summary if present).
    
    Returns:
        tuple[str, bool]: (response_text, is_grounded)
    """
    from services.llm import get_client
    
    # Build conversation context - use ALL history (including summary)
    # This matches what generate_response receives
    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    
    # Different prompts based on query type
    if query_type == "greeting":
        system_prompt = (
            "You are a helpful PDF assistant. The user is sending a greeting or pleasantry. "
            "Respond warmly and professionally. Keep it brief and friendly."
        )
    elif query_type == "confirmation":
        system_prompt = (
            "You are a helpful PDF assistant. The user is asking for confirmation of your previous answer. "
            "Review the conversation history and confidently confirm your previous answer if it was based on "
            "the PDF content. Reference the page number from your previous response. "
            "Be reassuring and professional."
        )
    elif query_type == "clarification":
        system_prompt = (
            "You are a helpful PDF assistant. The user is asking you to clarify or rephrase your previous answer. "
            "Review the conversation history and explain your previous answer in a different way, using simpler "
            "language or providing additional context. Keep the same citations."
        )
    elif query_type == "history_based":
        system_prompt = (
            "You are a helpful PDF assistant. The user is asking about something already discussed in the conversation. "
            "Review the conversation history and provide the information they're asking about. "
            "If the information is in the history, provide it with the original citations. "
            "If it's not in the history, say: 'I don't see that information in our conversation. Could you ask a new question?'"
        )
    else:
        system_prompt = (
            "You are a helpful PDF assistant. Answer the user's question based on the conversation history. "
            "If the answer is in the history, provide it. If not, say you need to check the documents."
        )
    
    prompt = (
        f"{system_prompt}\n\n"
        f"Conversation history:\n{history_text}\n\n"
        f"User's current question: {query}\n\n"
        "Your response:"
    )
    
    try:
        client = get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,  # Slightly creative for natural responses
            max_tokens=300,
        )
        response_text = resp.choices[0].message.content.strip()
        
        # For greetings and confirmations, always mark as grounded (they don't need PDF grounding)
        is_grounded = query_type in {"greeting", "confirmation", "clarification"}
        
        logger.info(f"Generated response from history for query_type='{query_type}'")
        return response_text, is_grounded
        
    except Exception as e:
        logger.error(f"Failed to generate from history: {e}")
        return "I apologize, but I'm having trouble processing your request. Could you please rephrase?", False


def _maybe_summarize(session_id: str) -> None:
    """Summarize conversation in fixed batches of SUMMARY_INTERVAL_TURNS exchanges.

    This intentionally summarizes at Q5, Q10, Q15... (every 5 full turns),
    rather than summarizing on every turn once the history exceeds a window.

    Storage after summarization:
        [SystemMessage(summary)]  ← rolling summary of all completed batches
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

            # We only summarize complete user/assistant turns.
            if len(conv_msgs) % 2 != 0:
                return

            turns = len(conv_msgs) // 2

            # Run summarization only at fixed 5-turn intervals: 5, 10, 15, ...
            if turns == 0 or turns % SUMMARY_INTERVAL_TURNS != 0:
                return

            to_summarize = conv_msgs

            new_summary = _summarize_messages(to_summarize, existing_summary)

            # Rebuild Neon history with only the rolling summary.
            lc_history.clear()
            lc_history.add_message(SystemMessage(content=new_summary))

            logger.info(
                f"Session {session_id}: summarized {len(to_summarize)} messages "
                f"into rolling summary (fixed batch interval: {SUMMARY_INTERVAL_TURNS} turns)."
            )
    except Exception as e:
        logger.error(f"Conversation summarization failed for session {session_id}: {e}")


@traceable(name="is_retrieval_required", run_type="llm")
def _is_retrieval_required(query: str, history: List[Dict]) -> tuple[bool, str]:
    """Determine if retrieval is needed or if the query can be answered from conversation history.
    
    Uses the same context as generate_response: full history (including summary if present).
    
    Returns:
        tuple[bool, str]: (needs_retrieval, query_type)
        - needs_retrieval: True if chunks should be retrieved, False if history is sufficient
        - query_type: "greeting", "confirmation", "elaboration", "new_question", "history_based"
    """
    if not history:
        return True, "new_question"
    
    # Use ALL history (including summary) - same as generate_response
    context_lines = "\n".join(
        f"{m['role'].upper()}: {m['content']}" for m in history
    )
    
    prompt = (
        "You are a query classification assistant. Analyze the follow-up question and determine "
        "if it requires retrieving new information from documents, or if it can be answered using "
        "only the conversation history.\n\n"
        
        "QUERY TYPES:\n\n"
        
        "1. GREETING - Simple greetings or pleasantries\n"
        "   Examples: \"thank you\", \"thanks\", \"okay\", \"got it\", \"hi\", \"hello\"\n"
        "   Needs retrieval: NO\n\n"
        
        "2. CONFIRMATION - Asking for confirmation of previous answer\n"
        "   Examples: \"are you sure?\", \"is that correct?\", \"really?\", \"are you certain?\"\n"
        "   Needs retrieval: NO (answer is in conversation history)\n\n"
        
        "3. CLARIFICATION - Asking to clarify/rephrase previous answer\n"
        "   Examples: \"what do you mean?\", \"can you clarify?\", \"I don't understand\"\n"
        "   Needs retrieval: NO (rephrasing existing answer)\n\n"
        
        "4. ELABORATION - Asking for MORE DETAIL on the same topic OR asking for alternative approaches to the same topic\n"
        "   Examples: \"explain in detail\", \"tell me more\", \"elaborate\", \"give me more information\"\n"
        "   Examples: \"can you teach me how to...\", \"what about doing it differently\", \"any other way?\"\n"
        "   Key indicators: References same topic with pronouns (\"it\", \"that\", \"one\"), asks for alternatives/more info\n"
        "   Needs retrieval: YES (need more chunks from documents)\n\n"
        
        "5. HISTORY_BASED - Question about something already discussed\n"
        "   Examples: \"what did you say about X?\", \"remind me about Y\", \"you mentioned Z\"\n"
        "   Needs retrieval: NO (answer is in conversation history)\n\n"
        
        "6. NEW_QUESTION - New question or topic requiring document retrieval\n"
        "   Examples: Any question about information not yet discussed AND not related to previous topic\n"
        "   Key: Must be BOTH new information AND unrelated to previous conversation\n"
        "   Needs retrieval: YES\n\n"
        
        "CRITICAL RULES:\n"
        "- If the question uses pronouns (\"it\", \"that\", \"one\", \"this\") referring to the previous topic, it's ELABORATION, not NEW_QUESTION\n"
        "- If the question asks for alternatives/different approaches to the same topic, it's ELABORATION\n"
        "- Only classify as NEW_QUESTION if it's truly unrelated to the previous conversation\n\n"
        
        "INSTRUCTIONS:\n"
        "- Analyze the follow-up question in context of the conversation history\n"
        "- Determine which type it is\n"
        "- Return ONLY ONE WORD: greeting, confirmation, clarification, elaboration, history_based, or new_question\n"
        "- No explanations, no extra text\n\n"
        
        "EXAMPLES:\n"
        "History: USER: Who is the PM? ASSISTANT: Narendra Modi [Page 5]\n"
        "Query: Are you sure?\n"
        "Output: confirmation\n\n"
        
        "History: USER: What is the revenue? ASSISTANT: $5.2M [Page 7]\n"
        "Query: Thank you!\n"
        "Output: greeting\n\n"
        
        "History: USER: What is the product? ASSISTANT: CloudSync Pro [Page 3]\n"
        "Query: Explain in detail\n"
        "Output: elaboration\n\n"
        
        "History: USER: How to open a company? ASSISTANT: The document doesn't contain that info.\n"
        "Query: Can't you teach me how to make one from the internet?\n"
        "Output: elaboration\n\n"
        
        "History: USER: What is the price? ASSISTANT: $99 [Page 5]\n"
        "Query: What are the features?\n"
        "Output: new_question\n\n"
        
        "History: USER: Who is the CEO? ASSISTANT: John Smith [Page 2]\n"
        "Query: What did you say about the CEO?\n"
        "Output: history_based\n\n"
        
        "History: USER: What is the product? ASSISTANT: CloudSync Pro [Page 3]\n"
        "Query: How do I install it?\n"
        "Output: elaboration\n\n"
        
        "History: USER: What is the methodology? ASSISTANT: Three-phase approach [Page 12]\n"
        "Query: Is there another way to do it?\n"
        "Output: elaboration\n\n"
        
        f"Conversation history:\n{context_lines}\n\n"
        f"Follow-up question: {query}\n\n"
        "Classification (one word only):"
    )
    
    try:
        from services.llm import get_client
        client = get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        query_type = resp.choices[0].message.content.strip().lower()
        
        # Map query types to retrieval requirement
        no_retrieval_types = {"greeting", "confirmation", "clarification", "history_based"}
        needs_retrieval = query_type not in no_retrieval_types
        
        logger.info(f"Query classification: '{query}' → type='{query_type}', needs_retrieval={needs_retrieval}")
        return needs_retrieval, query_type
        
    except Exception as e:
        logger.warning(f"Query classification failed, defaulting to retrieval: {e}")
        return True, "new_question"


def _history_block_for_rewrite(history: List[Dict]) -> str:
    """Format up to two prior turns with explicit recency labels for the rewrite LLM.
    
    No character limit - uses full message content for better context.
    """
    def _line(m: Dict) -> str:
        return f"{m['role'].upper()}: {m['content']}"  # No character limit

    n = len(history)
    if n <= 2:
        parts = ["Most recent conversation:"]
        parts.extend(_line(m) for m in history)
        return "\n".join(parts)
    older, recent = history[:-2], history[-2:]
    parts = [
        "Older conversation (second priority — use only if the follow-up clearly refers to this turn):",
    ]
    parts.extend(_line(m) for m in older)
    parts.append(
        "Most recent conversation (highest priority — prefer this when resolving pronouns, "
        '"that", elaboration, and ambiguous references):',
    )
    parts.extend(_line(m) for m in recent)
    return "\n".join(parts)


@traceable(name="rewrite_query", run_type="llm")
def _rewrite_query(query: str, history: List[Dict], query_type: str = "new_question") -> str:
    """Rewrite a follow-up query into a standalone query using conversation history.

    ``history`` should be at most the last two user/assistant turns (see
    ``REWRITE_QUERY_MAX_MESSAGES``); callers must not pass the full session history.

    Converts ambiguous follow-ups like "repeat again", "elaborate on that", or
    "what about the second point?" into specific, self-contained queries that
    the vector store can match against PDF content. Unrelated follow-ups are
    left unchanged; when related, the most recent turn is weighted above the
    older turn for resolving references.

    If there is no history, or the LLM call fails, the original query is returned
    unchanged so retrieval always proceeds.
    
    Args:
        query: The user's query
        history: Recent conversation history (max 4 messages)
        query_type: Classification from _is_retrieval_required ("new_question", "elaboration", etc.)
    """
    if not history:
        return query
    
    # If it's a completely new question (unrelated to history), don't rewrite
    if query_type == "new_question":
        logger.info(f"Query type is 'new_question', skipping rewrite: '{query}'")
        return query

    history_window = history[-REWRITE_QUERY_MAX_MESSAGES:]
    context_lines = _history_block_for_rewrite(history_window)
    prompt = (
        "You are a query rewriting assistant. Your job is to analyze whether a follow-up question "
        "is related to the conversation history (the one or two exchanges shown below), and only "
        "then decide whether to rewrite.\n\n"
        
        "INSTRUCTIONS:\n"
        "1. First, determine if the follow-up question is RELATED to either the older or the most "
        "recent conversation shown below.\n"
        "2. A question is RELATED if it:\n"
        "   - Uses pronouns referring to previous topics (it, that, this, they, those, etc.)\n"
        "   - Asks for elaboration/clarification (\"elaborate\", \"explain more\", \"tell me more\")\n"
        "   - References previous answers implicitly (\"what about X\" when X relates to prior topic)\n"
        "   - Continues the same topic without full context\n"
        "3. A question is UNRELATED if it:\n"
        "   - Introduces a completely new topic\n"
        "   - Is already fully self-contained with all necessary context\n"
        "   - Doesn't reference anything from the conversation history\n"
        "   - Changes the subject entirely\n\n"
        
        "RULES:\n"
        "- If UNRELATED: Return the follow-up question EXACTLY as given (unchanged, character-for-character intent)\n"
        "- If RELATED: Rewrite it as a standalone question using context from history. When both the older "
        "and most recent exchanges could apply, **prioritize the most recent conversation** — tie-break "
        "pronouns, \"that\", elaboration, and vague follow-ups in favor of the last user/assistant pair. "
        "Use the older exchange only when the follow-up clearly continues that earlier topic.\n"
        "- Return ONLY the question — no explanations, no quotes, no extra text\n"
        "- The rewritten question should be natural and conversational\n"
        "- Preserve the user's intent and question style\n\n"
        
        "EXAMPLES:\n"
        "Example 1 (RELATED - uses pronoun):\n"
        "History: USER: What is the main product? ASSISTANT: The main product is CloudSync Pro.\n"
        "Follow-up: What are its features?\n"
        "Output: What are the features of CloudSync Pro?\n\n"
        
        "Example 2 (UNRELATED - new topic):\n"
        "History: USER: What is the revenue? ASSISTANT: The revenue is $5.2M.\n"
        "Follow-up: What is the company's mission statement?\n"
        "Output: What is the company's mission statement?\n\n"
        
        "Example 3 (RELATED - elaboration request):\n"
        "History: USER: What is the methodology? ASSISTANT: The methodology involves three phases.\n"
        "Follow-up: Can you elaborate on that?\n"
        "Output: Can you elaborate on the three-phase methodology?\n\n"
        
        "Example 4 (RELATED - elaboration with detail request):\n"
        "History: USER: What is the product? ASSISTANT: The product is CloudSync Pro.\n"
        "Follow-up: Explain in detail\n"
        "Output: Explain CloudSync Pro in detail\n\n"
        
        "Example 5 (RELATED - implicit reference):\n"
        "History: USER: What was the 2023 revenue? ASSISTANT: The 2023 revenue was $5.2M.\n"
        "Follow-up: What about Q4?\n"
        "Output: What was the Q4 revenue in 2023?\n\n"
        
        "Example 6 (UNRELATED - already standalone):\n"
        "History: USER: What is the price? ASSISTANT: The price is $99.\n"
        "Follow-up: How many employees does the company have?\n"
        "Output: How many employees does the company have?\n\n"
        
        "Example 7 (RELATED — prioritize most recent turn):\n"
        "Older: USER: What is CloudSync Pro? ASSISTANT: CloudSync Pro is our flagship sync product.\n"
        "Most recent: USER: What about pricing? ASSISTANT: Enterprise tier is $499 per seat annually.\n"
        "Follow-up: Any discounts for nonprofits?\n"
        "Output: Are there discounts on the Enterprise $499-per-seat annual pricing for nonprofits?\n\n"
        
        "Now analyze this conversation:\n\n"
        f"Conversation history:\n{context_lines}\n\n"
        f"Follow-up question: {query}\n\n"
        "Your output (question only):"
    )

    try:
        from services.llm import get_client
        client = get_client()
        resp = client.chat.completions.create(
            model="llama-3.1-8b-instant",
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
    Intelligent RAG chat endpoint with conditional retrieval:
    1. Classifies query type (greeting, confirmation, elaboration, new question, etc.)
    2. If retrieval not needed: generates response from conversation history only
    3. If retrieval needed: retrieves chunks, rewrites query, and generates grounded response
    4. Persists conversation and returns response with citations
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

    # ═══════════════════════════════════════════════════════════════════════════
    # STEP 1: Determine if retrieval is required
    # Pass full history (including summary) for consistent context
    # ═══════════════════════════════════════════════════════════════════════════
    needs_retrieval, query_type = _is_retrieval_required(request.message, recent_history)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BRANCH A: No retrieval needed (greetings, confirmations, history-based)
    # Pass full history (including summary) for consistent context
    # ═══════════════════════════════════════════════════════════════════════════
    if not needs_retrieval:
        logger.info(f"Skipping retrieval for query_type='{query_type}'")
        
        try:
            response_text, is_grounded = _generate_from_history(
                request.message, recent_history, query_type
            )
        except Exception as e:
            logger.error(f"Failed to generate from history: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate response.")
        
        # Persist the turn
        _persist_turn(request.session_id, request.message, response_text)
        
        return ChatResponse(
            response=response_text,
            session_id=request.session_id,
            sources_used=[],  # No sources for history-based responses
            is_grounded=is_grounded,
            retrieval_score=None,
            confidence_level=None,
            num_sources=0,
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # BRANCH B: Retrieval needed (new questions, elaborations)
    # ═══════════════════════════════════════════════════════════════════════════
    logger.info(f"Retrieval required for query_type='{query_type}'")
    
    # Rewrite query for better retrieval (especially important for elaborations)
    # Only rewrite if it's an elaboration - new questions should not be rewritten
    # Pass last 4 messages (2 turns) for rewriting - no character limit
    retrieval_query = _rewrite_query(
        request.message, 
        recent_history[-REWRITE_QUERY_MAX_MESSAGES:] if len(recent_history) > REWRITE_QUERY_MAX_MESSAGES else recent_history,
        query_type
    )

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

    # Generate grounded response via Groq using the rewritten query for consistency
    try:
        response_text, is_grounded = generate_response(
            query=retrieval_query,  # Use rewritten query instead of original
            context_chunks=chunks,
            history=recent_history,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")

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
