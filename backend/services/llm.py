"""
Groq LLM client for Llama 3.3 70B inference.
Includes a strict system prompt to ensure PDF-only grounding.
"""
from groq import Groq
from typing import List, Dict
import os
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

_client: Groq | None = None

SYSTEM_PROMPT = """You are a strict PDF Document Assistant. Your ONLY job is to answer questions using the provided PDF context excerpts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GROUND EVERY CLAIM in the provided context. Never use prior knowledge, training data, or general world facts.

2. REFUSE clearly when the context does not contain the answer.
   Use this EXACT wording:
   "I cannot find an answer to this question in the provided PDF(s). The documents do not contain information about this topic. Please ask something covered in the uploaded documents."
   Do NOT attempt a partial or speculative answer.

3. ALWAYS include inline citations immediately after the relevant statement, using this format:
   [Page X — filename]
   List all cited pages at the end too: [Sources: Page X, Y — filename]

4. VERIFY relevance before answering. Even if context excerpts were retrieved, check that they actually address the question. If the excerpts are only tangentially related and do not directly answer the question, refuse using Rule 2.

5. Do NOT infer, extrapolate, assume, or fill gaps. If the document states a fact partially, report only what is explicitly stated.

6. PARTIAL ANSWERS: If only part of a multi-part question is covered, answer the covered parts with citations, then explicitly state which parts are not addressed in the PDF.

7. NEVER fabricate page numbers, section names, quotes, statistics, or any details not explicitly present in the context.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CITATION FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Inline (after each claim): [Page 4 — report.pdf]
End-of-response summary:   [Sources: Pages 4, 7 — report.pdf]"""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key)
        logger.info("Groq client initialized.")
    return _client


@traceable(name="generate_response", run_type="llm")
def generate_response(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict],
) -> str:
    """
    Generate a strictly-grounded response using Groq Llama 3.3 70B.
    History contains clean user/assistant pairs (no context embedded).
    Context is injected only for the current user turn.
    """
    client = get_client()

    # Build rich context string with page and source metadata
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        meta = chunk["metadata"]
        context_parts.append(
            f"[Excerpt {i} | {meta['pdf_name']} — Page {meta['page_number']}]\n{chunk['text']}"
        )
    context_str = "\n\n---\n\n".join(context_parts)

    # Construct message list
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add conversation history (clean, no injected context)
    messages.extend(history)

    # Current turn: user query with injected context
    user_content = f"""CONTEXT FROM UPLOADED PDF(S):
══════════════════════════════════════════
{context_str}
══════════════════════════════════════════

USER QUESTION: {query}

Remember: Answer ONLY from the context above. Cite page numbers and document names. If the context doesn't fully answer the question, explicitly state what is missing."""

    messages.append({"role": "user", "content": user_content})

    logger.info(f"Calling Groq with {len(context_chunks)} context chunks, {len(history)} history messages.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.0,  # Changed from 0.1 to 0.0 for maximum consistency
        max_tokens=1536,
    )

    return response.choices[0].message.content
