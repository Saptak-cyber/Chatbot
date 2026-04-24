"""
Groq LLM client for Llama 3.3 70B inference.
Includes a strict system prompt to ensure PDF-only grounding.
"""
from groq import Groq
from typing import List, Dict
import os
import logging

logger = logging.getLogger(__name__)

_client: Groq | None = None

SYSTEM_PROMPT = """You are a strict PDF Document Assistant. Your ONLY job is to answer questions using the provided PDF context.

STRICT RULES (non-negotiable):
1. Answer EXCLUSIVELY from the provided PDF context. Do NOT use any external knowledge or make assumptions beyond what is stated.
2. If the answer is NOT present in the provided context, respond EXACTLY with:
   "I cannot find an answer to this question in the provided PDF(s). The documents do not contain information about this topic. Please ask something covered in the uploaded documents."
3. ALWAYS cite your sources at the end of each response in this format: [Source: Page X, PDF: filename] or [Sources: Page X, Y — filename]
4. Be concise, accurate, and faithful to the source material.
5. Do NOT infer, extrapolate, hallucinate, or add information not explicitly stated in the context.
6. If the question is partially answerable, answer only the parts covered by the context and state clearly what is not covered."""


def get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is not set")
        _client = Groq(api_key=api_key)
        logger.info("Groq client initialized.")
    return _client


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

Remember: Answer ONLY from the context above. Cite page numbers and document names."""

    messages.append({"role": "user", "content": user_content})

    logger.info(f"Calling Groq with {len(context_chunks)} context chunks, {len(history)} history messages.")

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    return response.choices[0].message.content
