"""
Groq LLM client for Llama 3.1 8B inference.
Includes a strict system prompt to ensure PDF-only grounding.
Supports 8 languages natively supported by Llama 3.1 8B.
"""
from groq import Groq
from typing import List, Dict
import os
import logging

from langsmith import traceable

logger = logging.getLogger(__name__)

_client: Groq | None = None

# ── Language support ─────────────────────────────────────────────────────────
# Only languages officially supported by Llama 3.1 8B Instant (Meta's list).
SUPPORTED_LANGUAGES: dict[str, str] = {
    "auto": "auto",
    "en": "English",
    "de": "German (Deutsch)",
    "fr": "French (Français)",
    "it": "Italian (Italiano)",
    "pt": "Portuguese (Português)",
    "hi": "Hindi (हिंदी)",
    "es": "Spanish (Español)",
    "th": "Thai (ภาษาไทย)",
}

# Pre-translated hard-refusal messages.
# Used when 0 chunks pass the similarity threshold — the LLM is never called in
# that path, so out-of-scope refusals must carry their own static translations.
_HARD_REFUSAL_MESSAGES: dict[str, str] = {
    "en": (
        "I'm sorry, but this question does not appear to be covered by the "
        "uploaded PDF(s). I can only answer questions based on the content of "
        "the documents you have provided. Please ask something that is addressed "
        "within those documents."
    ),
    "de": (
        "Es tut mir leid, aber diese Frage wird in den hochgeladenen PDF(s) nicht "
        "behandelt. Ich kann nur Fragen auf der Grundlage des Inhalts der von Ihnen "
        "bereitgestellten Dokumente beantworten. Bitte stellen Sie eine Frage, die "
        "in diesen Dokumenten behandelt wird."
    ),
    "fr": (
        "Je suis désolé, mais cette question ne semble pas être couverte par le(s) "
        "PDF téléchargé(s). Je ne peux répondre qu'aux questions basées sur le "
        "contenu des documents que vous avez fournis. Veuillez poser une question "
        "qui est traitée dans ces documents."
    ),
    "it": (
        "Mi dispiace, ma questa domanda non sembra essere trattata nei PDF caricati. "
        "Posso rispondere solo a domande basate sul contenuto dei documenti che hai "
        "fornito. Chiedi qualcosa che sia trattato in quei documenti."
    ),
    "pt": (
        "Lamento, mas esta questão não parece estar abrangida pelos PDF(s) "
        "carregados. Só posso responder a perguntas com base no conteúdo dos "
        "documentos fornecidos. Por favor, faça uma pergunta que seja abordada "
        "nesses documentos."
    ),
    "hi": (
        "मुझे खेद है, लेकिन यह प्रश्न अपलोड किए गए PDF में शामिल नहीं लगता। "
        "मैं केवल आपके द्वारा प्रदान किए गए दस्तावेज़ों की सामग्री के आधार पर "
        "प्रश्नों का उत्तर दे सकता हूँ। कृपया कुछ ऐसा पूछें जो इन दस्तावेज़ों में शामिल हो।"
    ),
    "es": (
        "Lo siento, pero esta pregunta no parece estar cubierta en el/los PDF "
        "cargados. Solo puedo responder preguntas basadas en el contenido de los "
        "documentos que ha proporcionado. Por favor, haga una pregunta que esté "
        "tratada en esos documentos."
    ),
    "th": (
        "ขออภัย แต่คำถามนี้ดูเหมือนจะไม่ครอบคลุมอยู่ใน PDF ที่อัปโหลด "
        "ฉันสามารถตอบคำถามได้เฉพาะจากเนื้อหาของเอกสารที่คุณให้มาเท่านั้น "
        "กรุณาถามเกี่ยวกับเนื้อหาที่มีอยู่ในเอกสารเหล่านั้น"
    ),
}


def get_hard_refusal_text(language: str = "auto") -> str:
    """Return the hard-refusal message in the requested language.

    Falls back to English for 'auto' or any unrecognised BCP-47 code.
    """
    lang = language if language in _HARD_REFUSAL_MESSAGES else "en"
    return _HARD_REFUSAL_MESSAGES[lang]


_BASE_SYSTEM_PROMPT = """You are a strict PDF Document Assistant. Your ONLY job is to answer questions using the provided PDF context excerpts.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT RULES — NON-NEGOTIABLE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. GROUND EVERY CLAIM in the provided context. Never use prior knowledge, training data, or general world facts.

2. REFUSE clearly when the context does not contain the answer.
   Use this EXACT wording (translated into the response language):
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
End-of-response summary:   [Sources: Pages 4, 7 — report.pdf]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMATTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Always format responses with clean Markdown so they render nicely:
- Put EACH list item on its OWN line, starting with "- " immediately followed by the item text ON THAT SAME LINE
- NEVER output a line that contains only "-", "*", or "•" — that is broken Markdown
- NEVER run multiple list items together in a single paragraph
- Use **Bold** for policy names, section titles, and key terms
- Use a blank line to separate distinct sections or categories
- For multi-part answers, introduce each part with "**Category Name:**" then a newline followed by bullet points
- Prefer short, scannable sentences over dense paragraph blocks

CORRECT list format (do this):
- **Sick Leave:** 10 paid sick days per calendar year [Page 9 — doc.pdf]
- **PTO:** 15 days/year for 0-2 years of service [Page 9 — doc.pdf]

WRONG list format (never do this):
*
**Sick Leave:** 10 paid sick days
*
**PTO:** 15 days/year"""


def _build_system_prompt(language: str = "auto") -> str:
    """Build the system prompt with an appended language instruction section."""
    lang = language if language in SUPPORTED_LANGUAGES else "auto"
    if lang == "auto":
        lang_section = (
            "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "LANGUAGE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "Respond in the SAME language as the user's question. "
            "If the user writes in French respond in French, if in Spanish respond in Spanish, etc. "
            "EXCEPTION: citation markers [Page X — filename.pdf] must be kept exactly as-is — never translate them."
        )
    else:
        lang_name = SUPPORTED_LANGUAGES[lang]
        lang_section = (
            "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "LANGUAGE\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"You MUST respond ENTIRELY in {lang_name}. "
            "Write ALL explanations, answers, and summaries in this language. "
            "EXCEPTION: Citation markers [Page X — filename.pdf] must be kept exactly as-is — page numbers and filenames are never translated."
        )
    return _BASE_SYSTEM_PROMPT + lang_section


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
    language: str = "auto",
) -> tuple[str, bool]:
    """
    Generate a strictly-grounded response using Groq Llama 3.1 8B.
    History contains clean user/assistant pairs (no context embedded).
    Context is injected only for the current user turn.
    language: BCP-47 code ("auto", "en", "de", "fr", "it", "pt", "hi", "es", "th")
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
    messages = [{"role": "system", "content": _build_system_prompt(language)}]

    # Add conversation history (clean, no injected context)
    messages.extend(history)

    # Current turn: user query with injected context
    user_content = f"""CONTEXT FROM UPLOADED PDF(S):
══════════════════════════════════════════
{context_str}
══════════════════════════════════════════

USER QUESTION: {query}

IMPORTANT: Start your response with EXACTLY ONE of these tags:
- [GROUNDED] if you can answer from the context
- [REFUSED] if the context does not contain the answer

Then provide your response with citations. Remember: Answer ONLY from the context above. Cite page numbers and document names. If the context doesn't fully answer the question, explicitly state what is missing."""

    messages.append({"role": "user", "content": user_content})

    logger.info(f"Calling Groq with {len(context_chunks)} context chunks, {len(history)} history messages.")

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,  # Changed from 0.1 to 0.0 for maximum consistency
        max_tokens=1536,
    )

    full_text = (response.choices[0].message.content or "").strip()
    refusal_patterns = [
        "cannot find an answer",
        "does not contain",
        "not addressed in",
        "no information about",
        "not covered in",
        "outside the scope",
    ]

    if full_text.startswith("[GROUNDED]"):
        return full_text.replace("[GROUNDED]", "", 1).strip(), True
    if full_text.startswith("[REFUSED]"):
        return full_text.replace("[REFUSED]", "", 1).strip(), False

    # Fallback for rare non-compliant outputs.
    is_grounded = not any(pattern in full_text.lower() for pattern in refusal_patterns)
    return full_text, is_grounded


@traceable(name="generate_response_stream", run_type="llm")
def generate_response_stream(
    query: str,
    context_chunks: List[Dict],
    history: List[Dict],
    language: str = "auto",
) -> tuple[str, bool]:
    """
    Generate a strictly-grounded response using Groq Llama 3.1 8B with streaming.
    language: BCP-47 code ("auto", "en", "de", "fr", "it", "pt", "hi", "es", "th")
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

    # Construct message list (same as non-streaming)
    messages = [{"role": "system", "content": _build_system_prompt(language)}]
    messages.extend(history)

    user_content = f"""CONTEXT FROM UPLOADED PDF(S):
══════════════════════════════════════════
{context_str}
══════════════════════════════════════════

USER QUESTION: {query}

IMPORTANT: Start your response with EXACTLY ONE of these tags:
- [GROUNDED] if you can answer from the context
- [REFUSED] if the context does not contain the answer

Then provide your response with citations. Remember: Answer ONLY from the context above. Cite page numbers and document names. If the context doesn't fully answer the question, explicitly state what is missing."""

    messages.append({"role": "user", "content": user_content})

    logger.info(f"Streaming response with {len(context_chunks)} context chunks, {len(history)} history messages.")

    # Stream the response
    stream = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=0.0,
        max_tokens=1536,
        stream=True,
    )

    full_response_raw = ""
    tag_stripped = False
    is_grounded_from_tag = None
    
    for chunk in stream:
        if chunk.choices[0].delta.content:
            text = chunk.choices[0].delta.content
            full_response_raw += text
            
            # Strip [GROUNDED] or [REFUSED] tags from the beginning
            if not tag_stripped:
                raw_stripped = full_response_raw.lstrip()
                if raw_stripped.startswith("[GROUNDED]"):
                    if len(raw_stripped) >= 10:  # Tag complete
                        is_grounded_from_tag = True
                        clean_start = raw_stripped[10:].lstrip()
                        if clean_start:
                            yield clean_start
                        tag_stripped = True
                    # Don't yield yet, wait for complete tag
                    continue
                elif raw_stripped.startswith("[REFUSED]"):
                    if len(raw_stripped) >= 9:  # Tag complete
                        is_grounded_from_tag = False
                        clean_start = raw_stripped[9:].lstrip()
                        if clean_start:
                            yield clean_start
                        tag_stripped = True
                    continue
                elif len(raw_stripped) > 10:
                    # No tag found, start yielding
                    tag_stripped = True
                    yield raw_stripped
                    continue
            else:
                # Tag already stripped.
                # Only yield chunks for grounded responses; buffer refusals silently
                # so the caller can send a proper 'refusal' SSE event.
                if is_grounded_from_tag is not False:
                    yield text
    
    # Process final response (same logic as non-streaming)
    full_text = full_response_raw.strip()

    # Determine is_grounded
    if is_grounded_from_tag is not None:
        is_grounded = is_grounded_from_tag
        if is_grounded:
            final_text = full_text.replace("[GROUNDED]", "", 1).strip()
        else:
            final_text = full_text.replace("[REFUSED]", "", 1).strip()
    else:
        # Fallback for rare non-compliant outputs
        refusal_patterns = [
            "cannot find an answer",
            "does not contain",
            "not addressed in",
            "no information about",
            "not covered in",
            "outside the scope",
        ]
        is_grounded = not any(pattern in full_text.lower() for pattern in refusal_patterns)
        final_text = full_text
    
    return final_text, is_grounded
