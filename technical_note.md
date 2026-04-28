# Technical Note — DocMind PDF Conversational Agent

## 1. System Overview

DocMind is a **PDF-constrained Retrieval-Augmented Generation (RAG)** agent. Users upload PDF documents; the system indexes them semantically and allows free-form conversation strictly grounded in those documents. Every claim in a response is backed by a page-level citation.

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser (Next.js)                    │
│  ConversationList │ PDFSidebar │ ChatWindow │ MessageBubble  │
└─────────────────────────┬───────────────────────────────────┘
                          │  HTTPS / SSE
┌─────────────────────────▼───────────────────────────────────┐
│                  FastAPI Backend (Python)                    │
│                                                             │
│  POST /api/pdfs/upload   →  chunker.py → embedder.py       │
│  DELETE /api/pdfs/{id}   →  vector_store.py (delete)       │
│  POST /api/chat/stream   →  chat.py (RAG pipeline)         │
└────────────┬──────────────────────────┬─────────────────────┘
             │                          │
    ┌────────▼────────┐       ┌─────────▼──────────┐
    │  Qdrant Cloud   │       │  Groq Cloud         │
    │  (vector store) │       │  llama-3.1-8b-inst  │
    └─────────────────┘       └────────────────────┘
             │
    ┌────────▼────────────────┐
    │  HuggingFace Inference  │
    │  BAAI/bge-small-en-v1.5 │
    └─────────────────────────┘
```

### Key Components

| Component | Technology | Role |
|---|---|---|
| Frontend | Next.js 14, TypeScript | UI, SSE consumer, localStorage sessions |
| Backend | FastAPI, Python 3.11 | API, RAG orchestration, SSE producer |
| Vector Store | Qdrant Cloud | Semantic chunk storage & retrieval |
| LLM | Groq · Llama 3.1 8B Instant | Response generation |
| Embeddings | HF Inference API · BGE-small-en-v1.5 | Chunk & query embeddings |
| Chunking | LlamaIndex SemanticSplitterNodeParser | Meaning-aware splitting |
| Observability | LangSmith | Full trace of every RAG call |

---

## 3. RAG Pipeline (per request)

```
User message
    │
    ▼
[1] Query classification (_is_retrieval_required)
    ├── Greeting / history-based → _generate_from_history (no retrieval)
    └── New query → continue
    │
    ▼
[2] Query rewriting (_rewrite_query)
    Expand pronouns, add context from recent history
    │
    ▼
[3] Semantic retrieval (query_chunks)
    embed_query (BGE + instruction prefix)
    → Qdrant cosine search (top_k=10, min_score=0.20)
    → Returns ranked flat chunks sorted by cosine score
    │
    ▼
[4] Hard refusal gate
    0 chunks above min_score=0.20 → refuse without calling LLM
    │
    ▼
[5] LLM generation (generate_response_stream)
    System prompt: 7 grounding rules + citation format + output formatting
    User turn: injected chunks + question + [GROUNDED]/[REFUSED] instruction
    Streamed token-by-token via Groq SSE → forwarded to browser via FastAPI SSE
    │
    ▼
[6] Persistence
    Messages saved to session history (in-memory + summarised after N turns)
```

---

## 4. Key Design Decisions

### 4.1 Flat Semantic Chunking (Threshold=88)

**Decision:** Single-level semantic chunking per page using `SemanticSplitterNodeParser`.

- **Chunks** (~150–300 tokens) — produced by `SemanticSplitterNodeParser` with `breakpoint_threshold=88`. Lower threshold than the default produces more granular, semantically dense chunks → higher cosine precision at retrieval time.
- Each chunk carries a **contextual header** injected before the page text: `"Document: file.pdf | Section: 3. Leave Policy | Page: 7"` — encoding document position directly into the embedding space.
- A **cross-page tail overlap** (last 3 sentences of page N prepended to page N+1) prevents paragraphs split across page boundaries from losing context.

**Retrieval:** `query_chunks` embeds the rewritten query with a BGE instruction prefix, runs cosine search against Qdrant, and returns the top-10 flat chunks above `min_score=0.20`.

**Rationale:** Flat chunking avoids sending full-page parent texts to the LLM, staying well within the Groq free-tier 6,000 TPM limit. Semantic boundaries (rather than fixed-size windows) preserve meaning and improve citation accuracy.

**Trade-off:** Slightly less surrounding context per chunk compared to a parent-retrieval strategy, but eliminates 413 token-limit errors on the Groq free tier.

### 4.2 Strict PDF Grounding

The system prompt contains 7 explicit prohibitions (no inference, no extrapolation, no fabricated citations). Every LLM call begins with a `[GROUNDED]`/`[REFUSED]` tag instruction so refusals are structurally enforced, not style-dependent.

A secondary **keyword refusal gate** in `query_chunks` refuses queries that return 0 chunks above the cosine threshold, before the LLM is even called — eliminating hallucination on out-of-scope queries.

### 4.3 Streaming via SSE

The backend uses FastAPI `StreamingResponse` with Server-Sent Events. The Groq stream is forwarded token-by-token to the browser. This gives immediate first-token latency perception even on long responses.

History-sourced responses (greetings, clarifications) are simulated as word-by-word streams to maintain UI consistency.

### 4.4 Embedding Model — BGE-small-en-v1.5

Chosen over `all-MiniLM-L6-v2` because:
- Same 384-dimensional output (no Qdrant schema change needed)
- Significantly better MTEB retrieval benchmark scores
- Supports an instruction prefix for queries: `"Represent this sentence for searching relevant passages: "` — applied only to query embeddings, not passage embeddings, as recommended by the model authors

### 4.5 Multi-Session Conversation Threads

Sessions are stored entirely in browser `localStorage` — no server-side session database. The FastAPI backend is stateless with respect to session identity; only conversation history is held in-memory per active request. This keeps the backend horizontally scalable.

### 4.6 Multi-Language Support

Language is a stateless parameter passed on every request. The system prompt dynamically appends a `LANGUAGE` section with explicit instructions. Supported languages are restricted to those with documented Llama 3.1 8B competence: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai.

Citation markers (`[Page N — file.pdf]`) are always preserved in their original Latin-script form regardless of output language, ensuring human-verifiable grounding across all languages.

---

## 5. Trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| HF Inference API for embeddings | No GPU required, zero local memory | Network latency per embed call; rate-limited |
| Groq `llama-3.1-8b-instant` | ~700 tokens/s, real-time streaming | 6K TPM free-tier cap; smaller model than 70B |
| Qdrant Cloud free tier | No infra management | 1 GB storage cap; cold-start latency after idle |
| Flat semantic chunking (not hierarchical) | Stays within 6K TPM; simpler retrieval | Less surrounding context per chunk than full-page parent retrieval |
| Semantic chunking (not fixed-size) | Meaning-preserving boundaries | ~2× slower indexing due to embedding calls during split |
| localStorage sessions | Zero backend state, instant | No cross-device sync; data lost on browser clear |
| Flat Qdrant collection | Simple, no join logic | Cannot do true multi-hop hierarchical retrieval |
| 8-language support | Broad accessibility | Citation format must stay Latin-script; LLM accuracy varies by language |

---

## 6. Observability

All RAG-critical functions are decorated with `@traceable` (LangSmith):

- `query_chunks` — retriever trace (inputs: query, pdf_ids; outputs: chunks + scores)
- `generate_response` — LLM trace (non-streaming)
- `generate_response_stream` — LLM trace (streaming; output recorded as indexed token array)

Each trace captures latency, token counts, and retrieval scores, enabling per-query debugging without re-running the system.

---

## 7. Deployment Targets

| Layer | Platform |
|---|---|
| Frontend | Vercel (Next.js serverless) |
| Backend | Render (Docker, free tier) |
| Vector DB | Qdrant Cloud |
| LLM | Groq Cloud API |
| Embeddings | HuggingFace Inference API |
