# High-Level Design — PDF Conversational Agent

## 1. Overview

A **Retrieval-Augmented Generation (RAG)** web application that lets users upload PDF documents and have grounded, citation-backed conversations with their contents. The system uses a multi-layer defence against hallucination and refuses to answer questions outside the scope of the loaded documents.

**Conversational behaviour** is handled in three ways: (1) **query classification** to skip retrieval for greetings, clarifications, and history-based follow-ups; (2) **query rewriting** so short follow-ups like “repeat again” become retrievable standalone questions; (3) **rolling conversation summaries** in-memory after every five exchanges to bound token usage. **Multi-session threads** (ChatGPT-style) are persisted in `localStorage` — no server-side session database is required.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Browser (Client)                  │
│  ConversationList │ PDFSidebar │ ChatWindow          │
└─────────────────────────┬───────────────────────────┘
                          │  HTTPS / SSE
             ┌────────────▼─────────────┐
             │     FastAPI (Render)      │
             │  PDFs Router │ Chat Router│
             └──────┬───────────┬────────┘
                    │           │
      │  Qdrant Cloud  │  │      External APIs       │
      │ (vectors+meta) │  │  Groq llama-3.1-8b-inst  │
      └────────────────┘  │  HF bge-small-en-v1.5    │
                          │  HF bge-reranker-base    │
      ┌─────────────────┐ │  LangSmith (tracing)     │
      │ rank-bm25 index │ └──────────────────────────┘
      │ (in-memory)     │
      │ pdf_registry    │
      │ .json (metadata)│
      └─────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Frontend — Next.js 14 (Vercel)

| Component | Responsibility |
|---|---|
| `app/page.tsx` | Root page; owns global state (PDF list, active PDFs, conversation threads) |
| `components/PDFSidebar` | Tabbed sidebar — Chats tab (thread list) + Docs tab (upload/select/delete) |
| `components/ConversationList` | Thread management — switch, rename, delete; auto-title on first message |
| `components/ChatWindow` | Render conversation, send messages, SSE consumer, auto-title callback |
| `components/MessageBubble` | Markdown rendering (bold/lists/tables), copy-to-clipboard, citation chips, refusal badge |
| `components/LanguageSelector` | Language picker in header; z-index stacked above messages |
| `lib/api.ts` | All `fetch`/SSE calls to backend |
| `lib/types.ts` | Shared TS interfaces (`PDFInfo`, `Message`, `Citation`, `ConversationThread`) |

**State model:**

- `conversations` — `ConversationThread[]` persisted in `localStorage` (`docmind_conversations`)
- `activeThreadId` — currently viewed thread
- `pdfs` — list of uploaded PDFs (fetched from `/api/pdfs` on load)
- `activePdfIds` — PDFs currently scoped for RAG
- `sessionId` — UUID per thread; sent with every chat request

**UX:** Thread auto-titles on first user message; live preview (last message snippet) shown under each thread name.

---

## 3.2 Backend — FastAPI (Render)

#### Routers

| Router | Endpoints | Description |
|---|---|---|
| `pdfs.py` | `POST /api/upload` | Ingest, chunk, embed, and store a PDF |
| | `GET /api/pdfs` | Return list of all registered PDFs |
| | `DELETE /api/pdfs/{id}` | Remove PDF vectors from Qdrant and registry entry |
| `chat.py` | `POST /api/chat` | Query rewrite → vector retrieve → grounded response; persist turn; optional rolling summary |
| | `DELETE /api/chat/{session_id}` | Clear conversation history for a session in Neon |

#### Services

| Service | Technology | Role |
|---|---|---|
| `chunker.py` | PyMuPDF + LlamaIndex `SemanticSplitterNodeParser` (threshold=88) | Page text extraction; semantic chunking with cross-page overlap and contextual header injection |
| `embedder.py` | HuggingFace Inference API (`BAAI/bge-small-en-v1.5`) | Dense embeddings; BGE query instruction prefix at query time |
| `bm25_store.py` | `rank-bm25` | Sparse, exact-keyword retrieval. Built lazily by scrolling Qdrant. |
| `vector_store.py` | Qdrant Cloud | Store/retrieve chunk embeddings (cosine similarity); provides `query_chunks_hybrid` |
| `reranker.py` | HuggingFace Inference API (`BAAI/bge-reranker-base`) | Cross-encoder that scores and filters the RRF-merged chunk pool |
| `llm.py` | Groq API (`llama-3.1-8b-instant`) | Streaming grounded answers via SSE; multi-language; [GROUNDED]/[REFUSED] tag protocol |

#### Conversation & memory (`chat.py`)

| Mechanism | Purpose |
|---|---|
| **Query classification** (`_is_retrieval_required`) | Detects greetings, clarifications, and history-based questions — skips vector retrieval for those |
| **Query rewriting** (Groq, LangSmith `rewrite_query`) | When history exists, rewrite follow-ups into standalone questions used **only for retrieval**; original user message still sent to LLM |
| **Rolling summarization** (in-memory) | After each turn, if there are more than five exchanges, the oldest messages are summarized into a single system message; only the last five exchanges are kept verbatim |
| **Multi-language** | `response_language` BCP-47 code injected into system prompt; LLM responds in target language; citations stay in Latin script |

---

## 4. Data Flow

### 4.1 PDF Upload & Indexing

```
User selects file
      │
      ▼
PDFSidebar → POST /api/upload (multipart)
      │
      ▼
Backend: chunker.py
  ├─ PyMuPDF page.get_text() extracts text page-by-page
  │
  ├─ PyMuPDF font-size analysis detects section headings per page
  │
  ├─ Cross-page tail overlap: last 3 sentences of page N prepended
  │   to page N+1 so cross-boundary paragraphs are captured intact
  │
  ├─ Contextual header injected before each page’s text:
  │     "Document: report.pdf | Section: 3. Methodology | Page: 7"
  │
  └─ LlamaIndex SemanticSplitter (threshold=88) creates topic-coherent chunks
      │
      ▼
Backend: embedder.py
  └─ HuggingFace Inference API (BAAI/bge-small-en-v1.5) → float[] embedding per chunk
      │
      ▼
Backend: vector_store.py
  └─ Qdrant stores (chunk_text, embedding,
       {pdf_id, pdf_name, page_number, chunk_index, section})
      │
      ▼
Backend: pdf_registry.json updated with {pdf_id, name, page_count, chunk_count}
      │
      ▼
Response: { pdf_id, name, page_count, chunk_count }
```

### 4.2 RAG Chat Query

```
User types message → POST /api/chat/stream
  { message, session_id, active_pdf_ids, response_language }
      │
      ▼
Query classification (_is_retrieval_required)
  ├─ Greeting / history-based → _generate_from_history (no retrieval)
  └─ New query → continue
      │
      ▼
Query rewriting (if history non-empty)
  └─ Groq: follow-up → standalone query for retrieval only
      │
      ▼
vector_store.query_chunks_hybrid(retrieval_query, …)
  ├─ Parallel execution:
  │   ├─ Vector: Qdrant cosine similarity (top_vector_k=20, min_score=0.20)
  │   └─ Sparse: BM25 keyword search (top_bm25_k=20)
  ├─ Merge via Reciprocal Rank Fusion (RRF, k=60)
  ├─ Rerank via BAAI/bge-reranker-base (top_n=10)
  └─ Dynamic-k cutoff removes chunks < 80% of top reranker score
      │
      ▼
Layer 1 — Hard Refusal (min_score threshold)
  ├─ If no chunks pass threshold → immediate refusal; is_grounded=False; no LLM call
  └─ Otherwise pass chunks to answer LLM
      │
      ▼
Build prompt (llm.py)
  - System: 7 grounding rules + citation format + output formatting + language instruction
  - History: rolling summary + recent verbatim turns
  - Context: retrieved chunks with [Excerpt N | pdf — Page X] headers
  - User: original message + [GROUNDED]/[REFUSED] tag instruction
      │
      ▼
Groq LLaMA 3.1 8B Instant → streamed tokens via SSE
      │
      ▼
Layer 2 — Soft Refusal
  └─ [REFUSED] tag or refusal keyword → is_grounded=False
      │
      ▼
Persist turn (in-memory session history)
  └─ If > 5 exchanges: rolling summary of oldest messages
      │
      ▼
Frontend: MessageBubble — Markdown rendering, citations, copy-to-clipboard
```

---

## 5. Anti-Hallucination Architecture

| Layer | Where | Mechanism |
|---|---|---|
| **Layer 0** | Chunking (index time) | Contextual header injection encodes document position into embeddings; cross-page overlap reduces mid-paragraph splits |
| **Layer 1** | `vector_store.py` (retrieval) | Cosine `min_score=0.20` + Reranker Dynamic-k cutoff (80% relative threshold). No qualifying chunks → deterministic refusal |
| **Layer 2** | `llm.py` (generation) | Strict system prompt; LLM must refuse when context does not answer the question; refusal phrase sets `is_grounded=False` |

Query rewriting only affects **retrieval**; answers must still be grounded in retrieved excerpts.

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PDF text extraction | PyMuPDF `page.get_text()` | Simple, reliable; no heavy PDF stack in production |
| Chunking strategy | Page-scoped semantic (threshold=88) + cross-page overlap | More granular chunks; exact page metadata for citations |
| Contextual headers | Injected before embedding | Encodes document position in vectors; helps section-style queries |
| Query classification | `_is_retrieval_required` checks intent before retrieval | Skips Qdrant for greetings/clarifications; saves latency and tokens |
| Follow-up retrieval | LLM query rewrite + single `query_chunks` call | Fixes “repeat again” / anaphoric queries without multi-query expansion |
| Similarity threshold | `min_score=0.20` cosine | Hard filter before generation; reduces noise and off-topic answers |
| Long conversations | Rolling summary + last 5 exchanges verbatim | Bounded tokens; no external DB required |
| Embedding model | `BAAI/bge-small-en-v1.5` via HF Inference API | Better MTEB retrieval benchmarks than MiniLM; same 384-dim; no schema change |
| Query embedding | BGE instruction prefix | Aligns query vector with passage embeddings per model design |
| LLM | Groq `llama-3.1-8b-instant` | Fast streaming; 6K TPM free tier sufficient for evaluation |
| Vector store | Qdrant Cloud | Managed, cosine similarity |
| PDF metadata | `pdf_registry.json` | Lightweight; no second DB |
| Session threads | `localStorage` (`docmind_conversations`) | Zero backend state; survives reload; no server-side DB |
| Language support | `response_language` BCP-47 + dynamic system prompt | 8 languages; citations always in Latin script |
| Observability | LangSmith `@traceable` | Debug RAG pipeline per query |

---

## 7. Deployment Architecture

```
                ┌───────────────┐        ┌──────────────────┐
  User browser  │    Vercel     │        │    Render.com    │
  ──────────►   │  Next.js 14   │──REST──►  FastAPI/uvicorn  │
                │  (static SSR) │        │  port $PORT       │
                └───────────────┘        └────────┬─────────┘
                                                  │
                     ┌────────────────────────────┼────────────────────────┐
                     │                            │                        │
              ┌──────▼──────┐              ┌───────▼────────┐     ┌─────────▼─────────┐
              │ Qdrant Cloud│              │ pdf_registry   │     │ Neon PostgreSQL   │
              │ (managed)   │              │ .json          │     │ (chat_messages)   │
              │ QDRANT_URL  │              └────────────────┘     │ NEON_DATABASE_URL │
              │ QDRANT_KEY  │                                      └───────────────────┘
              └─────────────┘
```

- **Frontend** — Vercel; `NEXT_PUBLIC_API_URL` points at the Render backend.
- **Backend** — Render; secrets: `GROQ_API_KEY`, `HF_TOKEN`, `QDRANT_URL`, `QDRANT_API_KEY`, optional `LANGCHAIN_API_KEY` / LangSmith vars, `ALLOWED_ORIGINS`.
- **Build** — `backend/build.sh` installs `certifi` and `requirements.txt`.
- **CORS** — `ALLOWED_ORIGINS` should list the Vercel app origin in production.

---

## 8. External Dependencies

| Service | Purpose | Env Var(s) |
|---|---|---|
| Groq API | Answer generation, query rewriting, conversation summarization | `GROQ_API_KEY` |
| HuggingFace Inference API | Embeddings (`BAAI/bge-small-en-v1.5`) | `HF_TOKEN` |
| Qdrant Cloud | Vector database for embeddings | `QDRANT_URL`, `QDRANT_API_KEY` |
| LangSmith | Traces (`rewrite_query`, `generate_response_stream`, `query_chunks`) | `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` (optional) |

---

## 9. Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, lucide-react, uuid |
| Backend | FastAPI, Python, uvicorn |
| PDF & chunking | PyMuPDF, LlamaIndex `SemanticSplitterNodeParser` (threshold=88) |
| Embeddings | HuggingFace Inference API (`BAAI/bge-small-en-v1.5`) |
| Vector store | Qdrant Cloud |
| Session storage | `localStorage` — `docmind_conversations` (client-side) |
| LLM | Groq `llama-3.1-8b-instant` |
| Observability | LangSmith + `@traceable` |
| Frontend hosting | Vercel |
| Backend hosting | Render |
