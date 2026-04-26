# High-Level Design — PDF Conversational Agent

## 1. Overview

A **Retrieval-Augmented Generation (RAG)** web application that lets users upload PDF documents and have grounded, citation-backed conversations with their contents. The system uses a multi-layer defence against hallucination and refuses to answer questions outside the scope of the loaded documents.

**Conversational behaviour** is handled in three ways: (1) **query rewriting** so short follow-ups like “repeat again” become retrievable standalone questions, (2) **rolling conversation summaries** in the database after every five user/assistant exchanges to bound context while preserving long-horizon memory, and (3) **Neon PostgreSQL** as the source of truth for chat history (replacing local JSON files).

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Browser (Client)                            │
│                                                                      │
│  ┌─────────────────────┐          ┌──────────────────────────────┐  │
│  │    PDF Sidebar       │          │        Chat Window           │  │
│  │  - Upload PDF        │          │  - Auto-focus on keypress    │  │
│  │  - List / Select     │          │  - Send message              │  │
│  │  - Load / Delete     │          │  - Display AI response       │  │
│  └────────┬────────────┘          │  - Page + section citations  │  │
│           │                       │  - Confidence scores (%)      │  │
│           │                       │  - Out-of-scope refusal badge │  │
│           │                       └────────────┬─────────────────┘  │
│           └─────────────── lib/api.ts ──────────┘                   │
│                              (fetch over HTTP)                       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │ HTTPS / REST
                     ┌─────────────▼──────────────┐
                     │        FastAPI Backend       │
                     │         (Render.com)         │
                     │                             │
                     │  ┌─────────┐  ┌──────────┐   │
                     │  │  PDFs   │  │  Chat    │   │
                     │  │ Router  │  │ Router   │   │
                     │  └────┬────┘  └─────┬────┘   │
                     │       │             │       │
                     │  ┌────▼─────────────▼────┐   │
                     │  │       Services         │   │
                     │  │  Chunker │ Embedder    │   │
                     │  │  VectorStore │ LLM    │   │
                     │  └────┬─────────────┬────┘   │
                     └───────┼─────────────┼────────┘
                             │             │
              ┌──────────────▼──┐    ┌─────▼──────────────┐
              │   ChromaDB       │    │   External APIs     │
              │  (local/persist) │    │  ┌──────────────┐  │
              │  - Embeddings    │    │  │  Groq API    │  │
              │  - Chunk text    │    │  │ (LLaMA 3.3)  │  │
              │  - Metadata      │    │  └──────────────┘  │
              └──────────────────┘    │  ┌──────────────┐  │
                                      │  │ HuggingFace  │  │
              ┌───────────────────┐   │  │ Inference API│  │
              │  PDF Registry     │   │  │ (MiniLM emb) │  │
              │  (JSON file)      │   │  └──────────────┘  │
              └───────────────────┘   │  ┌──────────────┐  │
                                      │  │ LangSmith    │  │
              ┌───────────────────┐   │  │ (tracing)    │  │
              │  Neon PostgreSQL  │   │  └──────────────┘  │
              │  chat_messages    │   └────────────────────┘  │
              │  (LangChain     │                             │
              │   history)      │                             │
              └───────────────────┘                             │
```

---

## 3. Component Breakdown

### 3.1 Frontend — Next.js 14 (Vercel)

| Component | Responsibility |
|---|---|
| `app/page.tsx` | Root page; owns global state (PDF list, session ID, active PDFs) |
| `components/PDFSidebar` | Upload PDFs, view/select/delete, trigger "Load Selected" |
| `components/ChatWindow` | Render conversation, send messages, auto-focus textarea on any keypress, clear session |
| `components/MessageBubble` | Render messages with formatted text, page + section citation chips, confidence labels, refusal badge |
| `lib/api.ts` | All `fetch` calls to backend; single source of truth for API URLs |
| `lib/types.ts` | Shared TypeScript interfaces (`PDFInfo`, `Message`, `Citation`, `ChatResponse`) |

**State model:**

- `pdfs` — list of uploaded PDFs (fetched from `/api/pdfs` on load)
- `selectedPdfIds` — checkbox selection in sidebar
- `activePdfIds` — PDFs currently scoped for RAG (set on "Load Selected")
- `messages` — conversation history (persisted in `localStorage` per session)
- `sessionId` — UUID (`uuidv4`) persisted in `sessionStorage`; must be a valid UUID for backend chat history

**UX enhancement:** A global `keydown` listener auto-focuses the chat textarea when the user presses any printable key while no other input is focused, so typing immediately starts composing a message without clicking.

---

### 3.2 Backend — FastAPI (Render)

#### Routers

| Router | Endpoints | Description |
|---|---|---|
| `pdfs.py` | `POST /api/upload` | Ingest, chunk, embed, and store a PDF |
| | `GET /api/pdfs` | Return list of all registered PDFs |
| | `DELETE /api/pdfs/{id}` | Remove PDF vectors from Chroma and registry entry |
| `chat.py` | `POST /api/chat` | Query rewrite → vector retrieve → grounded response; persist turn; optional rolling summary |
| | `DELETE /api/chat/{session_id}` | Clear conversation history for a session in Neon |

#### Services

| Service | Technology | Role |
|---|---|---|
| `chunker.py` | PyMuPDF + LlamaIndex `SemanticSplitterNodeParser` | Page text extraction; semantic chunking per page with cross-page overlap and contextual header injection |
| `embedder.py` | HuggingFace Inference API (`all-MiniLM-L6-v2`) | Dense vector embeddings for chunks and queries |
| `vector_store.py` | ChromaDB (persistent) | Store and retrieve chunk embeddings with cosine similarity; `min_score` threshold filtering |
| `llm.py` | Groq API (`llama-3.3-70b-versatile`) | Generate strictly-grounded answers from retrieved context with inline citations |

#### Conversation & memory (`chat.py`)

| Mechanism | Purpose |
|---|---|
| **PostgresChatMessageHistory** (`langchain-postgres`) | Persist user/assistant turns in Neon (`chat_messages` table); `session_id` is the chat UUID |
| **Query rewriting** (Groq, LangSmith `rewrite_query`) | When history exists, rewrite follow-ups (“repeat again”, “elaborate”) into standalone questions used **only for retrieval**; the user’s original text is still sent to the answer LLM |
| **Rolling summarization** (Groq, LangSmith `summarize_conversation`) | After each turn, if there are more than five exchanges (10 messages) of *verbatim* conversation, the oldest messages are summarized into a single `SystemMessage`, merged with any prior summary, and only the last five exchanges are kept verbatim |

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
  ├─ Contextual header injected before each page's text:
  │     "Document: report.pdf | Section: 3. Methodology | Page: 7"
  │
  └─ LlamaIndex SemanticSplitter splits into topic-coherent chunks
      │
      ▼
Backend: embedder.py
  └─ HuggingFace Inference API → float[] embedding per chunk
      │
      ▼
Backend: vector_store.py
  └─ ChromaDB stores (chunk_text, embedding,
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
User types message → POST /api/chat
  { message, session_id, active_pdf_ids }
      │
      ▼
Load history from Neon (PostgresChatMessageHistory)
  ├─ Optional first SystemMessage = rolling summary of older conversation
  └─ Human/AI messages = recent verbatim window (≤ 5 exchanges after compression)
      │
      ▼
Convert to role/content dicts for the answer LLM (summary injected as synthetic user/assistant pair)
      │
      ▼
Query rewriting (if history non-empty)
  └─ Groq: follow-up → standalone query for retrieval only
      │
      ▼
vector_store.query_chunks(retrieval_query, …)
  ├─ Embed query; ChromaDB cosine similarity
  ├─ Filter: active_pdf_ids, top_k=8, min_score=0.20
  └─ Returns ranked chunks with scores
      │
      ▼
Layer 1 — Hard Refusal (min_score threshold)
  ├─ If no chunks pass threshold → immediate refusal; is_grounded=False; no answer LLM
  └─ Otherwise pass chunks to answer LLM
      │
      ▼
Build prompt (llm.py)
  - System: strict grounding rules, refusal instructions, citation format
  - History: full dict list (summary + recent turns)
  - Context: retrieved chunks with [Excerpt N | pdf — Page X] headers
  - User: original message + CONTEXT + “USER QUESTION:”
      │
      ▼
Groq LLaMA 3.3 70B → grounded answer with inline citations
      │
      ▼
Layer 2 — Soft Refusal
  ├─ If response contains "cannot find an answer" → is_grounded=False; strip citations
      │
      ▼
Persist turn to Neon (add_user_message, add_ai_message)
      │
      ▼
Rolling summarization (_maybe_summarize)
  └─ If verbatim turn count > 5: summarize oldest messages into SystemMessage; keep last 5 exchanges
      │
      ▼
Response: {
  response,
  is_grounded: bool,
  retrieval_score: float,
  sources_used: [{ pdf_name, page_number, section?, score? }]
}
      │
      ▼
Frontend: MessageBubble — citations, refusal badge, match %
```

---

## 5. Anti-Hallucination Architecture

| Layer | Where | Mechanism |
|---|---|---|
| **Layer 0** | Chunking (index time) | Contextual header injection encodes document position into embeddings; cross-page overlap reduces mid-paragraph splits |
| **Layer 1** | `vector_store.py` (retrieval) | Cosine similarity threshold `min_score=0.20`; no qualifying chunks → deterministic refusal without calling the answer LLM |
| **Layer 2** | `llm.py` (generation) | Strict system prompt; LLM must refuse when context does not answer the question; refusal phrase sets `is_grounded=False` |

Query rewriting only affects **retrieval**; answers must still be grounded in retrieved excerpts.

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PDF text extraction | PyMuPDF `page.get_text()` | Simple, reliable; no extra heavy PDF stack in production |
| Chunking strategy | Page-scoped semantic (LlamaIndex) + cross-page overlap | Coherent chunks; exact page metadata for citations |
| Contextual headers | Injected before embedding | Encodes document position in vectors; helps section-style queries |
| Follow-up retrieval | LLM query rewrite + single `query_chunks` call | Fixes “repeat again” / anaphoric queries without multi-query expansion |
| Similarity threshold | `min_score=0.20` cosine | Hard filter before generation; reduces noise and off-topic answers |
| Long conversations | Rolling summary + last 5 exchanges verbatim | Bounded tokens; LangSmith traces summarization |
| Embedding model | `all-MiniLM-L6-v2` via HF Inference API | No local GPU; good quality/size tradeoff |
| LLM | Groq `llama-3.3-70b-versatile` | Fast, strong instruction following |
| Vector store | ChromaDB (persistent local) | Embedded, cosine similarity |
| PDF metadata | `pdf_registry.json` | Lightweight metadata without a second DB for PDFs |
| Chat history | Neon PostgreSQL + `langchain-postgres` | Durable, scalable; UUID session keys |
| Observability | LangSmith (`@traceable` on rewrite, summarize, LLM, retriever) | Debug RAG and conversation pipeline in one project |
| Session identity | UUID in `sessionStorage` | Matches Neon / LangChain session id requirements |

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
              │ Persistent  │              │ pdf_registry   │     │ Neon PostgreSQL   │
              │ Volume      │              │ .json          │     │ (chat_messages)   │
              │ chroma_data/│              └────────────────┘     │ NEON_DATABASE_URL │
              └─────────────┘                                      └───────────────────┘
```

- **Frontend** — Vercel; `NEXT_PUBLIC_API_URL` points at the Render backend.
- **Backend** — Render; secrets: `GROQ_API_KEY`, `HF_TOKEN`, `NEON_DATABASE_URL`, optional `LANGCHAIN_API_KEY` / LangSmith vars, `ALLOWED_ORIGINS`.
- **Build** — `backend/build.sh` installs `certifi` and `requirements.txt` (no NLTK bootstrap required for current PDF path).
- **CORS** — `ALLOWED_ORIGINS` should list the Vercel app origin in production.

---

## 8. External Dependencies

| Service | Purpose | Env Var(s) |
|---|---|---|
| Groq API | Answer generation, query rewriting, conversation summarization | `GROQ_API_KEY` |
| HuggingFace Inference API | Embeddings (`all-MiniLM-L6-v2`) | `HF_TOKEN` |
| Neon | PostgreSQL for `chat_messages` (LangChain history) | `NEON_DATABASE_URL` |
| LangSmith | Traces (rewrite, summarize, `generate_response`, `query_chunks`) | `LANGCHAIN_API_KEY`, `LANGCHAIN_TRACING_V2`, `LANGCHAIN_PROJECT` (optional) |

---

## 9. Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, lucide-react, uuid |
| Backend | FastAPI, Python, uvicorn |
| PDF & chunking | PyMuPDF, LlamaIndex `SemanticSplitterNodeParser` |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Vector store | ChromaDB (persistent) |
| Conversation DB | Neon PostgreSQL, `langchain-postgres` (`PostgresChatMessageHistory`) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Observability | LangSmith + `@traceable` |
| Frontend hosting | Vercel |
| Backend hosting | Render |
