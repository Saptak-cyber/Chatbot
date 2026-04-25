# High-Level Design — PDF Conversational Agent

## 1. Overview

A **Retrieval-Augmented Generation (RAG)** web application that lets users upload PDF documents and have grounded, citation-backed conversations with their contents. The system uses a multi-layer defence against hallucination and refuses to answer questions outside the scope of the loaded documents.

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
                     │  ┌─────────┐  ┌──────────┐ │
                     │  │  PDFs   │  │  Chat    │ │
                     │  │ Router  │  │ Router   │ │
                     │  └────┬────┘  └─────┬────┘ │
                     │       │             │       │
                     │  ┌────▼─────────────▼────┐  │
                     │  │       Services         │  │
                     │  │  Chunker │ Embedder    │  │
                     │  │  VectorStore │ LLM     │  │
                     │  │  MultiQuery             │  │
                     │  └────┬─────────────┬────┘  │
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
                                      │  │ Unstructured │  │
              ┌───────────────────┐   │  │ (hi_res PDF) │  │
              │  Chat History     │   │  └──────────────┘  │
              │  (JSON file)      │   └────────────────────┘
              └───────────────────┘
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
- `sessionId` — UUID persisted in `sessionStorage`; ties chat history to backend

**UX enhancement:** A global `keydown` listener auto-focuses the chat textarea when the user presses any printable key while no other input is focused, so typing immediately starts composing a message without clicking.

---

### 3.2 Backend — FastAPI (Render)

#### Routers

| Router | Endpoints | Description |
|---|---|---|
| `pdfs.py` | `POST /api/upload` | Ingest, chunk, embed, and store a PDF |
| | `GET /api/pdfs` | Return list of all registered PDFs |
| | `DELETE /api/pdfs/{id}` | Remove PDF vectors from Chroma and registry entry |
| `chat.py` | `POST /api/chat` | Multi-query RAG: expand → retrieve → filter → respond |
| | `DELETE /api/chat/{session_id}` | Clear conversation history for a session |

#### Services

| Service | Technology | Role |
|---|---|---|
| `chunker.py` | UnstructuredPDFLoader + PyMuPDF + LlamaIndex `SemanticSplitterNodeParser` | High-quality text extraction (tables, multi-column); semantic chunking per page with cross-page overlap and contextual header injection |
| `embedder.py` | HuggingFace Inference API (`all-MiniLM-L6-v2`) | Dense vector embeddings for chunks and queries |
| `vector_store.py` | ChromaDB (persistent) | Store and retrieve chunk embeddings with cosine similarity; min-score threshold filtering |
| `multi_query.py` | Groq + ChromaDB | Generate query variants via LLM; parallel ChromaDB searches; deduplicate and re-score results |
| `llm.py` | Groq API (`llama-3.3-70b-versatile`) | Generate strictly-grounded answers from retrieved context with inline citations |

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
  ├─ UnstructuredPDFLoader (hi_res) extracts text page-by-page
  │    ├─ Tables reconstructed as readable text (not garbled)
  │    └─ Multi-column layouts read in correct reading order
  │    [fallback: PyMuPDF page.get_text() if unstructured unavailable]
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
  { query, session_id, active_pdf_ids }
      │
      ▼
multi_query.py — Query Expansion
  ├─ Groq LLaMA 3.3 70B generates 3 alternative phrasings of the query
  │     e.g. "what is sga" →
  │          "What does SGA stand for?"
  │          "Explain the concept of SGA"
  │          "Define SGA as described in the document"
  │
  ├─ ChromaDB searched independently for each variant
  │    (filter: active_pdf_ids, top_k=8 per variant)
  │
  ├─ Results deduplicated by chunk text content
  │
  └─ All unique chunks re-scored against the ORIGINAL query
       (so ranking and confidence reflect user's actual intent)
      │
      ▼
Layer 1 — Hard Refusal (min_score threshold)
  ├─ If ALL chunks score < 0.20 cosine similarity → immediate refusal
  │    is_grounded=False, no LLM call made
  └─ Otherwise pass filtered chunks to LLM
      │
      ▼
Build prompt:
  - System: strict grounding rules, refusal instructions, citation format
  - History: last 5 turns loaded from disk (chat_history.json)
  - Context: retrieved chunks with [Excerpt N | pdf — Page X] headers
  - User: current query
      │
      ▼
Groq LLaMA 3.3 70B → generates grounded answer with inline citations
      │
      ▼
Layer 2 — Soft Refusal (LLM-level detection)
  ├─ If response contains "cannot find an answer" → is_grounded=False
  └─ Citations stripped from refusal responses
      │
      ▼
Response: {
  answer,
  is_grounded: bool,
  retrieval_score: float,        ← max cosine similarity of retrieved chunks
  citations: [{
    pdf_name, page_number,
    section,                     ← active heading when chunk was indexed
    score                        ← best chunk similarity on that page
  }]
}
      │
      ▼
Frontend: MessageBubble renders answer + citation chips
  ├─ Grounded: green citation bar with page, §section, High/Medium/Low label
  └─ Refused:  amber refusal bubble + "Out of scope" badge
```

---

## 5. Anti-Hallucination Architecture

The system uses a **three-layer defence** against hallucination:

| Layer | Where | Mechanism |
|---|---|---|
| **Layer 0** | Chunking (index time) | Contextual header injection encodes document position into every chunk embedding; cross-page overlap prevents mid-paragraph splits |
| **Layer 1** | `vector_store.py` (retrieval) | Cosine similarity threshold `min_score=0.20` — chunks below threshold are discarded before the LLM sees them; zero qualifying chunks → immediate deterministic refusal |
| **Layer 2** | `llm.py` (generation) | Strict system prompt with 7 explicit rules; LLM instructed to refuse when retrieved context doesn't actually answer the question; refusal phrase detected in response and flagged as `is_grounded=False` |

---

## 6. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| PDF text extraction | `UnstructuredPDFLoader` (hi_res) → PyMuPDF fallback | Correctly handles tables and multi-column layouts; fallback ensures reliability |
| Chunking strategy | Page-scoped semantic (LlamaIndex) + cross-page overlap | Preserves contextual coherence; exact page metadata for citations; overlap bridges page-break paragraphs |
| Contextual headers | Injected before embedding (Anthropic-style) | Encodes document position into vector; improves retrieval for section-reference queries |
| Retrieval strategy | Multi-query expansion (3 variants + original) | Dramatically improves recall for short/acronym/vague queries that a single embedding misses |
| Similarity threshold | `min_score=0.20` cosine | Hard filter before LLM call; prevents noise chunks from reaching generation; deterministic refusal for off-topic queries |
| Embedding model | `all-MiniLM-L6-v2` via HF Inference API | Lightweight, strong semantic quality; no local GPU needed |
| LLM | Groq `llama-3.3-70b-versatile` | Fast inference, free tier, strong instruction following |
| Vector store | ChromaDB (persistent local) | Zero-config, embedded, cosine similarity |
| PDF metadata | JSON file registry | Lightweight; avoids a full SQL DB for simple key-value mapping |
| Chat history | Disk-persisted JSON (`chat_history.json`) | Survives server restarts; stays in sync with frontend `localStorage` log |
| Session identity | UUID in `sessionStorage` | Per-tab isolation, no user auth required |
| Citation granularity | Page number + section heading + confidence score | Satisfies evaluation requirement for page/section reference; confidence score demonstrates grounding quality |

---

## 7. Deployment Architecture

```
                ┌───────────────┐        ┌──────────────────┐
  User browser  │    Vercel     │        │    Render.com    │
  ──────────►   │  Next.js 14   │──REST──►  FastAPI/uvicorn  │
                │  (static SSR) │        │  port $PORT       │
                └───────────────┘        └────────┬─────────┘
                                                  │
                                         ┌────────▼──────────┐
                                         │ Persistent Volume  │
                                         │  chroma_data/      │
                                         │  pdf_registry.json │
                                         │  chat_history.json │
                                         └────────────────────┘
```

- **Frontend** — deployed to **Vercel**; `NEXT_PUBLIC_API_URL` set to the Render backend URL.
- **Backend** — deployed to **Render** as a Python web service; `GROQ_API_KEY` and `HF_TOKEN` set as environment secrets.
- **CORS** — backend `ALLOWED_ORIGINS` env var restricts access to the Vercel domain in production.

---

## 8. External Dependencies

| Service | Purpose | Env Var |
|---|---|---|
| Groq API | LLM inference — answer generation + query variant generation | `GROQ_API_KEY` |
| HuggingFace Inference API | Sentence embeddings (`all-MiniLM-L6-v2`) | `HF_TOKEN` |

---

## 9. Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, lucide-react |
| Backend | FastAPI, Python 3.14, uvicorn |
| PDF Extraction | `unstructured[pdf]` (hi_res) via `langchain-community`, PyMuPDF fallback |
| Chunking | PyMuPDF (heading detection), LlamaIndex `SemanticSplitterNodeParser` |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Multi-Query | Native Groq + ChromaDB (no LangChain dependency) |
| Vector Store | ChromaDB (persistent) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
