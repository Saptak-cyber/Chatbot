# High-Level Design — PDF Conversational Agent

## 1. Overview

A **Retrieval-Augmented Generation (RAG)** web application that lets users upload PDF documents and have grounded, citation-backed conversations with their contents. The system refuses to answer questions outside the scope of the loaded documents.

---

## 2. Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────┐
│                          Browser (Client)                            │
│                                                                      │
│  ┌─────────────────────┐          ┌──────────────────────────────┐  │
│  │    PDF Sidebar       │          │        Chat Window           │  │
│  │  - Upload PDF        │          │  - Send message              │  │
│  │  - List / Select     │          │  - Display AI response       │  │
│  │  - Load / Delete     │          │  - Show page citations       │  │
│  └────────┬────────────┘          └────────────┬─────────────────┘  │
│           │                                     │                    │
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
              └───────────────────┘   └────────────────────┘
```

---

## 3. Component Breakdown

### 3.1 Frontend — Next.js 14 (Vercel)

| Component | Responsibility |
|---|---|
| `app/page.tsx` | Root page; owns global state (PDF list, session ID, active PDFs) |
| `components/PDFSidebar` | Upload PDFs, view/select/delete, trigger "Load Selected" |
| `components/ChatWindow` | Render conversation, send messages, clear session |
| `components/MessageBubble` | Render individual messages with formatted text and citation chips |
| `lib/api.ts` | All `fetch` calls to backend; single source of truth for API URLs |
| `lib/types.ts` | Shared TypeScript interfaces (`PDFInfo`, `Message`, `Citation`, `ChatResponse`) |

**State model:**
- `pdfs` — list of uploaded PDFs (fetched from `/api/pdfs` on load)
- `selectedPdfIds` — checkbox selection in sidebar
- `activePdfIds` — PDFs currently scoped for RAG (set on "Load Selected")
- `messages` — conversation history (local; mirrors backend session)
- `sessionId` — UUID persisted in `sessionStorage`; ties chat history to backend

---

### 3.2 Backend — FastAPI (Render)

#### Routers

| Router | Endpoints | Description |
|---|---|---|
| `pdfs.py` | `POST /api/upload` | Ingest, chunk, embed, and store a PDF |
| | `GET /api/pdfs` | Return list of all registered PDFs |
| | `DELETE /api/pdfs/{id}` | Remove PDF vectors from Chroma and registry entry |
| `chat.py` | `POST /api/chat` | RAG query: retrieve → prompt → respond |
| | `DELETE /api/chat/{session_id}` | Clear in-memory conversation history |

#### Services

| Service | Technology | Role |
|---|---|---|
| `chunker.py` | PyMuPDF + LlamaIndex `SemanticSplitterNodeParser` | Extract page text from PDF; semantically split into coherent chunks |
| `embedder.py` | HuggingFace Inference API (`all-MiniLM-L6-v2`) | Generate dense vector embeddings for chunks and queries |
| `vector_store.py` | ChromaDB (persistent) | Store and retrieve chunk embeddings with metadata |
| `llm.py` | Groq API (`llama-3.3-70b-versatile`) | Generate grounded answers using retrieved context |

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
  ├─ PyMuPDF extracts text page-by-page
  └─ LlamaIndex SemanticSplitter splits into semantic chunks
      │
      ▼
Backend: embedder.py
  └─ HuggingFace Inference API → float[] embedding per chunk
      │
      ▼
Backend: vector_store.py
  └─ ChromaDB stores (chunk_text, embedding, {pdf_id, pdf_name, page_number, chunk_index})
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
vector_store.query(query_embedding, filter: active_pdf_ids)
  └─ Returns top-K chunks with page metadata
      │
      ▼
Build prompt:
  - System: "Answer only from provided context. Cite page numbers."
  - History: last N turns (keyed by session_id)
  - Context: retrieved chunks
  - User: current query
      │
      ▼
Groq LLaMA 3.3 70B → generates answer
      │
      ▼
Parse citations from response
      │
      ▼
Response: { answer, citations: [{ pdf_name, page_number }] }
      │
      ▼
Frontend: MessageBubble renders answer + citation chips
```

---

## 5. Key Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Chunking strategy | Semantic (LlamaIndex) over fixed-size | Preserves contextual coherence of chunks |
| Embedding model | `all-MiniLM-L6-v2` via HF Inference API | Lightweight, strong semantic quality; no local GPU needed |
| LLM | Groq `llama-3.3-70b-versatile` | Fast inference, free tier, strong instruction following |
| Vector store | ChromaDB (persistent local) | Zero-config, embedded, cosine similarity |
| PDF metadata | JSON file registry | Lightweight; avoids a full SQL DB for simple key-value mapping |
| Chat history | In-memory dict | Simplicity; acceptable loss on server restart |
| Session identity | UUID in `sessionStorage` | Per-tab isolation, no user auth required |
| Frontend state | React local state only | Single-page app; no complex cross-component state needed |

---

## 6. Deployment Architecture

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
                                         └────────────────────┘
```

- **Frontend** — deployed to **Vercel**; `NEXT_PUBLIC_API_URL` set to the Render backend URL.
- **Backend** — deployed to **Render** as a Python web service; `GROQ_API_KEY` and `HF_TOKEN` set as environment secrets.
- **CORS** — backend `ALLOWED_ORIGINS` env var restricts access to the Vercel domain in production.

---

## 7. External Dependencies

| Service | Purpose | Env Var |
|---|---|---|
| Groq API | LLM inference (LLaMA 3.3 70B) | `GROQ_API_KEY` |
| HuggingFace Inference API | Sentence embeddings (MiniLM) | `HF_TOKEN` |

---

## 8. Tech Stack Summary

| Layer | Technology |
|---|---|
| Frontend | Next.js 14, React 18, TypeScript, lucide-react |
| Backend | FastAPI, Python, uvicorn |
| Chunking | PyMuPDF (fitz), LlamaIndex SemanticSplitter |
| Embeddings | HuggingFace Inference API (`all-MiniLM-L6-v2`) |
| Vector Store | ChromaDB (persistent) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Frontend Hosting | Vercel |
| Backend Hosting | Render |
