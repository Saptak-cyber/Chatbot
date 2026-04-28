# PDF Conversational Agent

A strictly grounded RAG conversational agent built with **Next.js** (frontend) and **FastAPI** (backend).

## 🎯 Recent Improvements

**Enhanced for evaluation criteria:**
- ✅ **Enhanced refusal detection** - More robust out-of-scope handling
- ✅ **Confidence scoring** - Transparent reliability indicators
- ✅ **Optimized chunking** - Better semantic boundaries and citations
- ✅ **Deterministic responses** - Temperature 0.0 for consistency

📖 **See [IMPROVEMENT_SUMMARY.md](IMPROVEMENT_SUMMARY.md) for details**

## Features
- 📄 Upload multiple PDFs
- 🔍 Semantic chunking via LlamaIndex `SemanticSplitterNodeParser` (threshold=88)
- 🗂️ Select which PDFs the agent should use
- 💬 Multi-session conversation threads (ChatGPT-style, persisted in `localStorage`)
- 📎 Page-level citations for every response
- 🚫 Strict refusal for out-of-scope queries
- 🌍 Multi-language responses (EN, DE, FR, IT, PT, HI, ES, TH)
- 📋 Copy-to-clipboard on assistant messages
- 📊 Markdown table rendering
- 🗑️ Delete PDFs on demand

## Tech Stack
| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14 (TypeScript) → Vercel |
| Backend | FastAPI (Python) → Render |
| Chunking | LlamaIndex `SemanticSplitterNodeParser` (threshold=88) |
| Embeddings | HuggingFace Inference API — `BAAI/bge-small-en-v1.5` |
| Vector DB | Qdrant Cloud |
| LLM | Groq — Llama 3.1 8B Instant |
| Observability | LangSmith |

---

## Local Development

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add GROQ_API_KEY, HF_TOKEN, QDRANT_URL, QDRANT_API_KEY,
# and optionally LANGCHAIN_API_KEY for LangSmith tracing
uvicorn main:app --reload
```
Backend runs at `http://localhost:8000`.

### Frontend
```bash
cd frontend
npm install
cp .env.local.example .env.local
# .env.local already points to localhost:8000
npm run dev
```
Frontend runs at `http://localhost:3000`.

---

## Deployment

### Backend → Render
1. Push to GitHub
2. Create a new **Web Service** on [render.com](https://render.com)
3. Point it to the `backend/` directory (set **Root Directory** to `backend`)
4. Set environment variables:
   - `GROQ_API_KEY` — from [console.groq.com](https://console.groq.com)
   - `HF_TOKEN` — from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - `QDRANT_URL` — from [cloud.qdrant.io](https://cloud.qdrant.io) (your cluster URL)
   - `QDRANT_API_KEY` — from [cloud.qdrant.io](https://cloud.qdrant.io) (your API key)
   - `NEXT_PUBLIC_API_URL` - https://your-service-name.onrender.com
5. Build & start commands auto-detected from `render.yaml`

### Frontend → Vercel
1. Import repo to [vercel.com](https://vercel.com)
2. Set **Root Directory** to `frontend`
3. Add environment variable:
   - `NEXT_PUBLIC_API_URL` = `https://your-render-app.onrender.com`
4. Deploy

---

## API Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `POST` | `/api/upload` | Upload & index a PDF |
| `GET` | `/api/pdfs` | List all PDFs |
| `DELETE` | `/api/pdfs/{id}` | Delete a PDF |
| `POST` | `/api/chat` | RAG chat message |
| `DELETE` | `/api/chat/{session_id}` | Clear chat history |

## Required API Keys
- **GROQ_API_KEY**: Get from [console.groq.com](https://console.groq.com/)
- **HF_TOKEN**: Get from [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
- **QDRANT_URL**: Get from [cloud.qdrant.io](https://cloud.qdrant.io) (your cluster URL)
- **QDRANT_API_KEY**: Get from [cloud.qdrant.io](https://cloud.qdrant.io) (your API key)

---

## 📚 Documentation

- **[HLD.md](HLD.md)** — High-level design and architecture
- **[Short_Technical_Note.md](Short_Technical_Note.md)** — Architecture, design decisions, trade-offs
- **[technical_note.md](technical_note.md)** — Detailed technical reference
- **[test_instruction.md](test_instruction.md)** — Evaluator test guide with TechNova-specific queries
