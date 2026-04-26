# PDF Conversational Agent

A strictly grounded RAG conversational agent built with **Next.js** (frontend) and **FastAPI** (backend).

## 🎯 Recent Improvements

**Enhanced for evaluation criteria:**
- ✅ **Two-stage adaptive retrieval** - Better recall without sacrificing precision
- ✅ **Enhanced refusal detection** - More robust out-of-scope handling
- ✅ **Confidence scoring** - Transparent reliability indicators
- ✅ **Optimized chunking** - Better semantic boundaries and citations
- ✅ **Deterministic responses** - Temperature 0.0 for consistency

📖 **See [IMPROVEMENT_SUMMARY.md](IMPROVEMENT_SUMMARY.md) for details**

## Features
- 📄 Upload multiple PDFs
- 🔍 Semantic chunking via LlamaIndex `SemanticSplitterNodeParser`
- 🗂️ Select which PDFs the agent should use
- 💬 Conversational history (per session)
- 📎 Page-level citations for every response
- 🚫 Strict refusal for out-of-scope queries
- 🗑️ Delete PDFs on demand

## Tech Stack
| Layer | Technology |
|-------|------------|
| Frontend | Next.js 14 (TypeScript) → Vercel |
| Backend | FastAPI (Python) → Render |
| Chunking | LlamaIndex SemanticSplitterNodeParser |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | Qdrant Cloud |
| LLM | Groq — Llama 3.3 70B |

---

## Local Development

### Backend
```bash
cd backend
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env: add GROQ_API_KEY, HF_TOKEN, QDRANT_URL, and QDRANT_API_KEY
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

- **[IMPROVEMENT_SUMMARY.md](IMPROVEMENT_SUMMARY.md)** - Overview of recent enhancements
- **[IMPROVEMENTS.md](IMPROVEMENTS.md)** - Detailed technical implementation
- **[TESTING_GUIDE.md](TESTING_GUIDE.md)** - Comprehensive test cases and evaluation rubric
- **[EVALUATOR_QUICK_START.md](EVALUATOR_QUICK_START.md)** - Quick evaluation guide
- **[HLD.md](HLD.md)** - High-level design and architecture
- **[COLD_START_FEATURE.md](COLD_START_FEATURE.md)** - Cold start detection feature
