"""
FastAPI application entry point.
Pre-loads embedding model, chunker, and Qdrant client on startup to avoid cold-start latency.
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm all services on startup."""
    logger.info("=== PDF Agent API starting up ===")
    try:
        from services.embedder import get_embed_model
        from services.chunker import get_splitter
        from services.vector_store import get_client

        logger.info("Loading embedding model...")
        get_embed_model()
        logger.info("Loading semantic chunker...")
        get_splitter()
        logger.info("Initializing vector store...")
        get_client()

        # Ensure Neon chat_messages table exists (idempotent)
        neon_url = os.getenv("NEON_DATABASE_URL")
        if neon_url:
            import psycopg
            from langchain_postgres import PostgresChatMessageHistory
            with psycopg.connect(neon_url) as conn:
                PostgresChatMessageHistory.create_tables(conn, "chat_messages")
            logger.info("Neon chat_messages table ready.")
        else:
            logger.warning("NEON_DATABASE_URL not set — conversation history will not be persisted.")

        logger.info("=== All services ready. API is live. ===")
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise

    yield

    logger.info("=== PDF Agent API shutting down ===")


app = FastAPI(
    title="PDF Conversational Agent API",
    description=(
        "A strictly grounded RAG API. Upload PDFs, select which ones to query, "
        "and get answers with page-level citations. Out-of-scope questions are refused."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS — allow all origins for development; restrict to Vercel URL in production via env var
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "*")
allowed_origins = [o.strip() for o in allowed_origins_str.split(",")] if allowed_origins_str != "*" else ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
from routers import pdfs, chat  # noqa: E402  (imported after app init)

app.include_router(pdfs.router, prefix="/api")
app.include_router(chat.router, prefix="/api")


@app.get("/", tags=["Health"])
def health_check():
    return {"status": "ok", "message": "PDF Agent API is running"}


@app.get("/health", tags=["Health"])
def health():
    return {"status": "healthy"}
