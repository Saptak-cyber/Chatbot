# RAG Chat Query Pipeline — Detailed Architecture

> **System:** DocMind PDF Conversational Agent  
> **Endpoint:** `POST /api/chat/stream`  
> **Protocol:** Server-Sent Events (SSE) — tokens streamed token-by-token  
> **Model:** Groq `llama-3.1-8b-instant`  
> **Embeddings:** HuggingFace Inference API — `BAAI/bge-small-en-v1.5`

---

## 1. High-Level Pipeline Overview

```mermaid
flowchart TD
    A([👤 User Message]) --> B[POST /api/chat/stream\nwith session_id · active_pdf_ids · response_language]
    B --> C[Load Conversation History\nfrom Neon PostgreSQL]
    C --> D{Query Classification\n_is_retrieval_required}

    D -- "greeting / confirmation\nclarification / history_based" --> E[Branch A\n_generate_from_history]
    D -- "new_question / elaboration" --> F[Branch B\nRAG Retrieval Pipeline]

    E --> E1[Stream word-by-word\nwith grounding constraint]
    E1 --> E2[SSE: type=chunk events]
    E2 --> E3[SSE: type=done\nis_grounded · sources=[]]
    E3 --> Z[Persist Turn → Neon]

    F --> F1[Query Rewriting\n_rewrite_query]
    F1 --> F2[Semantic Retrieval\nquery_chunks]
    F2 --> F3{Hard Refusal Gate\nchunks found?}
    F3 -- "0 chunks above\nmin_score=0.20" --> F4[SSE: type=refusal\nLocalised message]
    F3 -- "chunks retrieved" --> F5[LLM Generation\ngenerate_response_stream]
    F5 --> F6{Tag Detection\n[GROUNDED] or [REFUSED]?}
    F6 -- "[GROUNDED]" --> F7[Stream chunks\nSSE: type=chunk]
    F6 -- "[REFUSED]" --> F8[Buffer silently\nSSE: type=refusal]
    F7 --> F9[SSE: type=done\ncitations · confidence · score]
    F8 --> F9
    F4 --> Z
    F9 --> Z

    style A fill:#6c63ff,color:#fff
    style Z fill:#22c55e,color:#fff
    style F4 fill:#ef4444,color:#fff
    style F8 fill:#ef4444,color:#fff
    style D fill:#f59e0b,color:#fff
    style F3 fill:#f59e0b,color:#fff
    style F6 fill:#f59e0b,color:#fff
```

---

## 2. Step-by-Step Breakdown

### Step 0 — Request Ingestion

```
POST /api/chat/stream
{
  "message":           "What is the notice period for SB4?",
  "session_id":        "550e8400-e29b-41d4-a716-446655440000",
  "active_pdf_ids":    ["pdf_abc123"],
  "response_language": "en"          ← BCP-47 code; "auto" = match user's language
}
```

**Validation gates (before any ML work):**

| Check | Limit | Error |
|---|---|---|
| `active_pdf_ids` must be non-empty | — | HTTP 400 |
| `message` must be non-empty | — | HTTP 400 |
| `message` length | ≤ 2 000 chars | HTTP 400 |

---

### Step 1 — Load Conversation History

```mermaid
sequenceDiagram
    participant R as chat.py Router
    participant N as Neon PostgreSQL
    participant H as _history_to_dicts()

    R->>N: PostgresChatMessageHistory(table, session_id)
    N-->>R: [SystemMessage?, HumanMessage*, AIMessage*]
    R->>H: convert to role/content dicts
    H-->>R: [{"role":"user","content":"..."}, ...]
    Note over H: SystemMessage (rolling summary) →<br/>injected as synthetic user/assistant pair<br/>so every LLM API accepts it
```

**History schema after conversion:**
```python
[
  # Optional rolling summary (compressed older turns)
  {"role": "user",      "content": "[Summary of our earlier conversation]\n..."},
  {"role": "assistant", "content": "Understood. I have the full context..."},

  # Recent verbatim turns (max 5 exchanges)
  {"role": "user",      "content": "What is the L&D budget?"},
  {"role": "assistant", "content": "The annual L&D budget is $2,000 [Page 13 — handbook.pdf]"},
]
```

> **Rolling summarisation** (`_maybe_summarize`) fires every 5 complete exchanges: the oldest messages are collapsed into a `SystemMessage` in Neon to keep the token footprint bounded.

---

### Step 2 — Query Classification

```mermaid
flowchart LR
    Q["User query\n(any language)"] --> CL["LLM Classifier\nllama-3.1-8b-instant\ntemp=0.0 · max_tokens=20"]
    CL --> T{Query Type}

    T --> G["greeting\n✗ no retrieval"]
    T --> C["confirmation\n✗ no retrieval"]
    T --> CL2["clarification\n✗ no retrieval"]
    T --> H["history_based\n✗ no retrieval"]
    T --> E["elaboration\n✓ retrieval"]
    T --> N["new_question\n✓ retrieval"]

    style G fill:#22c55e,color:#fff
    style C fill:#22c55e,color:#fff
    style CL2 fill:#22c55e,color:#fff
    style H fill:#22c55e,color:#fff
    style E fill:#3b82f6,color:#fff
    style N fill:#3b82f6,color:#fff
```

**Classifier safety rules (added to prompt):**
- Any **factual question** (colour, number, name, definition) NOT in history → `new_question`
- Non-English / Hinglish queries → `new_question` unless clearly referencing prior history
- Unrecognised output → default `new_question` (over-retrieve, never hallucinate)
- Output parsed as `.split()[0]` — only first word taken to guard against multi-word responses

**Special case — no history (first message):**  
Pattern-matched against a list of English greeting keywords (`hello`, `hi`, `thank you`, …). Anything else → `new_question` immediately, no LLM call.

---

### Branch A — History-Based Response

```mermaid
sequenceDiagram
    participant R as chat.py
    participant G as _generate_from_history()
    participant L as Groq LLM
    participant C as SSE Client

    R->>G: query · history · query_type · language
    G->>G: build system_prompt for query_type<br/>+ grounding_constraint (no world knowledge)<br/>+ lang_note
    G->>L: chat.completions.create(temp=0.3, max_tokens=300)
    L-->>G: response_text
    G-->>R: (response_text, is_grounded)
    loop word by word
        R->>C: data: {"type":"chunk","content":"word "}
        Note right of R: asyncio.sleep(0.01) between words
    end
    R->>C: data: {"type":"done","is_grounded":true,"sources":[]}
    R->>R: _persist_turn(session_id, query, response)
```

**Grounding constraint (appended to every branch):**
> *"Do NOT use general world knowledge. If the user's question asks for factual information NOT present in the conversation history, respond ONLY with: 'I can only answer questions based on the uploaded PDF documents.'"*

---

### Branch B — RAG Retrieval Pipeline

#### B1 — Query Rewriting

```mermaid
flowchart TD
    Q[User query] --> QT{query_type?}
    QT -- "new_question" --> SKIP[Skip rewrite\nuse original query]
    QT -- "elaboration" --> RW["_rewrite_query()\nllama-3.1-8b-instant\ntemp=0.0 · max_tokens=150"]
    RW --> |"Uses last 4 messages\n(2 turns) as context"| OUT[Rewritten standalone query]
    SKIP --> OUT
```

**Examples:**

| Original | Rewritten |
|---|---|
| `"Tell me more"` | `"Tell me more about the L&D budget and what it covers"` |
| `"What about SB5?"` | `"What is the notice period for an SB5 role?"` |
| `"What is the health policy?"` | *(unchanged — new topic)* |

> The rewriter is **skipped entirely for `new_question`** — unrelated queries must not be polluted with prior context.

---

#### B2 — Semantic Retrieval

```mermaid
sequenceDiagram
    participant C as chat.py
    participant E as embedder.py
    participant Q as Qdrant Cloud
    participant VS as vector_store.py

    C->>VS: query_chunks(query, pdf_ids, top_k=10, min_score=0.20)
    VS->>E: embed_query(query)
    Note over E: Prepend BGE instruction prefix:<br/>"Represent this sentence for searching<br/>relevant passages: {query}"
    E->>E: HuggingFace Inference API<br/>BAAI/bge-small-en-v1.5 → float[384]
    E-->>VS: query_vector
    VS->>Q: search(collection, query_vector,\n  filter={pdf_id ∈ active_pdf_ids},\n  limit=top_k, score_threshold=min_score)
    Q-->>VS: [{id, score, payload: {text, pdf_name, page, section}}]
    VS-->>C: ranked chunks (cosine score ↓)
```

**Chunk payload structure:**
```python
{
  "text":     "13.2 Involuntary Termination\nIn cases of termination without cause...",
  "score":    0.743,
  "metadata": {
    "pdf_id":       "pdf_abc123",
    "pdf_name":     "TechNova_Employee_Handbook.pdf",
    "page_number":  17,
    "chunk_index":  3,
    "section":      "13. Separation from Employment"
  }
}
```

---

#### B3 — Hard Refusal Gate

```mermaid
flowchart LR
    CH{chunks\nreturned?} -- "len == 0\n0 chunks above\nmin_score=0.20" --> R["get_hard_refusal_text(language)\nPre-translated in 8 languages\nEN · DE · FR · IT · PT · HI · ES · TH"]
    R --> SSE1["SSE: type=refusal\ncontent=localised_message"]
    SSE1 --> SSE2["SSE: type=done\nis_grounded=false · sources=[]"]
    SSE2 --> P[Persist turn]

    CH -- "≥ 1 chunk\nabove threshold" --> LLM[Continue to LLM]

    style R fill:#ef4444,color:#fff
    style SSE1 fill:#ef4444,color:#fff
```

> This gate fires **before the LLM is ever called** — eliminating hallucination on out-of-scope queries entirely.

---

#### B4 — LLM Generation & Streaming

```mermaid
sequenceDiagram
    participant C as chat.py
    participant L as llm.py
    participant G as Groq API
    participant FE as Browser (SSE)

    C->>L: generate_response_stream(query, chunks, history, language)
    L->>L: Build messages list:
    Note over L: 1. System: 7 rules + citation format<br/>   + output formatting rules<br/>   + LANGUAGE section<br/>2. History: summary + recent turns<br/>3. User: chunks + query + tag instruction

    L->>G: chat.completions.create(stream=True,\n  model=llama-3.1-8b-instant,\n  temp=0.0, max_tokens=1536)

    loop Token stream
        G-->>L: delta.content (token)
        L->>L: Accumulate in full_response_raw
        alt Tag not yet stripped
            L->>L: lstrip() buffer, check for [GROUNDED]/[REFUSED]
            opt [GROUNDED] complete
                L->>L: is_grounded_from_tag = True
                L-->>C: yield clean_start (text after tag)
                C-->>FE: data: {"type":"chunk","content":"..."}
            end
            opt [REFUSED] complete
                L->>L: is_grounded_from_tag = False
                Note over L: Buffer silently — do NOT yield
            end
        else Tag stripped, is_grounded = True
            L-->>C: yield token
            C-->>FE: data: {"type":"chunk","content":"token"}
            C->>C: asyncio.sleep(0) — yield to event loop
        else Tag stripped, is_grounded = False (REFUSED)
            L->>L: buffer token silently
        end
    end

    L-->>C: StopIteration(value=(final_text, is_grounded))
```

---

#### B5 — Post-Stream: Refusal vs Grounded Path

```mermaid
flowchart TD
    SI[StopIteration captured\nreturned_response · is_grounded] --> CHK{is_grounded?}

    CHK -- "False\nsoft refusal" --> SR["SSE: type=refusal\ncontent=full_response\n(LLM's refusal message)"]
    SR --> D1["SSE: type=done\nis_grounded=false\nsources=[] · num_sources=0"]

    CHK -- "True" --> CIT["_build_citations(chunks)\nDeduplicate by (pdf_name, page)\nKeep highest score per page"]
    CIT --> CONF{retrieval_score}
    CONF -- "≥ 0.40" --> H[confidence='high']
    CONF -- "0.28–0.39" --> M[confidence='medium']
    CONF -- "< 0.28" --> LO[confidence='low']
    H & M & LO --> D2["SSE: type=done\nis_grounded=true\nsources=[Citation...]\nconfidence_level · num_sources\nretrieval_score"]

    D1 --> P[_persist_turn]
    D2 --> P

    style SR fill:#ef4444,color:#fff
    style D1 fill:#ef4444,color:#fff
    style D2 fill:#22c55e,color:#fff
```

---

## 3. SSE Event Reference

The browser receives a stream of `text/event-stream` events:

| `type` | When | Key Fields |
|---|---|---|
| `metadata` | After retrieval, before LLM | `retrieval_score` |
| `chunk` | Each streamed token | `content` |
| `refusal` | Hard or soft refusal | `content` (localised message) |
| `done` | Stream complete | `is_grounded`, `sources`, `confidence_level`, `num_sources`, `retrieval_score` |
| `error` | Exception in pipeline | `message` |

**Example stream (grounded answer):**
```
data: {"type":"metadata","retrieval_score":0.743}

data: {"type":"chunk","content":"The notice period for an SB4 role is "}
data: {"type":"chunk","content":"**4 weeks**"}
data: {"type":"chunk","content":" [Page 17 — TechNova_Employee_Handbook.pdf]."}

data: {"type":"done","is_grounded":true,"sources":[{"pdf_name":"TechNova_Employee_Handbook.pdf","page_number":17,"section":"13. Separation from Employment","score":0.743}],"confidence_level":"high","num_sources":1,"retrieval_score":0.743}
```

**Example stream (hard refusal — French):**
```
data: {"type":"refusal","content":"Je suis désolé, mais cette question ne semble pas être couverte par le(s) PDF téléchargé(s)..."}

data: {"type":"done","is_grounded":false,"sources":[],"confidence_level":null,"num_sources":0}
```

---

## 4. Anti-Hallucination Defense Layers

```mermaid
flowchart LR
    Q([Query]) --> L0

    subgraph L0["Layer 0 — Index Time"]
        direction TB
        I1[Contextual header\ninjected into every chunk]
        I2[Semantic boundaries\npreserve meaning]
        I3[Cross-page overlap\nprevents split paragraphs]
    end

    L0 --> L1

    subgraph L1["Layer 1 — Classification"]
        direction TB
        C1[Factual queries forced\nto retrieval path]
        C2[Non-English queries\ndefault to new_question]
        C3[Grounding constraint\nin history branch]
    end

    L1 --> L2

    subgraph L2["Layer 2 — Retrieval Gate"]
        direction TB
        R1[cosine min_score=0.20\nhard threshold]
        R2[0 chunks → deterministic\nrefusal without LLM call]
        R3[Localised refusal\nin user's language]
    end

    L2 --> L3

    subgraph L3["Layer 3 — Generation"]
        direction TB
        G1[7 explicit grounding rules\nin system prompt]
        G2["[GROUNDED]/[REFUSED]\nstructural tag protocol"]
        G3[Soft refusal detection\nvia tag — not regex]
    end

    L3 --> A([Answer or Refusal])

    style L0 fill:#1e3a5f,color:#fff
    style L1 fill:#1e3a5f,color:#fff
    style L2 fill:#7f1d1d,color:#fff
    style L3 fill:#14532d,color:#fff
```

---

## 5. Token Budget at Each Stage

| Stage | Model | `max_tokens` | `temperature` | Purpose |
|---|---|---|---|---|
| Query classification | `llama-3.1-8b-instant` | 20 | 0.0 | One-word output |
| Query rewriting | `llama-3.1-8b-instant` | 150 | 0.0 | Standalone query |
| History response | `llama-3.1-8b-instant` | 300 | 0.3 | Short, warm reply |
| RAG answer (stream) | `llama-3.1-8b-instant` | 1 536 | 0.0 | Full grounded answer |
| Rolling summary | `llama-3.1-8b-instant` | 1 024 | 0.0 | Compress old turns |
| Summary compression | `llama-3.1-8b-instant` | 512 | 0.0 | Re-compress if >800 tok |

> **Groq free tier limit:** 6 000 TPM. The flat semantic chunking strategy (threshold=88, top_k=10) was specifically chosen to keep the total prompt well under this limit.

---

## 6. Full Pipeline Sequence (Happy Path)

```mermaid
sequenceDiagram
    actor U as User
    participant FE as Next.js Frontend
    participant API as FastAPI /chat/stream
    participant DB as Neon PostgreSQL
    participant CLS as Classifier LLM
    participant RW as Rewriter LLM
    participant QD as Qdrant Cloud
    participant HF as HuggingFace API
    participant GR as Groq LLM (streamed)

    U->>FE: Types message
    FE->>API: POST /api/chat/stream (SSE request)
    API->>DB: Load session history
    DB-->>API: [summary?, recent turns]
    API->>CLS: Classify query type
    CLS-->>API: "new_question"
    API->>RW: Rewrite query (if elaboration; else skip)
    RW-->>API: standalone_query
    API->>HF: Embed standalone_query (BGE + prefix)
    HF-->>API: float[384] query vector
    API->>QD: Cosine search (top_k=10, min_score=0.20)
    QD-->>API: ranked chunks
    API-->>FE: data: {"type":"metadata","retrieval_score":0.74}
    API->>GR: Stream: system + history + chunks + query
    loop Streaming tokens
        GR-->>API: token delta
        API-->>FE: data: {"type":"chunk","content":"token"}
    end
    GR-->>API: StopIteration(final_text, is_grounded=True)
    API-->>FE: data: {"type":"done","sources":[...],"confidence":"high"}
    API->>DB: Persist turn (user + assistant)
    API->>DB: Maybe summarise (every 5 turns)
```
