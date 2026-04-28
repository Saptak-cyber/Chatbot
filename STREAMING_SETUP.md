# Streaming Response Setup

## ✅ What Was Added

Real-time streaming responses using Server-Sent Events (SSE).

---

## 🎯 Benefits

- **Instant feedback:** Response appears immediately
- **Better UX:** Like ChatGPT/Claude
- **Same accuracy:** All intelligent retrieval features work

---

## 📁 Files Modified

### Backend

1. **`backend/services/llm.py`**
   - Added `generate_response_stream()` function
   - Yields text chunks as they arrive from Groq

2. **`backend/routers/chat.py`**
   - Added `/api/chat/stream` endpoint
   - Supports intelligent retrieval (greetings, confirmations, elaborations)
   - Returns SSE format

### Frontend

3. **`frontend/lib/api.ts`**
   - Added `sendMessageStream()` function
   - Handles SSE parsing with callbacks

4. **`frontend/components/ChatWindow.tsx`**
   - Updated `handleSend()` to use streaming
   - Updates message content in real-time

---

## 🔧 How It Works

### Backend Flow
```python
# 1. Classify query type
needs_retrieval, query_type = _is_retrieval_required(...)

# 2a. If no retrieval: stream from history
if not needs_retrieval:
    response = _generate_from_history(...)
    # Stream word by word

# 2b. If retrieval needed: stream from LLM
else:
    chunks = query_chunks(...)
    for chunk in generate_response_stream(...):
        yield chunk
```

### Frontend Flow
```typescript
// Create placeholder message
const assistantMessage = { id: '...', content: '', ... }

// As chunks arrive, append to content
onChunk: (chunk) => {
  message.content += chunk  // Real-time update!
}

// When done, add metadata
onDone: (data) => {
  message.sources_used = data.sources
  message.is_grounded = data.is_grounded
}
```

---

## 📊 Event Types

| Event | When | Data |
|-------|------|------|
| `metadata` | After retrieval | `{ retrieval_score }` |
| `chunk` | During generation | `{ content: "text" }` |
| `done` | After completion | `{ is_grounded, sources, confidence }` |
| `refusal` | Out of scope | `{ content: "refusal message" }` |
| `error` | On failure | `{ message: "error" }` |

---

## 🧪 Testing

```bash
# 1. Start backend
cd backend
uvicorn main:app --reload

# 2. Start frontend
cd frontend
npm run dev

# 3. Test
# - Upload PDF
# - Ask question
# - Watch response stream word-by-word ✨
```

---

## 🔄 Backward Compatibility

Both endpoints work:
- `/api/chat` - Non-streaming (complete response)
- `/api/chat/stream` - Streaming (SSE)

Frontend uses streaming by default.

---

## ✅ Summary

**Added:**
- ✅ Streaming LLM responses
- ✅ SSE support
- ✅ Real-time UI updates
- ✅ Works with intelligent retrieval

**Result:** Instant perceived responses! 🎉
