# Test Instructions for Evaluators

## Quick Start (5 Minutes)

### Prerequisites
- Modern web browser (Chrome, Firefox, Safari, Edge)
- Sample PDF document (10-50 pages recommended)
- Internet connection

### Access the Application

**Frontend:** [Your Vercel URL]  
**Backend:** [Your Render URL]

---

## Step-by-Step Testing Guide

### Phase 1: Setup (2 minutes)

#### 1.1 Open the Application
1. Navigate to the frontend URL
2. Wait for the page to load
3. You should see two panels:
   - **Left:** PDF Sidebar (upload and manage PDFs)
   - **Right:** Chat Window (ask questions)

#### 1.2 Upload a Test PDF
1. Click **"Upload PDF"** button in the left sidebar
2. Select a PDF file from your computer
3. Wait for upload to complete (~5-10 seconds)
4. You should see:
   - PDF name in the sidebar
   - Page count
   - Chunk count
   - Green checkmark indicating success

**Recommended Test PDFs:**
- Academic research paper
- Company annual report
- Technical documentation
- Product manual

**Note:** First upload may take longer (~10-15s) due to cold start on Render free tier.

#### 1.3 Load the PDF
1. Check the checkbox next to your uploaded PDF
2. Click **"Load Selected PDFs"** button
3. The PDF name should appear as a chip in the chat header
4. Chat input should now be enabled

---

### Phase 2: Valid Query Testing (10 minutes)

Test queries that **should** be answered correctly.

#### 2.1 Simple Factual Question

**Purpose:** Test basic retrieval and citation.

**Example Query:**
```
What is the main topic of this document?
```

**What to Check:**
- ✅ Response is accurate
- ✅ Page number cited (e.g., `[Page 1 — filename.pdf]`)
- ✅ Confidence level shown (High/Medium/Low)
- ✅ Response appears within ~2 seconds

**Expected Result:**
```
The main topic of this document is [accurate answer]. [Page X — filename.pdf]

[Sources: Page X — filename.pdf]
```

---

#### 2.2 Multi-Page Synthesis

**Purpose:** Test cross-section information retrieval.

**Example Query:**
```
What are the key findings mentioned in this report?
```

**What to Check:**
- ✅ Response synthesizes information from multiple pages
- ✅ Multiple page citations (e.g., `[Page 3, 7, 12 — filename.pdf]`)
- ✅ All cited pages are relevant
- ✅ No information from outside the PDF

**Expected Result:**
```
The key findings are:
1. [Finding 1] [Page 3 — filename.pdf]
2. [Finding 2] [Page 7 — filename.pdf]
3. [Finding 3] [Page 12 — filename.pdf]

[Sources: Pages 3, 7, 12 — filename.pdf]
```

---

#### 2.3 Terminology Variation

**Purpose:** Test semantic matching with different phrasings.

**Example Query:**
```
If PDF says "revenue": Ask "What were the earnings?"
If PDF says "ROI": Ask "What is the return on investment?"
```

**What to Check:**
- ✅ System finds the information despite different terminology
- ✅ Response uses the PDF's terminology
- ✅ Correct page citation

**Expected Result:**
System should find "revenue" when asked about "earnings".

---

#### 2.4 Follow-Up Question

**Purpose:** Test conversational memory and query rewriting.

**Example Conversation:**
```
Turn 1: "What is the methodology used?"
Turn 2: "Can you elaborate on that?"
```

**What to Check:**
- ✅ Second query understands "that" refers to methodology
- ✅ Response provides more detail
- ✅ Citations are relevant to methodology
- ✅ Conversation history is maintained

**Expected Result:**
Second response should elaborate on methodology without needing to re-specify the topic.

---

#### 2.5 Partial Information

**Purpose:** Test handling of incomplete information.

**Example Query:**
```
What is the price and warranty period?
(Assuming PDF only mentions price)
```

**What to Check:**
- ✅ Answers the part that's available (price)
- ✅ Explicitly states what's missing (warranty)
- ✅ Cites page for available information
- ✅ No fabricated information

**Expected Result:**
```
The price is $X [Page Y — filename.pdf]. However, the document does not contain information about the warranty period.
```

---

### Phase 3: Invalid Query Testing (5 minutes)

Test queries that **should** be refused.

#### 3.1 Completely Unrelated Topic

**Purpose:** Test out-of-scope refusal.

**Example Query:**
```
What's the weather like today?
```

**What to Check:**
- ✅ Clear refusal message
- ✅ `is_grounded: false` (visible as red "Out of Scope" badge)
- ✅ No attempt to answer
- ✅ Suggests asking about PDF content

**Expected Result:**
```
🚫 Out of Scope

I cannot find an answer to this question in the provided PDF(s). The documents do not contain information about this topic. Please ask something covered in the uploaded documents.
```

---

#### 3.2 Information Not in PDF

**Purpose:** Test refusal for related but absent information.

**Example Query:**
```
Who is the CFO of the company?
(Assuming CFO is not mentioned in PDF)
```

**What to Check:**
- ✅ Clear refusal
- ✅ Explicitly states information is not in the document
- ✅ No fabricated answer
- ✅ No external knowledge used

**Expected Result:**
```
🚫 Out of Scope

I cannot find an answer to this question in the provided PDF(s). The document does not mention the CFO or this specific role.
```

---

#### 3.3 Opinion/Prediction Request

**Purpose:** Test refusal for subjective queries.

**Example Query:**
```
What will happen next year?
```

**What to Check:**
- ✅ Refuses to speculate
- ✅ Only reports what's explicitly stated in PDF
- ✅ No predictions or opinions

**Expected Result:**
```
🚫 Out of Scope

I cannot find an answer to this question in the provided PDF(s). The document does not contain predictions or information about future events.
```

---

### Phase 4: Edge Cases (5 minutes)

#### 4.1 Empty Query
**Action:** Try to send an empty message  
**Expected:** Send button should be disabled

#### 4.2 No PDFs Loaded
**Action:** Clear loaded PDFs, try to send a message  
**Expected:** Input should be disabled with hint "Load PDFs from the sidebar first…"

#### 4.3 Very Long Query
**Action:** Paste a 500-word question  
**Expected:** System should handle gracefully, may take slightly longer

#### 4.4 Special Characters
**Action:** Ask a question with special characters: `What is the "revenue" (in $)?`  
**Expected:** System should handle normally

---

## Evaluation Rubric

### For Each Valid Query (Score 0-10)

**Accuracy (0-3 points)**
- 3: Completely accurate, all facts correct
- 2: Mostly accurate, minor imprecision
- 1: Partially accurate, significant errors
- 0: Incorrect or hallucinated

**Citations (0-2 points)**
- 2: All claims cited, page numbers correct
- 1: Some citations missing or incorrect
- 0: No citations or all incorrect

**Grounding (0-3 points)**
- 3: Perfectly grounded, no external knowledge
- 2: Mostly grounded, minor inference
- 1: Some hallucination or speculation
- 0: Significant hallucination

**Response Quality (0-2 points)**
- 2: Clear, well-structured, complete
- 1: Somewhat unclear or incomplete
- 0: Confusing or very incomplete

**Total: 10 points per query**

---

### For Each Invalid Query (Pass/Fail)

**Pass Criteria:**
- ✅ Clear refusal message
- ✅ `is_grounded: false` (red badge)
- ✅ No hallucinated information
- ✅ Helpful explanation

**Fail Criteria:**
- ❌ Attempts to answer (should refuse)
- ❌ Fabricates information
- ❌ Uses external knowledge
- ❌ Vague or unhelpful refusal

---

## Expected Performance Benchmarks

### Valid Queries

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Average Score | ≥8.5/10 | 7.0-8.4 | <7.0 |
| Accuracy 3/3 | ≥80% | 60-79% | <60% |
| Citations 2/2 | ≥90% | 70-89% | <70% |
| Grounding 3/3 | ≥85% | 65-84% | <65% |

### Invalid Queries

| Metric | Target | Acceptable | Poor |
|--------|--------|------------|------|
| Refusal Rate | 100% | ≥90% | <90% |
| No Hallucination | 100% | ≥95% | <95% |

---

## Common Issues & Troubleshooting

### Issue: "Cold start detected" banner appears

**Cause:** Backend was sleeping (Render free tier)  
**Solution:** Wait 10-15 seconds for backend to wake up  
**Note:** This is expected on first request after inactivity

---

### Issue: Upload fails with timeout

**Cause:** Large PDF or slow connection  
**Solution:** Try a smaller PDF (<10MB) or check internet connection

---

### Issue: Response takes >5 seconds

**Cause:** Cold start or high load  
**Solution:** Subsequent requests should be faster (~1-2s)

---

### Issue: Citations show wrong page numbers

**Cause:** PDF page numbering mismatch  
**Solution:** Verify page numbers manually in PDF (system uses 1-indexed)

---

### Issue: Response is empty or error message

**Cause:** Backend error or API rate limit  
**Solution:** Check browser console (F12) for error details, try again in 1 minute

---

## Advanced Testing (Optional)

### Test Streaming Responses

**Purpose:** Verify real-time response streaming.

**Action:**
1. Ask a question
2. Watch the response appear word-by-word
3. Citations should appear after response completes

**Expected:** Response should start appearing within ~500ms (after retrieval).

---

### Test Conversation History

**Purpose:** Verify conversation persistence.

**Action:**
1. Have a multi-turn conversation (5+ exchanges)
2. Refresh the page
3. Check if conversation is restored

**Expected:** Conversation should be restored from localStorage.

---

### Test Multiple PDFs

**Purpose:** Verify multi-document querying.

**Action:**
1. Upload 2-3 PDFs
2. Load all of them
3. Ask a question that requires information from multiple PDFs

**Expected:** Response should cite pages from different PDFs.

---

### Test Clear Chat

**Purpose:** Verify conversation reset.

**Action:**
1. Have a conversation
2. Click "Clear" button
3. Confirm the dialog
4. Check if messages are cleared

**Expected:** All messages should disappear, chat history deleted.

---

### Test PDF Deletion

**Purpose:** Verify PDF removal.

**Action:**
1. Upload a PDF
2. Click the trash icon next to it
3. Confirm deletion
4. Check if PDF is removed from sidebar

**Expected:** PDF should be removed, vector data deleted.

---

## Verification Checklist

Use this checklist to ensure comprehensive testing:

### Setup
- [ ] Application loads successfully
- [ ] PDF upload works
- [ ] PDF appears in sidebar with correct metadata
- [ ] Load PDFs button works
- [ ] Chat input is enabled after loading

### Valid Queries
- [ ] Simple factual questions answered correctly
- [ ] Multi-page synthesis works
- [ ] Terminology variations handled
- [ ] Follow-up questions work
- [ ] Partial information handled gracefully

### Invalid Queries
- [ ] Unrelated topics refused
- [ ] Missing information refused
- [ ] Opinion requests refused
- [ ] No hallucinated information

### Citations
- [ ] All responses have page citations
- [ ] Page numbers are correct
- [ ] Citation format is consistent
- [ ] Multiple pages cited when appropriate

### Confidence Indicators
- [ ] Confidence level shown (High/Medium/Low)
- [ ] High confidence → accurate answers
- [ ] Low confidence → uncertain or refused
- [ ] Out of scope badge for refusals

### User Experience
- [ ] Responses appear within ~2 seconds
- [ ] Streaming works (if enabled)
- [ ] Auto-focus on keypress works
- [ ] Conversation history persists
- [ ] Clear chat works
- [ ] PDF deletion works

---

## Test Report Template

Use this template to document your findings:

```markdown
# Test Report: PDF Conversational Agent

**Date:** [Date]
**Tester:** [Name]
**PDF Used:** [PDF name, pages, topic]

## Summary
- Valid Queries Tested: X/5
- Invalid Queries Tested: X/3
- Average Score (Valid): X.X/10
- Refusal Rate (Invalid): X%

## Detailed Results

### Valid Queries

| # | Query | Score | Accuracy | Citations | Grounding | Quality | Notes |
|---|-------|-------|----------|-----------|-----------|---------|-------|
| 1 | ... | 9/10 | 3/3 | 2/2 | 3/3 | 2/2 | Perfect |
| 2 | ... | 7/10 | 2/3 | 2/2 | 3/3 | 2/2 | Minor error |
| 3 | ... | 8/10 | 3/3 | 2/2 | 2/3 | 2/2 | Slight inference |
| 4 | ... | 9/10 | 3/3 | 2/2 | 3/3 | 2/2 | Excellent |
| 5 | ... | 8/10 | 3/3 | 1/2 | 3/3 | 2/2 | Missing citation |

**Average: X.X/10**

### Invalid Queries

| # | Query | Refused? | Hallucination? | Quality | Notes |
|---|-------|----------|----------------|---------|-------|
| 1 | ... | Yes | No | Clear | Perfect refusal |
| 2 | ... | Yes | No | Clear | Good explanation |
| 3 | ... | Yes | No | Clear | Helpful |

**Refusal Rate: X%**

## Issues Found
1. [Issue description]
2. [Issue description]

## Strengths
1. [Strength description]
2. [Strength description]

## Recommendations
1. [Recommendation]
2. [Recommendation]

## Overall Assessment
[Pass/Fail] — [Brief explanation]
```

---

## Quick Test Script (Minimal)

If time is limited, use this minimal test set:

### 3 Valid Queries
1. **Simple fact:** "What is the main topic?"
2. **Multi-page:** "What are the key findings?"
3. **Follow-up:** "Can you elaborate on that?"

### 2 Invalid Queries
1. **Unrelated:** "What's the weather?"
2. **Not in PDF:** "Who is the CEO?" (if not mentioned)

### Pass Criteria
- Valid: ≥2/3 score ≥8/10
- Invalid: 2/2 refuse correctly
- Citations: All correct

**Time Required:** ~5 minutes

---

## Additional Resources

### Browser Developer Tools
- **Console (F12):** Check for errors
- **Network Tab:** Inspect API requests/responses
- **Application Tab:** View localStorage (conversation history)

### Backend Logs
- **Render Dashboard:** View backend logs for debugging
- **LangSmith:** View traces for retrieval and generation (if configured)

### API Endpoints (for manual testing)
- `GET /health` — Health check
- `GET /api/pdfs` — List uploaded PDFs
- `POST /api/chat` — Send message (non-streaming)
- `POST /api/chat/stream` — Send message (streaming)

---

## Success Criteria

The system **passes evaluation** if:

1. ✅ **Valid Queries:** ≥80% score ≥8/10
2. ✅ **Invalid Queries:** ≥90% refuse correctly
3. ✅ **Citations:** ≥90% have correct page numbers
4. ✅ **Hallucinations:** <5% of responses
5. ✅ **Confidence Calibration:** High confidence → >95% accurate

---

## Contact & Support

If you encounter issues during testing:

1. **Check browser console** (F12) for error messages
2. **Try refreshing** the page
3. **Wait 1 minute** if rate limited
4. **Try a different PDF** if upload fails
5. **Check backend status** on Render dashboard

---

## Final Notes

### What Makes This System Good?

1. **Strictly Grounded:** Never uses external knowledge
2. **Precise Citations:** Every claim has a page reference
3. **Robust Refusal:** Clearly refuses out-of-scope queries
4. **Conversational:** Handles follow-ups naturally
5. **Transparent:** Confidence levels show reliability

### What to Look For

**Good Signs ✅**
- High retrieval scores (>0.30)
- Clear, specific answers
- Multiple page citations
- Appropriate refusals
- Consistent confidence levels

**Red Flags 🚩**
- Vague or generic answers
- Missing citations
- Incorrect page numbers
- Hallucinated facts
- Attempts to answer out-of-scope queries

---

## Ready to Test!

Follow the phases in order:
1. **Setup** (2 min) — Upload and load PDF
2. **Valid Queries** (10 min) — Test correct answers
3. **Invalid Queries** (5 min) — Test refusals
4. **Edge Cases** (5 min) — Test boundaries

**Total Time:** ~20-25 minutes for comprehensive testing

**Good luck!** 🚀
