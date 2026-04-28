# Test Instructions for Evaluators — DocMind PDF Conversational Agent

## Prerequisites

| Requirement | Detail |
|---|---|
| Test PDF | Any multi-section document (10–50 pages). Suggested: a company policy handbook, academic paper, or product manual. The evaluator should read it briefly before testing. |
| Browser | Chrome or Firefox (latest) |
| URL | `https://chatbot-iota-orpin-68.vercel.app/` or `http://localhost:3000` for local |
| Backend health | Visit `https://chatbot-jfcb.onrender.com/health` — must return `{"status": "ok"}` |

---

## Setup

1. Open the app. The **Documents** tab is visible in the left sidebar.
2. Click **Upload PDF** and select your test document.
3. Wait for the success toast: *"✓ filename.pdf — N chunks ready"*
4. Check the checkbox next to the uploaded PDF.
5. Click **Load Selected PDFs**. The button should change to **Loaded (1)**.
6. Switch to the **Chats** tab. A "New conversation" has been created automatically.

---

## Test Suite

### Category A — Factual Retrieval (Grounding)

These tests verify that the agent answers correctly and cites sources.

| # | Query | Expected behaviour |
|---|---|---|
| A1 | Ask a direct factual question clearly answered in the PDF (e.g., *"What is the notice period for SB3 roles?"*) | Response contains the correct fact + `[Page N — filename.pdf]` citation |
| A2 | Ask for a list of items covered in a specific section | Response uses bullet points, each with a citation |
| A3 | Ask the same question twice in the same conversation | Second response says *"As I mentioned…"* and references the earlier answer |
| A4 | Ask a question whose answer spans two adjacent pages | Response synthesises both pages and cites both page numbers |

**Pass criteria:** Every factual claim has a verifiable `[Page N — filename.pdf]` inline citation. Cross-reference at least 3 claims against the source PDF.

---

### Category B — Refusal / Out-of-Scope

These tests verify the agent does NOT hallucinate.

| # | Query | Expected behaviour |
|---|---|---|
| B1 | Ask something completely unrelated to the document (e.g., *"What is the capital of France?"*) | Agent refuses: *"I cannot find an answer to this question in the provided PDF(s)."* |
| B2 | Ask about a topic plausibly related but not in the document (e.g., *"What is the CEO's salary?"* for an HR handbook) | Agent refuses; does NOT invent a number |
| B3 | Ask a leading question trying to get the agent to confirm false information (e.g., *"The handbook says PTO is 30 days, right?"* when it isn't) | Agent corrects the premise with the actual figure and citation |

**Pass criteria:** Zero fabricated facts. All refusals are polite and specific.

---

### Category C — Conversation Continuity

| # | Query sequence | Expected behaviour |
|---|---|---|
| C1 | Ask *"What are the main sections?"* → follow up *"Tell me more about the second one"* | Agent understands the pronoun reference and elaborates on the correct section |
| C2 | Ask a question → then ask *"Can you explain that more simply?"* | Agent rephrases without losing the original citations |
| C3 | Ask for a summary → then ask *"Which page was that on?"* | Agent recalls and states the page number from its previous answer |

**Pass criteria:** Context is correctly maintained across at least 4 turns.

---

### Category D — Multi-Session Threads

| # | Action | Expected behaviour |
|---|---|---|
| D1 | Click **+ New conversation** | A fresh chat opens with no prior messages; previous conversation still visible in the list |
| D2 | Ask a question in the new session | Answer is independent; does not bleed context from the other session |
| D3 | Click the first conversation in the list | Previous messages reload correctly |
| D4 | Hover over a conversation → click the pencil icon → type a new name → press Enter | Name updates immediately in the list |
| D5 | Hover over a conversation → click the trash icon → confirm | Conversation removed; if it was active, the next conversation becomes active |

**Pass criteria:** All 5 actions complete without errors.

---

### Category E — Language Support

| # | Query | Expected behaviour |
|---|---|---|
| E1 | Click the language selector → choose **Français** → ask a factual question | Entire response is in French; citations remain in the form `[Page N — filename.pdf]` |
| E2 | Switch to **हिंदी** → ask the same question | Response is in Hindi (Devanagari script); citations still in Latin script |
| E3 | Switch back to **Auto-detect** → ask in Spanish | Response is in Spanish |

**Pass criteria:** Language switches take effect immediately with no page reload. Citations are always preserved in their standard format.

---

### Category F — UI & Formatting

| # | Check | Expected behaviour |
|---|---|---|
| F1 | Hover over an assistant message | A **Copy** button appears; clicking it copies the text; button shows *✓ Copied* for 2 seconds |
| F2 | Ask a question that produces a bullet list | Bullets are rendered as proper `•` list items, not raw `* text` |
| F3 | Ask a question that produces a table | Table renders with alternating row shading and is horizontally scrollable |
| F4 | Open the language dropdown | It appears **above** the messages, not behind them |
| F5 | Resize the browser to 1280 × 800 | Layout is not broken; sidebar and chat panel remain usable |

---

### Category G — Stress & Edge Cases

| # | Action | Expected behaviour |
|---|---|---|
| G1 | Upload a second PDF → load both → ask a cross-document question | Response draws from whichever document is relevant; cites both if applicable |
| G2 | Ask a very long multi-part question | Agent answers each part or explicitly states which parts are not covered |
| G3 | Send an empty message (just spaces) | Input is rejected; no API call is made |
| G4 | Ask a question immediately after uploading (before loading) | Agent warns that no PDFs are loaded |
| G5 | Refresh the browser | Conversations and messages reload from localStorage; active session restored |

---

### Category H — Document-Specific Queries (TechNova Employee Handbook)

Use **`TechNova_Employee_Handbook.pdf`** for this category.  
All expected answers and page numbers below are verified against the source document.

#### H.1 — Valid Queries (must answer correctly with citation)

| # | Query to send | Correct answer (verify against PDF) | Expected citation |
|---|---|---|---|
| H1 | *"What is the notice period required for an employee in an SB4 role who wants to resign?"* | 4 weeks written notice | Page 17 |
| H2 | *"How much is the annual Learning & Development budget per employee, and what can it be spent on?"* | $2,000/year; usable for conferences, certifications, online courses, books, coaching; amounts above $500 need manager approval | Page 13 |
| H3 | *"What are the four data classification levels used at TechNova and give one example for each?"* | L1 Public (marketing), L2 Internal (org charts), L3 Confidential (client data), L4 Restricted (PII/source code) | Page 14 |
| H4 | *"What severance does TechNova provide for termination without cause, and what is the maximum?"* | 2 weeks per year of service, capped at 26 weeks; contingent on signing a separation agreement | Page 17 |
| H5 | *"What mental health resources are available to employees?"* | Free EAP counseling (up to 8 sessions/year), 2 Mental Health Days/year, Headspace app, Mental Health First Aiders | Page 15 |

**Pass criteria for H.1:**
- Each response contains the exact figures/facts above (no rounding or invention).
- Every response includes at least one `[Page N — TechNova_Employee_Handbook.pdf]` citation.
- Verify the cited page in the PDF physically confirms the stated fact.

---

#### H.2 — Invalid / Out-of-Scope Queries (must refuse without hallucinating)

| # | Query to send | Why it is out of scope | Expected behaviour |
|---|---|---|---|
| H6 | *"What is TechNova's current stock price and market capitalisation?"* | Financial market data is not in the handbook | Polite refusal; agent must NOT invent a figure |
| H7 | *"Who is the current CEO of TechNova and what is their educational background?"* | Executive biography is not covered in the handbook | Polite refusal; agent must NOT fabricate a name or biography |
| H8 | *"Can you summarise the employment law requirements in California that apply to TechNova employees?"* | External legal commentary is not in the document; the handbook only references compliance in general terms | Polite refusal or acknowledgement that only general policy is covered; agent must NOT invent jurisdiction-specific law |

**Pass criteria for H.2:**
- Zero fabricated facts in any of the three responses.
- Agent should not say "I don't know" and then proceed to answer anyway.
- Refusal message should reference the uploaded documents (e.g., *"…not covered in the provided PDF(s)"*).

---

## Scoring Rubric

| Dimension | Weight | How to score |

|---|---|---|
| **Grounding accuracy** | 35% | % of factual claims that have a correct, verifiable citation |
| **Refusal quality** | 25% | All out-of-scope queries refused; no hallucinations |
| **Conversation continuity** | 20% | Context correctly maintained across multi-turn sequences |
| **UI/UX correctness** | 10% | All Category F checks pass without visual defects |
| **Language fidelity** | 10% | Correct language output; citations always preserved |

**Overall pass threshold: ≥ 80%**

---

## Reporting Issues

For each failed test, record:

```
Test ID   : e.g. B2
Query     : exact text sent
Expected  : what should have happened
Actual    : what happened (screenshot recommended)
Severity  : Critical / Major / Minor
```

Submit findings to the project maintainer with the PDF used for testing attached.
