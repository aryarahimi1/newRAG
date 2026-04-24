# Multi-Turn Conversation Memory — Feature Report

## What was added

The app now supports multi-turn conversations. Users can ask follow-up questions naturally without repeating context — the model remembers what was said earlier in the thread.

---

## How it works

**Before:** Every question was completely isolated. The LLM only saw the current question + retrieved chunks.

**After:** The last 2 completed turns (question + answer pairs) are injected into the message list before the current question, so the model has conversation context when generating the answer.

```
System prompt
  → [prior turn 1: user question]
  → [prior turn 1: assistant answer]
  → [prior turn 2: user question]
  → [prior turn 2: assistant answer]
  → Current question + retrieved CONTEXT chunks   ← new question answered here
```

---

## Files changed

| File | What changed |
|---|---|
| `rag/generate.py` | `generate()` accepts a `history` param — list of `{question, answer}` dicts — and injects them as real `user`/`assistant` messages before the current turn. Capped at last 2 turns to keep prompt size small. |
| `rag/pipeline.py` | `run()` accepts and forwards `history` to the generator. |
| `frontend` + `api/main.py` | Chat UI with multi-turn history; detail panels (PII, drugs, auto-ingest, citations, debug) below the latest response. Sidebar includes corpus stats, pipeline controls, sample questions, and clear conversation. |

---

## What the UI looks like now

- Chat bubbles for every turn (user + assistant)
- Detail panels (PII redaction, drug detection, citations, debug) shown below the latest response
- Sample questions moved to sidebar with a "Use sample" button
- **Clear conversation** button resets the whole thread

---

## Example of follow-up flow

```
User:      Can I take ibuprofen with lisinopril?
Assistant: [answer with citations]

User:      What about at a higher dose?
Assistant: [answer that understands "higher dose" refers to ibuprofen + lisinopril]

User:      And is aspirin safer in that case?
Assistant: [compares aspirin vs ibuprofen in context of lisinopril]
```

Previously all three questions would return independent answers with no connection between them.
