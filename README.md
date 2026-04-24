# Drug Interaction RAG

A small, end-to-end Retrieval-Augmented Generation demo that answers natural-language questions about drug interactions. Built as a 2-day technical-interview demo: single process, single Streamlit UI, no API layer, strong grounding + citations, and a PII redaction layer in front of the LLM.

> **Educational demo only — not medical advice.** Answers are grounded in publicly available FDA DailyMed and NIH MedlinePlus content. Always verify with a licensed clinician.

---

## Pipeline

```
user question
   │
   ▼
┌─────────────────────┐
│ 1. PII redaction    │  Presidio → regex fallback
└─────────────────────┘
   │ redacted text
   ▼
┌─────────────────────┐
│ 2. Drug detection   │  NIH RxNorm approximateTerm API
│    (RxNorm)         │  → {mention → canonical_name, rxcui, score}
└─────────────────────┘
   │ list of DrugMention
   ▼
┌─────────────────────┐
│ 3. Coverage + auto- │  Any detected drug not already in Chroma?
│    ingest (on miss) │  → call scripts.ingest.ingest_drugs([drug])
└─────────────────────┘  → DailyMed + MedlinePlus fetch + chunk + upsert
   │
   ▼
┌─────────────────────┐
│ 4. Embed (local)    │  sentence-transformers/all-MiniLM-L6-v2
└─────────────────────┘
   │ 384-d vector
   ▼
┌─────────────────────┐
│ 5. Vector search    │  Chroma (cosine, HNSW), top-20
└─────────────────────┘
   │ 20 candidate chunks
   ▼
┌─────────────────────┐
│ 6. Cross-encoder    │  cross-encoder/ms-marco-MiniLM-L-6-v2, top-5
│    rerank           │
└─────────────────────┘
   │ 5 best chunks + metadata
   ▼
┌─────────────────────┐
│ 7. Prompt + LLM     │  DeepSeek v3.2-exp via OpenRouter
│    call             │  forced citations, "I don't know" clause
└─────────────────────┘
   │
   ▼
answer + [n] citations → Streamlit UI (expandable source chunks)
```

---

## Project layout

```
drug-rag/
├── streamlit_app.py        # UI; calls the pipeline directly
├── rag/
│   ├── pipeline.py         # orchestrator: redact → detect → auto-ingest → retrieve → rerank → generate
│   ├── redact.py           # PII layer (Presidio + regex fallback)
│   ├── drug_detect.py      # RxNorm-backed drug mention detection
│   ├── retrieve.py         # Chroma + MiniLM embedder
│   ├── rerank.py           # MS-MARCO cross-encoder
│   └── generate.py         # DeepSeek via OpenRouter (OpenAI SDK)
├── scripts/
│   └── ingest.py           # fetch + chunk + embed + persist
├── data/
│   ├── chroma_db/          # persistent vector store (built by ingest)
│   └── raw/                # optional: drop DrugBank XML/CSV here
├── requirements.txt
├── .env.example
└── README.md
```

---

## Setup

Python 3.11 recommended.

```bash
# 1. Create and activate a venv
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install
pip install --upgrade pip
pip install -r requirements.txt

# 3. Presidio needs a small spaCy model. Skip this if you're fine with the
#    regex fallback (the app will auto-detect and fall back silently).
python -m spacy download en_core_web_sm

# 4. Copy the env file and paste your OpenRouter key
cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free OpenRouter key at <https://openrouter.ai/keys>. DeepSeek v3.2-exp is inexpensive (~$0.27 / M input, $0.41 / M output tokens as of this build).

## Build the corpus

```bash
python -m scripts.ingest                        # ~45 drugs, full default run
python -m scripts.ingest --reset                # wipe + rebuild
python -m scripts.ingest --drugs ibuprofen,warfarin,lisinopril
python -m scripts.ingest --skip-medlineplus     # DailyMed only
```

The script fetches:

- **FDA DailyMed** SPL XML labels (gold standard for "Drug Interactions", "Warnings", "Contraindications" sections). Picks the most recent labeling per drug via the `spls.json` API, then parses section titles.
- **NIH MedlinePlus** drug monographs (patient-friendly plain language). Resolves the drug → monograph URL via the MedlinePlus web-search endpoint, then scrapes `<section>` elements.
- **DrugBank Open Data** (optional). If you drop `drugbank_open_structures.csv` or the full `drugbank_all_full_database.xml` into `data/raw/`, it will be parsed too. DrugBank requires free registration; the link is on the Open Data page.

Expect ~400–1000 chunks after the default run (varies with what DailyMed has live).

## Run the demo

```bash
streamlit run streamlit_app.py
```

Open <http://localhost:8501>. Try:

- `Can I take ibuprofen with lisinopril for my blood pressure?`
- `Is it safe to combine warfarin and aspirin?`
- `I'm John Smith (john@example.com). Does metformin interact with alcohol?` — shows the PII layer
- `What happens if I take sertraline and tramadol together?` — classic serotonin-syndrome interaction

The UI shows, in order:

1. Original vs. redacted question and which PII entities were scrubbed
2. The LLM answer with inline `[n]` citations
3. Expandable citation cards for each of the top-5 reranked chunks, with source URL + similarity score
4. A debug pane with the full top-20 pre-rerank list and per-stage timings

---

## Design decisions

**Why a two-stage retriever (bi-encoder top-20 → cross-encoder top-5)?**
Bi-encoders are ~1000× cheaper per comparison than cross-encoders because the passage embeddings are precomputed once at ingest time. But they're less accurate — they measure *semantic similarity*, not *question-to-passage relevance*. Doing cheap retrieval to top-20 and then rerunning a cross-encoder on those 20 gives near-cross-encoder accuracy with near-bi-encoder latency. Standard industry pattern.

**Why Chroma (not FAISS / pgvector / Pinecone)?**
Zero infrastructure for a 2-day demo, persists to a local directory, good enough for ≤10k chunks. If this were going to production I'd move to pgvector (join with patient/med metadata) or a managed vector DB.

**Why local embeddings instead of the OpenAI/Voyage/Cohere embedding APIs?**
(1) Free, (2) no data leaves the machine at indexing time, (3) fast enough on CPU for this corpus size. `all-MiniLM-L6-v2` is 384-dim and trained on 1B+ sentence pairs — a well-known sensible default.

**Why the PII layer *before* anything else?**
Two reasons. First, it's the only stage that sees the raw user input, so it has to run first to keep raw PII out of downstream logs / vector queries / LLM prompts. Second, we deliberately run redaction *before* embedding so that even the vector DB query never sees PII. Drug names are not PII so retrieval quality is preserved.

**Why Presidio with a regex fallback?**
Presidio is Microsoft's well-maintained PII detector and handles `PERSON`, `LOCATION`, `DATE_TIME` etc. far better than regex. But it depends on spaCy models that need a separate `spacy download` step, which is a reliability risk for a live demo. So the module auto-falls-back to a conservative regex set (emails, phone numbers, SSNs, credit cards, name patterns like "I'm X") if Presidio fails to load. The sidebar shows which backend is live.

**Why DeepSeek v3.2-exp via OpenRouter, not OpenAI/Anthropic directly?**
(1) One API key, one SDK (OpenAI's), swap providers via model slug. (2) DeepSeek v3.2 is Pareto-optimal on cost vs. reasoning quality for grounded Q&A; it doesn't need to be brilliant, it needs to faithfully quote the retrieved context. (3) OpenRouter gives usage attribution via `HTTP-Referer` and `X-Title` headers which makes it trivial to monitor in the demo.

**Why auto-ingest unknown drugs?**
A static corpus is a brittle demo. When the user asks about a drug we haven't indexed, the naive RAG answer is "I don't know" — technically safe, but unhelpful. Instead we extract drug mentions from the (redacted) query with NIH's RxNorm API, check them against `VectorStore.indexed_drugs()`, and if any are missing we kick off an on-demand fetch of just those drugs from DailyMed + MedlinePlus before retrieval runs. The first question about rifampin costs ~5-10s of ingestion latency; every subsequent question about rifampin is instant because the chunks now live in Chroma. A `threading.Lock` prevents duplicate concurrent ingests. RxNorm is the right detector here because (a) it's free + no-auth, (b) it's the authoritative drug terminology maintained by NIH, (c) it handles brand names ("Advil" → "ibuprofen") and misspellings via fuzzy matching.

**Why the "I don't know" clause?**
For medical content, hallucination risk is the whole game. The system prompt explicitly instructs the model to answer *only* from the supplied `CONTEXT` passages, to cite `[n]`, and to return a literal "I don't know" if the context is insufficient. Low `temperature=0.1` pushes further against creative embellishment. During the demo you can intentionally ask about a drug that's not in the corpus to show this safety behavior working.

**Why Streamlit, no API layer?**
One process, one file, fastest demo. In production the pipeline would live behind a FastAPI service with per-user auth, request logging for audit, rate limiting, and the PII layer would integrate with whatever EHR the patient data is coming from. For an interview demo, the extra surface area is pure overhead.

---

## Known limitations (worth discussing live)

- **Single-turn only.** No conversation memory. A real product would maintain a thread with the same grounding constraints applied on every turn.
- **No reranking by source type.** DailyMed (FDA-approved labels) should arguably outrank MedlinePlus (patient-ed) on clinical questions. Would add a per-source prior or re-rank with a learned reward model.
- **No negation handling in retrieval.** "ibuprofen is *not* contraindicated with X" and "ibuprofen *is* contraindicated with X" embed to similar vectors. Mitigated by the cross-encoder but not eliminated.
- **No hybrid BM25 + dense search.** For drug names (rare proper nouns), BM25 would be very complementary to dense retrieval. Easy upgrade.
- **No evaluation harness.** Would add a golden set of ~50 interaction Q&A pairs with expected citations + an LLM-as-judge scorer to track regressions.
- **PII redaction is conservative.** Presidio's `PERSON` detector has false positives on capitalized drug names in unusual phrasing. For a real product I'd train a domain-specific NER or add a denylist of drug names to the recognizer.
- **DrugBank requires manual registration** to download even the Open Data portion, so it isn't auto-fetched. DailyMed + MedlinePlus alone give sufficient corpus for the demo.

---

## Troubleshooting

**`OPENROUTER_API_KEY is not set`** — copy `.env.example` to `.env` and paste your key, or toggle "Skip LLM call (retrieval only)" in the sidebar to demo the retrieval/rerank pipeline without the LLM.

**`Presidio failed to load`** — run `python -m spacy download en_core_web_sm`, or ignore and the regex fallback will handle emails/phones/SSNs/names.

**`No documents fetched`** — DailyMed's API occasionally rate-limits. Retry after a minute or pass `--skip-dailymed` and rely on MedlinePlus only.

**The vector store is empty** — you haven't run `python -m scripts.ingest` yet, or you're pointing `CHROMA_DIR` at a different path than the ingest script used.
