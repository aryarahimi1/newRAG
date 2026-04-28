# Medication Reference

A small, end-to-end demo (**Medication Reference**) that answers natural-language questions about drug interactions (retrieval-augmented generation under the hood). Built as a technical-interview demo: **FastAPI** serves the existing `rag` pipeline unchanged; **SvelteKit** (static SPA) is the UI. Strong grounding + citations, and a PII redaction layer in front of the LLM.

> **Educational demo only — not medical advice.** Answers are grounded in publicly available FDA DailyMed labels, NIH MedlinePlus monographs, and OpenFDA Drug Enforcement (recall) records. Always verify with a licensed clinician.

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
   │                      (threading lock avoids duplicate concurrent ingests)
   ▼
┌─────────────────────┐
│ 4. Embed (local)    │  sentence-transformers/all-MiniLM-L6-v2
└─────────────────────┘
   │ 384-d vector
   ▼
┌─────────────────────┐
│ 5. Hybrid search    │  Chroma dense + BM25 fusion, top-k
└─────────────────────┘
   │ candidate chunks
   ▼
┌─────────────────────┐
│ 6. Cross-encoder    │  cross-encoder/ms-marco-MiniLM-L-6-v2, top-k
│    rerank           │
└─────────────────────┘
   │ best chunks + metadata
   ▼
┌─────────────────────┐
│ 7. Prompt + LLM     │  DeepSeek v3.2-exp via OpenRouter
│    call             │  forced citations, "I don't know" clause
└─────────────────────┘
   │
   ▼
answer + [n] citations → Svelte UI (FastAPI `/api/chat`)
```

**Ingest vs. chat:** Building or rebuilding the corpus is still `python -m scripts.ingest` (not exposed in the UI). When you **chat** about a drug missing from Chroma, **auto-ingest** runs inside the same pipeline before retrieval: RxNorm detects the name, `ingest_drugs` fetches DailyMed + MedlinePlus, embeds, and persists chunks — then retrieval runs. First question about a new drug pays ingest latency; later questions hit the DB immediately.

---

## Project layout

```
newRAG/
├── api/
│   └── main.py             # FastAPI: /api/chat, /api/corpus/stats, static UI (optional)
├── frontend/               # SvelteKit 5 + adapter-static (npm run dev / build)
├── rag/
│   ├── pipeline.py         # orchestrator: redact → detect → auto-ingest → retrieve → rerank → generate
│   ├── redact.py           # PII layer (Presidio + regex fallback)
│   ├── drug_detect.py      # RxNorm-backed drug mention detection
│   ├── query_rewrite.py    # PK intent expansion before hybrid retrieval
│   ├── retrieve.py         # Chroma + MedCPT encoders + BM25 hybrid
│   ├── rerank.py           # MS-MARCO cross-encoder
│   └── generate.py         # DeepSeek via OpenRouter (OpenAI SDK)
├── scripts/
│   └── ingest.py           # fetch + chunk + embed + persist
├── data/
│   ├── chroma_db/          # persistent vector store (built by ingest)
│   └── raw/                # optional: drop DrugBank XML/CSV here
├── requirements.txt
├── run.sh                  # backend: uvicorn api.main:app
├── .env.example
└── README.md
```

---

## Setup

Python 3.11+ recommended (3.12 works).

```bash
# 1. Create and activate a venv
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install Python deps
pip install --upgrade pip
pip install -r requirements.txt

# 3. Presidio needs a small spaCy model (optional; regex fallback exists)
python -m spacy download en_core_web_sm

# 4. Copy the env file and paste your OpenRouter key
cp .env.example .env
# edit .env, set OPENROUTER_API_KEY=sk-or-v1-...
```

Get a free OpenRouter key at <https://openrouter.ai/keys>. DeepSeek v3.2-exp is inexpensive (~$0.27 / M input, $0.41 / M output tokens as of this build).

### Frontend (Node 20+)

```bash
cd frontend
npm install
```

## Build the corpus

```bash
python -m scripts.ingest                        # ~45 drugs, full default run
python -m scripts.ingest --reset                # wipe + rebuild
python -m scripts.ingest --drugs ibuprofen,warfarin,lisinopril
python -m scripts.ingest --skip-medlineplus     # DailyMed only
```

The script fetches **FDA DailyMed** SPL labels, **NIH MedlinePlus** monographs, **OpenFDA Drug Enforcement** recalls (active and recent FDA drug recalls per drug, last ~2 years), and optionally **DrugBank Open Data** if you add files under `data/raw/`.

## Run the demo

**Terminal 1 — API (loads embedder, reranker, etc. once):**

```bash
./run.sh
# or: python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Open <http://localhost:8000/docs> for OpenAPI.

**Terminal 2 — Svelte dev server** (proxies `/api` → FastAPI):

```bash
cd frontend && npm run dev
```

Open <http://localhost:5173>. Set `CORS_ORIGINS` in `.env` if you use another origin.

**Single-process option:** After `cd frontend && npm run build`, copy or symlink `frontend/build` next to the app; FastAPI serves the static site from `/` when that directory exists, while `/api/*` stays the API.

---

## Design decisions

**Why FastAPI + Svelte instead of Streamlit?** Clear separation: the browser never imports Python or touches Chroma directly; CORS and JSON contracts keep the surface small. The **same** `RAGPipeline.run()` path as before preserves auto-ingest, the ingest lock, and hybrid retrieval.

**Why a two-stage retriever (bi-encoder top-k → cross-encoder top-k)?** Bi-encoders are cheap at scale; cross-encoders are accurate on small candidate sets. Standard pattern.

**Why Chroma?** Zero external infra for a demo; persists locally. Production would likely use pgvector or a managed vector DB.

**Why local embeddings?** No API cost at ingest; no data leaves the machine at indexing time.

**Why PII before retrieval?** Raw PII never reaches embedding queries or the LLM context builder.

**Why auto-ingest unknown drugs?** Static corpora break demos. RxNorm + on-demand `ingest_drugs` fills Chroma before retrieval so answers stay grounded.

**Why DeepSeek via OpenRouter?** One OpenAI-compatible client; swap models via env.

---

## Updates & improvements

### Retrieval quality

- **Clinical embedding model** — swapped `BAAI/bge-base-en-v1.5` for the asymmetric `ncbi/MedCPT-Query-Encoder` / `ncbi/MedCPT-Article-Encoder` pair, trained on 255M PubMed query-article pairs. Understands regulatory language (`CYP3A4`, `concomitant use`, `AUC`) natively. Requires re-ingest.
- **Adaptive search routing** — vector search runs first; if the top result confidence is high (score ≥ 0.75 with a clear gap to rank 4), BM25 is skipped entirely. Otherwise falls through to full hybrid RRF merge. Self-correcting — no routing rules to maintain.
- **Source-aware chunking** — DailyMed and DrugBank use 600-char chunks (dense regulatory content, one idea per chunk); MedlinePlus uses 1000-char chunks (flowing prose). Previously all sources used a flat 2000-char limit that crammed multiple unrelated items into one chunk.
- **Section prefix on chunks** — every stored chunk is prefixed with its FDA section name: `[Drug Interactions] ...`, `[Boxed Warning] ...`. The embedder and reranker now have an explicit section signal, not just raw text.
- **Smart BM25 tokenizer** — replaced naive `str.split()` with a tokenizer that handles hyphenated drug names (`co-amoxiclav` → `["co-amoxiclav", "co", "amoxiclav"]`), dosage strings (`500mg` → `["500mg", "500", "mg"]`), and camelCase headers.
- **BM25 index persistence** — index is saved to `data/chroma_db/bm25_index.pkl` after every ingest and loaded on startup. Eliminates the cold-start rebuild latency spike that previously blocked the first query after every restart.
- **DrugBank prose serialization** — CSV rows were previously serialized as pipe-delimited key-value strings (`"key: val | key: val"`), which embed poorly. Now converted to prose sentences before embedding.

### API

- **Conversation-aware chat** — `POST /api/chat` and `POST /api/chat/stream` accept a JSON **`history`** array: `{ role: "user"|"assistant", content: string }` in order, alternating user then assistant per completed exchange. Omit or send `[]` for the first question in a thread. Used for scoped retrieval as well as the LLM.
- **SSE streaming endpoint** — `/api/chat/stream` emits Server-Sent Events: `status`, `token`, `pre_answer`, `result`, `error`, and `done` frames. The UI renders tokens as they arrive instead of waiting for the full response.
- **Security hardening** — model identifier removed from `/api/config` (was leaking provider/model names to unauthenticated callers). Internal fields (`persist_dir`, `embedding_model`, `collection`) stripped from `/api/corpus/stats`. Raw exception details no longer surfaced to clients.

### UI

- **Multi-session chat** — collapsible sidebar with independent chat sessions, each maintaining its own message history, status log, and pipeline result.
- **Streaming token display** — answer streams in word-by-word via SSE; pre-answer message shown immediately when auto-ingest triggers for an unknown drug.
- **Design refresh** — animated background, DM Sans + JetBrains Mono fonts, custom scrollbar, spacing and border-radius design tokens.
- **Configurable API proxy** — `VITE_API_PROXY` env var controls the backend target (defaults to `http://127.0.0.1:8000`).

### Latest (since prior README)

- **Multi-turn follow-ups** — Each chat keeps its thread in the browser (localStorage). Every request sends **`history`** as completed user/assistant pairs. The backend now threads that conversation into RxNorm detection, hybrid retrieval, and reranking (prior user turns are concatenated redaction-safe, capped by size), not only into the LLM prompt—short follow-ups like “what about with aspirin?” still resolve drugs and sources from earlier turns. The generator includes up to 12 prior Q&A pairs before the grounded `CONTEXT` turn (was 2). Optional tuning: `RAG_THREAD_PRIOR_TURNS`, `RAG_THREAD_MAX_CHARS`, `RAG_LLM_HISTORY_TURNS` in `.env` (defaults in `.env.example`).
- **Product naming** — **Medication Reference** is used consistently: FastAPI title, `OPENROUTER_TITLE` in `.env`, and `User-Agent` strings on RxNorm and ingest HTTP clients (replacing the older “drug RAG” wording).
- **PK-aware query rewrite** — new `rag/query_rewrite.py` detects lay phrasing for pharmacokinetic questions (e.g. how long a drug stays in the body, time to peak / onset) and *appends* matching SPL-style terms (`half-life`, `clearance`, `Tmax`, etc.) to the search query so hybrid retrieval can hit **Clinical Pharmacology** / **Pharmacokinetics** sections without re-ingest. Invoked from `retrieve.py` before embedding the query.
- **Ingest: OpenFDA enforcement** — `scripts.ingest` pulls **Drug Enforcement** (recall) records from OpenFDA per drug by default (disable with `--skip-openfda`), so the corpus can include recall class and reason alongside labels. Additional DailyMed **section** keywords (e.g. `CLINICAL PHARMACOLOGY`, `PHARMACOKINETICS`, `MECHANISM OF ACTION`, `DESCRIPTION`) are prioritized for chunking.
- **LLM system prompt** — `generate.py` frames the assistant as a broader **medication information** tool (label-supported indications, PK/MoA when present in context), not only drug–drug interactions.
- **Frontend** — **light / dark** theme with persistence (`src/lib/theme.ts`, `data-theme` on `<html>`). **isomorphic-dompurify** sanitizes rendered answer HTML. Chat sessions use stable localStorage keys under the Medication Reference prefix; the UI can surface **recall** metadata when chunks carry OpenFDA enforcement source types. Substantial layout and styling updates in `+page.svelte` and `app.css` / `app.html`.

---

## Known limitations (worth discussing live)

- **No auth / rate limits** on the API — add before any real deployment.
- **No reranking by source type** (DailyMed vs MedlinePlus).
- **Negation in retrieval** remains a hard problem; cross-encoder helps but does not fix it.
- **No evaluation harness** — would add golden Q&A + regression checks.
- **DrugBank** requires manual download into `data/raw/`.

---

## Troubleshooting

**`OPENROUTER_API_KEY is not set`** — add to `.env`, or enable **Skip LLM** in the UI for retrieval-only.

**`Presidio failed to load`** — `python -m spacy download en_core_web_sm`, or rely on regex fallback.

**CORS errors** — set `CORS_ORIGINS` to your dev origin (e.g. `http://localhost:5173`).

**Empty vector store** — run `python -m scripts.ingest` first, or check `CHROMA_DIR`.
