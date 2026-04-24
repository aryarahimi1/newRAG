# Drug Interaction RAG

A small, end-to-end Retrieval-Augmented Generation demo that answers natural-language questions about drug interactions. Built as a technical-interview demo: **FastAPI** serves the existing `rag` pipeline unchanged; **SvelteKit** (static SPA) is the UI. Strong grounding + citations, and a PII redaction layer in front of the LLM.

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
drug-rag/
├── api/
│   └── main.py             # FastAPI: /api/chat, /api/corpus/stats, static UI (optional)
├── frontend/               # SvelteKit 5 + adapter-static (npm run dev / build)
├── rag/
│   ├── pipeline.py         # orchestrator: redact → detect → auto-ingest → retrieve → rerank → generate
│   ├── redact.py           # PII layer (Presidio + regex fallback)
│   ├── drug_detect.py      # RxNorm-backed drug mention detection
│   ├── retrieve.py         # Chroma + MiniLM embedder + BM25 hybrid
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

The script fetches **FDA DailyMed** SPL labels, **NIH MedlinePlus** monographs, and optionally **DrugBank Open Data** if you add files under `data/raw/`.

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
