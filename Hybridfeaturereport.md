# Hybrid BM25 + Dense Retrieval — Feature Report

## What was added

`rag/retrieve.py` now runs **two retrievers in parallel** on every query and merges their results using **Reciprocal Rank Fusion (RRF)** before returning the top-k chunks to the reranker.

| Component | Role |
|---|---|
| `all-MiniLM-L6-v2` (existing) | Dense semantic search via Chroma vector index |
| `BM25Okapi` (new, `rank-bm25`) | Keyword search over the full corpus text |
| RRF merge (new) | Combines both ranked lists: `score = Σ 1 / (60 + rank)` |

## Key implementation details

- **`VectorStore._build_bm25()`** — fetches all documents from Chroma and builds the BM25 index in memory. Called lazily on first search.
- **`VectorStore._bm25_search()`** — tokenizes the query, scores every document, returns the top candidates.
- **`VectorStore._rrf_merge()`** — standard RRF with k=60. Chunks that appear in both lists get a combined rank boost; chunks unique to one list are still included.
- **`VectorStore.add()`** — invalidates the BM25 index whenever new documents are added (e.g. auto-ingest), so the next search rebuilds it automatically.
- No changes required to `pipeline.py` or `streamlit_app.py` — the upgrade is fully transparent to the rest of the stack.

## Why this helps

Drug names (`warfarin`, `metformin`, `clopidogrel`) are rare proper nouns. Dense embeddings compress them into semantic neighbourhood vectors, which can miss chunks that literally contain the word. BM25 finds exact matches perfectly. The hybrid approach gets both: semantic recall from the embedder and lexical precision from BM25.

## Dependency added

`rank-bm25==0.2.2` added to `requirements.txt`.
