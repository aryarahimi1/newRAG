"""Chroma-backed vector store wrapper.

Embeddings come from `sentence-transformers/all-MiniLM-L6-v2`. The model is
loaded once and reused; on the first call it downloads ~90 MB from Hugging
Face to `~/.cache/huggingface`.

Collection metadata stores only lightweight fields; full chunk text lives in
Chroma's document store so we can show it verbatim in the UI.
"""

from __future__ import annotations

import rag._transformers_env  # noqa: F401 — before sentence_transformers

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION = "drug_docs"
DEFAULT_PERSIST_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma_db"


@dataclass
class RetrievedChunk:
    """Shape of a single retrieved chunk."""

    id: str
    text: str
    metadata: dict
    score: float  # similarity (1 - cosine distance)


class VectorStore:
    """Thin wrapper over chromadb PersistentClient + local embedder."""

    def __init__(
        self,
        persist_dir: Path | str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ):
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        logger.info("Loading embedder: %s", embedding_model)
        self._embedder = SentenceTransformer(embedding_model)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        # BM25 index — built lazily on first hybrid search, invalidated on add().
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[dict] = []

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # normalize_embeddings=True so cosine == dot product. Matches the
        # `hnsw:space=cosine` setting above.
        return self._embedder.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()

    def add(
        self,
        ids: Sequence[str],
        documents: Sequence[str],
        metadatas: Sequence[dict],
        batch_size: int = 64,
    ) -> None:
        if not ids:
            return
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            batch_ids = list(ids[start:end])
            batch_docs = list(documents[start:end])
            batch_meta = list(metadatas[start:end])
            batch_emb = self.embed(batch_docs)
            self._collection.add(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_meta,
                embeddings=batch_emb,
            )
        # Invalidate the BM25 index so it rebuilds with the new documents.
        self._bm25 = None

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------

    def _build_bm25(self) -> None:
        """Fetch all documents from Chroma and build the BM25Okapi index."""
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._bm25_ids = all_data.get("ids") or []
        self._bm25_docs = all_data.get("documents") or []
        self._bm25_metas = all_data.get("metadatas") or []
        tokenized = [doc.lower().split() for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d documents", len(self._bm25_docs))

    def _bm25_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Return top_k chunks scored by BM25."""
        if self._bm25 is None:
            self._build_bm25()
        scores = self._bm25.get_scores(query.lower().split())  # type: ignore[union-attr]
        # Pair each score with its index, sort descending, take top_k.
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        chunks: List[RetrievedChunk] = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunks.append(
                RetrievedChunk(
                    id=self._bm25_ids[idx],
                    text=self._bm25_docs[idx],
                    metadata=self._bm25_metas[idx] or {},
                    score=float(scores[idx]),
                )
            )
        return chunks

    @staticmethod
    def _rrf_merge(
        dense: List[RetrievedChunk],
        bm25: List[RetrievedChunk],
        top_k: int,
        k: int = 60,
    ) -> List[RetrievedChunk]:
        """Reciprocal Rank Fusion: score = Σ 1/(k + rank). k=60 is standard."""
        rrf: dict[str, float] = {}
        chunk_map: dict[str, RetrievedChunk] = {}

        for rank, chunk in enumerate(dense, start=1):
            rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (k + rank)
            chunk_map[chunk.id] = chunk

        for rank, chunk in enumerate(bm25, start=1):
            rrf[chunk.id] = rrf.get(chunk.id, 0.0) + 1.0 / (k + rank)
            if chunk.id not in chunk_map:
                chunk_map[chunk.id] = chunk

        ranked = sorted(rrf.items(), key=lambda x: x[1], reverse=True)[:top_k]
        from dataclasses import replace
        return [replace(chunk_map[cid], score=score) for cid, score in ranked]

    # ------------------------------------------------------------------
    # Public search — hybrid by default
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        top_k: int = 20,
        where: Optional[dict] = None,
    ) -> List[RetrievedChunk]:
        if self.count() == 0:
            return []

        # Dense retrieval — fetch a larger candidate pool before merging.
        dense_k = top_k * 2
        emb = self.embed([query])[0]
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=min(dense_k, self.count()),
            where=where,
        )
        dense_chunks: List[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(
            res.get("ids", [[]])[0],
            res.get("documents", [[]])[0],
            res.get("metadatas", [[]])[0],
            res.get("distances", [[]])[0],
        ):
            dense_chunks.append(
                RetrievedChunk(id=cid, text=doc, metadata=meta or {}, score=1.0 - float(dist))
            )

        # BM25 retrieval (skip per-field `where` filter — BM25 is corpus-wide).
        bm25_chunks = self._bm25_search(query, top_k=dense_k)

        # Merge and return top_k.
        return self._rrf_merge(dense_chunks, bm25_chunks, top_k=top_k)

    def indexed_drugs(self) -> set:
        """Return the set of drug names currently represented in the store.

        Cheap-ish: fetches metadatas only (no embeddings, no documents).
        """
        if self.count() == 0:
            return set()
        peek = self._collection.get(include=["metadatas"])
        drugs = set()
        for m in peek.get("metadatas", []) or []:
            if m and m.get("drug_name"):
                drugs.add(str(m["drug_name"]).lower())
        return drugs

    def has_drug(self, drug_name: str) -> bool:
        """True if at least one chunk's metadata.drug_name matches."""
        if self.count() == 0:
            return False
        res = self._collection.get(
            where={"drug_name": drug_name.lower()}, limit=1, include=[]
        )
        return bool(res.get("ids"))

    def stats(self) -> dict:
        n_chunks = self.count()
        sources = set()
        drugs = set()
        if n_chunks > 0:
            peek = self._collection.get(include=["metadatas"])
            for m in peek.get("metadatas", []) or []:
                if not m:
                    continue
                if "source_url" in m:
                    sources.add(m["source_url"])
                if "drug_name" in m and m["drug_name"]:
                    drugs.add(str(m["drug_name"]).lower())
        return {
            "n_chunks": n_chunks,
            "n_sources": len(sources),
            "n_drugs": len(drugs),
            "drugs": sorted(drugs),
            "embedding_model": self.embedding_model_name,
            "collection": self.collection_name,
            "persist_dir": str(self.persist_dir),
        }

    def reset(self) -> None:
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )


_default_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        persist_dir = os.environ.get("CHROMA_DIR", str(DEFAULT_PERSIST_DIR))
        _default_store = VectorStore(persist_dir=persist_dir)
    return _default_store
