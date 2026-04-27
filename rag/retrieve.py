"""Chroma-backed vector store wrapper.

Embeddings come from `BAAI/bge-base-en-v1.5`. The model is loaded once and
reused; on the first call it downloads ~400 MB from Hugging Face to
`~/.cache/huggingface`. BGE uses asymmetric encoding: queries get an
instruction prefix ("Represent this sentence for searching relevant passages:")
while documents are embedded plain — this is handled automatically.

Collection metadata stores only lightweight fields; full chunk text lives in
Chroma's document store so we can show it verbatim in the UI.
"""

from __future__ import annotations

import rag._transformers_env  # noqa: F401 — before sentence_transformers

import logging
import os
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

from rag.query_rewrite import expand_pk_query

DEFAULT_EMBEDDING_MODEL = "ncbi/MedCPT-Article-Encoder"
DEFAULT_QUERY_ENCODER = "ncbi/MedCPT-Query-Encoder"
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

        logger.info("Loading encoders: doc=%s query=%s", embedding_model, DEFAULT_QUERY_ENCODER)
        self._doc_embedder = SentenceTransformer(embedding_model)
        self._query_embedder = SentenceTransformer(DEFAULT_QUERY_ENCODER)

        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        self._bm25_pickle = self.persist_dir / "bm25_index.pkl"

        # BM25 index — loaded from disk if available, otherwise built lazily.
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_ids: List[str] = []
        self._bm25_docs: List[str] = []
        self._bm25_metas: List[dict] = []

        if self._bm25_pickle.exists() and self._bm25_pickle.stat().st_size > 0:
            try:
                with open(self._bm25_pickle, "rb") as fh:
                    saved = pickle.load(fh)
                self._bm25 = saved["bm25"]
                self._bm25_ids = saved["ids"]
                self._bm25_docs = saved["docs"]
                self._bm25_metas = saved["metas"]
                logger.info("BM25 index loaded from disk (%d docs)", len(self._bm25_docs))
            except Exception as exc:
                logger.warning("BM25 pickle unreadable (%s); will rebuild on first search", exc)
                self._bm25 = None

    @property
    def collection(self):
        return self._collection

    def count(self) -> int:
        return self._collection.count()

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        # normalize_embeddings=True so cosine == dot product. Matches the
        # `hnsw:space=cosine` setting above.
        return self._doc_embedder.encode(
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
        self._rebuild_and_persist_bm25()

    # ------------------------------------------------------------------
    # BM25 helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        # camelCase split before lowercasing
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
        text = text.lower()
        raw_tokens = re.split(r"[^a-z0-9\-]+", text)
        seen: dict[str, None] = {}
        for tok in raw_tokens:
            tok = tok.strip("-")
            if not tok:
                continue
            if tok not in seen:
                seen[tok] = None
            if "-" in tok:
                for part in tok.split("-"):
                    if part and part not in seen:
                        seen[part] = None
            elif re.fullmatch(r"\d+[a-z]+", tok):
                num = re.match(r"(\d+)", tok).group(1)
                alpha = re.search(r"([a-z]+)$", tok).group(1)
                if num not in seen:
                    seen[num] = None
                if alpha not in seen:
                    seen[alpha] = None
        return list(seen.keys())

    def _build_bm25(self) -> None:
        """Fetch all documents from Chroma and build the BM25Okapi index."""
        all_data = self._collection.get(include=["documents", "metadatas"])
        self._bm25_ids = all_data.get("ids") or []
        self._bm25_docs = all_data.get("documents") or []
        self._bm25_metas = all_data.get("metadatas") or []
        tokenized = [self._tokenize(doc) for doc in self._bm25_docs]
        self._bm25 = BM25Okapi(tokenized)
        logger.info("BM25 index built over %d documents", len(self._bm25_docs))

    def _rebuild_and_persist_bm25(self) -> None:
        self._build_bm25()
        try:
            with open(self._bm25_pickle, "wb") as fh:
                pickle.dump(
                    {
                        "bm25": self._bm25,
                        "ids": self._bm25_ids,
                        "docs": self._bm25_docs,
                        "metas": self._bm25_metas,
                    },
                    fh,
                )
            logger.info("BM25 index persisted to %s", self._bm25_pickle)
        except Exception as exc:
            logger.warning("Failed to persist BM25 index: %s", exc)

    def _bm25_search(
        self,
        query: str,
        top_k: int,
        drug_filter: Optional[Set[str]] = None,
    ) -> List[RetrievedChunk]:
        """Return top_k chunks scored by BM25.

        If ``drug_filter`` is set, only chunks whose metadata ``drug_name`` is
        in that set (lowercased) are eligible — otherwise interaction-heavy
        labels from unrelated drugs dominate hybrid fusion.
        """
        if self._bm25 is None:
            self._build_bm25()
        scores = self._bm25.get_scores(self._tokenize(query))  # type: ignore[union-attr]
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        chunks: List[RetrievedChunk] = []
        for idx in top_indices:
            if len(chunks) >= top_k:
                break
            if scores[idx] <= 0:
                continue
            meta = self._bm25_metas[idx] or {}
            if drug_filter is not None:
                dn = str(meta.get("drug_name") or "").lower()
                if dn not in drug_filter:
                    continue
            chunks.append(
                RetrievedChunk(
                    id=self._bm25_ids[idx],
                    text=self._bm25_docs[idx],
                    metadata=meta,
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
        restrict_to_drug_names: Optional[Sequence[str]] = None,
    ) -> List[RetrievedChunk]:
        if self.count() == 0:
            return []

        rewrite = expand_pk_query(query)
        retrieval_query = rewrite.rewritten
        if rewrite.changed:
            logger.info(
                "PK query rewrite fired: intents=%s", list(rewrite.matched)
            )

        drug_filter: Optional[Set[str]] = None
        eff_where = where
        if restrict_to_drug_names:
            drug_filter = {
                str(d).strip().lower()
                for d in restrict_to_drug_names
                if d and str(d).strip()
            }
            if drug_filter:
                # Scoped retrieval: ignore caller ``where`` if it conflicts; drug
                # scoping is the primary filter for Q&A.
                eff_where = {"drug_name": {"$in": sorted(drug_filter)}}

        # Bias dense retrieval toward RxNorm canonical names when the user said
        # brands (Vyvanse, Ritalin) but chunks are tagged with ingredients.
        embed_query = retrieval_query
        if drug_filter:
            embed_query = (
                f"{retrieval_query}\nRelevant drugs (RxNorm): {', '.join(sorted(drug_filter))}"
            )

        # Dense retrieval — fetch a larger candidate pool before merging.
        dense_k = top_k * 2
        emb = self._query_embedder.encode(
            [embed_query],
            normalize_embeddings=True,
            show_progress_bar=False,
        ).tolist()[0]
        res = self._collection.query(
            query_embeddings=[emb],
            n_results=min(dense_k, self.count()),
            where=eff_where,
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

        _confident = False
        if dense_chunks:
            top_score = dense_chunks[0].score
            if top_score >= 0.75:
                if len(dense_chunks) < 4:
                    _confident = True
                elif (top_score - dense_chunks[3].score) >= 0.05:
                    _confident = True

        logger.debug(
            "search: %s path (top_score=%.3f)",
            "dense-only" if _confident else "hybrid",
            dense_chunks[0].score if dense_chunks else 0.0,
        )

        if _confident:
            merged = dense_chunks[:top_k]
        else:
            bm25_chunks = self._bm25_search(
                retrieval_query, top_k=dense_k, drug_filter=drug_filter
            )
            merged = self._rrf_merge(dense_chunks, bm25_chunks, top_k=top_k)

        # If the corpus has no rows for these drugs, fall back to global search.
        if drug_filter and not merged:
            logger.warning(
                "No chunks for drug_name in %s; falling back to unscoped retrieval",
                drug_filter,
            )
            return self.search(query, top_k=top_k, where=where, restrict_to_drug_names=None)

        # When 2+ drugs are scoped, guarantee every drug contributes its key
        # clinical sections — otherwise the LLM may only see one drug's
        # "Drug Interactions" and refuse to reason about the combination.
        if drug_filter and len(drug_filter) >= 2:
            merged = self._ensure_per_drug_coverage(
                merged, drug_filter=drug_filter, top_k=top_k
            )

        return merged

    # ------------------------------------------------------------------
    # Per-drug coverage helpers
    # ------------------------------------------------------------------

    _PRIORITY_SECTION_KEYWORDS = (
        "BOXED WARNING",
        # OpenFDA ingest stores exact title-case prefix + classification (not kw.title()).
        "FDA Recall - Class I",
        "DRUG INTERACTIONS",
        "CONTRAINDICATIONS",
        "WARNINGS AND PRECAUTIONS",
        "WARNINGS",
        "CLINICAL PHARMACOLOGY",
        "PHARMACOKINETICS",
    )

    @staticmethod
    def _priority_section_chroma_value(kw: str) -> str:
        """Map a priority keyword to the stored ``metadata.section`` value."""
        if kw.startswith("FDA Recall"):
            return kw
        return kw.title()

    def _fetch_priority_chunk_for_drug(
        self, drug_name: str
    ) -> Optional[RetrievedChunk]:
        """Return one high-priority clinical chunk for ``drug_name`` if available."""
        for kw in self._PRIORITY_SECTION_KEYWORDS:
            section_eq = self._priority_section_chroma_value(kw)
            try:
                got = self._collection.get(
                    where={
                        "$and": [
                            {"drug_name": drug_name},
                            {"section": {"$eq": section_eq}},
                        ]
                    },
                    limit=1,
                    include=["documents", "metadatas"],
                )
            except Exception:  # noqa: BLE001
                got = {}
            ids = got.get("ids") or []
            if not ids:
                continue
            docs = got.get("documents") or [""]
            metas = got.get("metadatas") or [{}]
            return RetrievedChunk(
                id=ids[0], text=docs[0], metadata=metas[0] or {}, score=0.0
            )
        # Fallback: any chunk for the drug.
        try:
            got = self._collection.get(
                where={"drug_name": drug_name},
                limit=1,
                include=["documents", "metadatas"],
            )
        except Exception:  # noqa: BLE001
            return None
        ids = got.get("ids") or []
        if not ids:
            return None
        docs = got.get("documents") or [""]
        metas = got.get("metadatas") or [{}]
        return RetrievedChunk(
            id=ids[0], text=docs[0], metadata=metas[0] or {}, score=0.0
        )

    def _ensure_per_drug_coverage(
        self,
        merged: List[RetrievedChunk],
        *,
        drug_filter: Set[str],
        top_k: int,
    ) -> List[RetrievedChunk]:
        """Inject a priority chunk for any drug under-represented in ``merged``."""
        present: dict[str, int] = {}
        for c in merged:
            dn = str(c.metadata.get("drug_name") or "").lower()
            if dn:
                present[dn] = present.get(dn, 0) + 1

        missing = [d for d in drug_filter if present.get(d, 0) == 0]
        if not missing:
            return merged

        existing_ids = {c.id for c in merged}
        injected: List[RetrievedChunk] = []
        for d in missing:
            extra = self._fetch_priority_chunk_for_drug(d)
            if extra is None or extra.id in existing_ids:
                continue
            existing_ids.add(extra.id)
            injected.append(extra)

        if not injected:
            return merged

        # Keep result size at top_k by trimming from the over-represented drugs.
        result = list(merged)
        if len(result) + len(injected) > top_k:
            # Drop from the end (lowest fused score) until there's room.
            overflow = len(result) + len(injected) - top_k
            result = result[: max(0, len(result) - overflow)]
        return result + injected

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
        self._bm25 = None
        self._bm25_ids = []
        self._bm25_docs = []
        self._bm25_metas = []
        if self._bm25_pickle.exists():
            self._bm25_pickle.unlink(missing_ok=True)


_default_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _default_store
    if _default_store is None:
        persist_dir = os.environ.get("CHROMA_DIR", str(DEFAULT_PERSIST_DIR))
        _default_store = VectorStore(persist_dir=persist_dir)
    return _default_store
