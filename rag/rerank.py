"""Cross-encoder reranker.

`cross-encoder/ms-marco-MiniLM-L-6-v2` takes a (query, passage) pair and
returns a relevance score. Slower than bi-encoder retrieval but much more
accurate, so we retrieve top-20 cheaply with the embedder, then rerank to
top-5 for the LLM.
"""

from __future__ import annotations

import rag._transformers_env  # noqa: F401 — before sentence_transformers

import logging
from dataclasses import replace
from typing import List, Optional, Sequence

from sentence_transformers import CrossEncoder

from rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class Reranker:
    def __init__(self, model_name: str = DEFAULT_RERANKER_MODEL):
        logger.info("Loading reranker: %s", model_name)
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        chunks: Sequence[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []
        pairs = [(query, c.text) for c in chunks]
        scores = self._model.predict(pairs, show_progress_bar=False).tolist()
        # Overwrite `score` with cross-encoder relevance so the UI shows the
        # final ordering signal.
        scored = [replace(c, score=float(s)) for c, s in zip(chunks, scores)]
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored[:top_k]


_default_reranker: Optional[Reranker] = None


def get_reranker() -> Reranker:
    global _default_reranker
    if _default_reranker is None:
        _default_reranker = Reranker()
    return _default_reranker
