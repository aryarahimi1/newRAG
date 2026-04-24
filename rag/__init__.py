"""RAG package for drug-interaction Q&A.

Avoid eager imports here: `from rag.retrieve import get_store` should not load
the LLM pipeline (or Chroma) until needed.
"""

from typing import TYPE_CHECKING

__all__ = ["RAGPipeline", "RAGResult"]

if TYPE_CHECKING:
    from rag.pipeline import RAGPipeline, RAGResult


def __getattr__(name: str):
    if name == "RAGPipeline":
        from rag.pipeline import RAGPipeline

        return RAGPipeline
    if name == "RAGResult":
        from rag.pipeline import RAGResult

        return RAGResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
