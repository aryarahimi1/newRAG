"""End-to-end RAG pipeline orchestrator.

Intentionally a plain class rather than a framework chain — for a 2-day
demo, debuggability beats abstraction. Each stage returns structured data
that the UI can surface verbatim.

Stages
------
1. redact          PII → [ENTITY] placeholders
2. detect          Extract drug mentions via RxNorm
3. auto-ingest     (optional) Fetch + index any mentioned drug we don't
                   already have, so the LLM has something real to ground on
4. retrieve        Embed + search Chroma, top-20
5. rerank          Cross-encoder, top-8 (default)
6. generate        DeepSeek via OpenRouter, grounded + cited
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional

from rag.drug_detect import DrugDetector, DrugMention, get_detector, missing_drugs
from rag.generate import (
    DeepSeekGenerator,
    GenerationError,
    GenerationResult,
    get_generator,
)
from rag.redact import PIIRedactor, RedactionResult, get_redactor
from rag.rerank import Reranker, get_reranker
from rag.retrieve import RetrievedChunk, VectorStore, get_store

logger = logging.getLogger(__name__)

_INGEST_LOCK = threading.Lock()


@dataclass
class StageTiming:
    redact_ms: float = 0.0
    detect_ms: float = 0.0
    ingest_ms: float = 0.0
    retrieve_ms: float = 0.0
    rerank_ms: float = 0.0
    generate_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return (
            self.redact_ms
            + self.detect_ms
            + self.ingest_ms
            + self.retrieve_ms
            + self.rerank_ms
            + self.generate_ms
        )


@dataclass
class AutoIngestResult:
    missing: List[DrugMention] = field(default_factory=list)
    ingested: List[str] = field(default_factory=list)
    added_chunks: int = 0
    error: Optional[str] = None
    skipped: bool = False


@dataclass
class RAGResult:
    question: str
    redaction: RedactionResult
    detected_drugs: List[DrugMention]
    auto_ingest: AutoIngestResult
    retrieved: List[RetrievedChunk]
    reranked: List[RetrievedChunk]
    generation: Optional[GenerationResult]
    timing: StageTiming = field(default_factory=StageTiming)
    error: Optional[str] = None
    warnings: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "redaction": asdict(self.redaction),
            "detected_drugs": [asdict(d) for d in self.detected_drugs],
            "auto_ingest": asdict(self.auto_ingest),
            "retrieved": [asdict(c) for c in self.retrieved],
            "reranked": [asdict(c) for c in self.reranked],
            # Omit the `model` field: it carries the internal model identifier
            # (e.g. provider/model-name) which must not be disclosed to clients.
            "generation": (
                {k: v for k, v in asdict(self.generation).items() if k != "model"}
                if self.generation else None
            ),
            "timing": asdict(self.timing),
            "error": self.error,
            "warnings": self.warnings,
        }


class RAGPipeline:
    def __init__(
        self,
        store: Optional[VectorStore] = None,
        redactor: Optional[PIIRedactor] = None,
        reranker: Optional[Reranker] = None,
        detector: Optional[DrugDetector] = None,
        generator: Optional[DeepSeekGenerator] = None,
        top_k_retrieve: int = 20,
        top_k_rerank: int = 8,
        auto_ingest: bool = True,
    ):
        self.store = store or get_store()
        self.redactor = redactor or get_redactor()
        self.reranker = reranker or get_reranker()
        self.detector = detector or get_detector()
        self._generator = generator
        self.top_k_retrieve = top_k_retrieve
        self.top_k_rerank = top_k_rerank
        self.auto_ingest = auto_ingest

    @property
    def generator(self) -> DeepSeekGenerator:
        if self._generator is None:
            self._generator = get_generator()
        return self._generator

    def run(
        self,
        question: str,
        skip_generation: bool = False,
        auto_ingest: Optional[bool] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_token: Optional[Callable[[str], None]] = None,
        on_missing_drugs: Optional[Callable[[List[str]], None]] = None,
        history: Optional[List[dict]] = None,
        top_k_retrieve: Optional[int] = None,
        top_k_rerank: Optional[int] = None,
    ) -> RAGResult:
        timing = StageTiming()
        do_ingest = self.auto_ingest if auto_ingest is None else auto_ingest
        k_retrieve = top_k_retrieve if top_k_retrieve is not None else self.top_k_retrieve
        k_rerank = top_k_rerank if top_k_rerank is not None else self.top_k_rerank

        def _status(msg: str) -> None:
            logger.info(msg)
            if on_status is not None:
                try:
                    on_status(msg)
                except Exception:  # noqa: BLE001
                    pass

        # ----- 1. Redact ---------------------------------------------------
        _status("Redacting PII…")
        t0 = time.perf_counter()
        redaction = self.redactor.redact(question)
        timing.redact_ms = (time.perf_counter() - t0) * 1000

        # ----- 2. Detect drug mentions ------------------------------------
        _status("Detecting drug mentions via RxNorm…")
        t0 = time.perf_counter()
        try:
            detected = self.detector.detect(redaction.redacted)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Drug detection failed: %s", exc)
            detected = []
        timing.detect_ms = (time.perf_counter() - t0) * 1000

        # ----- 3. Auto-ingest missing drugs -------------------------------
        ai_result = AutoIngestResult()
        if do_ingest and detected:
            indexed = self.store.indexed_drugs()
            ai_result.missing = missing_drugs(detected, indexed)
            if ai_result.missing:
                if on_missing_drugs is not None:
                    try:
                        on_missing_drugs([m.canonical for m in ai_result.missing])
                    except Exception:  # noqa: BLE001
                        pass
                t0 = time.perf_counter()
                ai_result = self._auto_ingest(ai_result, on_status=_status)
                timing.ingest_ms = (time.perf_counter() - t0) * 1000
        elif not do_ingest:
            ai_result.skipped = True

        # ----- 4. Retrieve -------------------------------------------------
        _status("Retrieving top-k from Chroma…")
        t0 = time.perf_counter()
        scope = [m.canonical for m in detected] if detected else None
        retrieved = self.store.search(
            redaction.redacted,
            top_k=k_retrieve,
            restrict_to_drug_names=scope,
        )
        timing.retrieve_ms = (time.perf_counter() - t0) * 1000

        # ----- 5. Rerank ---------------------------------------------------
        _status("Reranking with cross-encoder…")
        t0 = time.perf_counter()
        reranked = self.reranker.rerank(
            redaction.redacted, retrieved, top_k=k_rerank
        )
        timing.rerank_ms = (time.perf_counter() - t0) * 1000

        # ----- 6. Generate -------------------------------------------------
        generation: Optional[GenerationResult] = None
        error: Optional[str] = None
        if not skip_generation:
            _status("Generating grounded answer with DeepSeek…")
            t0 = time.perf_counter()
            try:
                if on_token is not None:
                    generation = self.generator.generate_stream(
                        redaction.redacted, reranked, history=history, on_token=on_token
                    )
                else:
                    generation = self.generator.generate(redaction.redacted, reranked, history=history)
            except GenerationError as exc:
                # GenerationError already carries a safe user-facing message;
                # the original cause has been logged inside generate().
                logger.error("Generation failed (classified): %s", exc.safe_message)
                error = exc.safe_message
            except Exception as exc:  # noqa: BLE001
                logger.exception("Generation failed (unclassified)")
                error = "An unexpected error occurred. Please try again."
            timing.generate_ms = (time.perf_counter() - t0) * 1000

        return RAGResult(
            question=question,
            redaction=redaction,
            detected_drugs=detected,
            auto_ingest=ai_result,
            retrieved=retrieved,
            reranked=reranked,
            generation=generation,
            timing=timing,
            error=error,
        )

    def _auto_ingest(
        self,
        ai_result: AutoIngestResult,
        on_status: Callable[[str], None],
    ) -> AutoIngestResult:
        # Imported lazily so the package doesn't pull in requests/bs4 on paths
        # that don't need auto-ingest.
        from scripts.ingest import ingest_drugs

        missing_names = [m.canonical for m in ai_result.missing]
        pretty = ", ".join(missing_names)
        on_status(f"New drug(s) detected: {pretty}. Learning now…")

        with _INGEST_LOCK:
            # Re-check under the lock in case a sibling request just ingested
            # the same drug.
            indexed_now = self.store.indexed_drugs()
            still_missing_mentions = [
                m
                for m in ai_result.missing
                if m.canonical.lower() not in indexed_now
            ]
            still_missing = [m.canonical for m in still_missing_mentions]
            if not still_missing:
                on_status("Already covered — another request beat us to it.")
                ai_result.ingested = missing_names
                return ai_result
            try:
                result = ingest_drugs(
                    drugs=still_missing,
                    store=self.store,
                    on_progress=on_status,
                    drug_mentions=still_missing_mentions,
                )
                ai_result.ingested = result.get("drugs", still_missing)
                ai_result.added_chunks = result.get("added_chunks", 0)
                if ai_result.added_chunks == 0:
                    ai_result.error = (
                        "No documents returned by upstream sources. "
                        "The LLM will answer 'I don't know' based on context."
                    )
            except Exception as exc:  # noqa: BLE001
                logger.exception("Auto-ingest failed")
                ai_result.error = "Could not load drug information automatically. Results may be incomplete."

        return ai_result
