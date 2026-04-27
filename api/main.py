"""FastAPI service exposing the existing RAG pipeline (including RxNorm + auto-ingest).

The pipeline logic lives in ``rag.pipeline`` unchanged: concurrent requests share
one ``RAGPipeline`` instance; ``_INGEST_LOCK`` in that module still prevents duplicate
on-the-fly ingests for the same drug.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import queue as _queue
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, List, Optional

import rag._transformers_env  # noqa: F401 — before transformers
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag.pipeline import RAGPipeline
from rag.retrieve import get_store

logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_pipeline: Optional[RAGPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        with _pipeline_lock:
            if _pipeline is None:
                logger.info("Initializing RAGPipeline (heavy models load once)…")
                _pipeline = RAGPipeline()
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.basicConfig(level=logging.INFO)
    # Warm critical resources at startup so first chat isn't alone paying cold start.
    try:
        get_pipeline()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Pipeline warm-up deferred: %s", exc)
    yield


app = FastAPI(
    title="Medication Reference API",
    description="PII redaction → RxNorm → auto-ingest → hybrid retrieve → rerank → generate",
    version="0.2.0",
    lifespan=lifespan,
)

_cors_origins = os.environ.get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins.split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=8000)
    history: List[ChatMessage] = Field(default_factory=list)
    top_k_retrieve: int = Field(20, ge=5, le=50)
    top_k_rerank: int = Field(5, ge=1, le=10)
    auto_ingest: bool = True
    skip_generation: bool = Field(
        default_factory=lambda: not bool(os.environ.get("OPENROUTER_API_KEY"))
    )


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    """Non-secret UI hints (PII backend availability)."""
    try:
        p = get_pipeline()
        redactor_backend = p.redactor.backend
    except Exception:  # noqa: BLE001
        redactor_backend = "unknown"
    return {
        # Intentionally omit the raw model identifier — it leaks provider and
        # model names to unauthenticated callers and was reproduced verbatim in
        # upstream error payloads surfaced to users.
        "pii_backend": redactor_backend,
        "has_openrouter_key": bool(os.environ.get("OPENROUTER_API_KEY")),
    }


@app.get("/api/corpus/stats")
def corpus_stats() -> dict[str, Any]:
    store = get_store()
    raw = store.stats()
    # Strip fields that expose internal infrastructure details: filesystem paths,
    # embedding model names, and collection identifiers.
    return {
        "n_chunks": raw.get("n_chunks"),
        "n_sources": raw.get("n_sources"),
        "n_drugs": raw.get("n_drugs"),
        "drugs": raw.get("drugs", []),
    }


@app.post("/api/chat")
def chat(body: ChatRequest) -> dict[str, Any]:
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    # Alternating user/assistant pairs in message order (matches prior UI behavior).
    history_pairs: List[dict[str, str]] = []
    i = 0
    msgs = body.history
    while i + 1 < len(msgs):
        if msgs[i].role == "user" and msgs[i + 1].role == "assistant":
            history_pairs.append(
                {"question": msgs[i].content, "answer": msgs[i + 1].content}
            )
            i += 2
        else:
            i += 1

    pipeline = get_pipeline()
    status_messages: List[str] = []

    def on_status(msg: str) -> None:
        status_messages.append(msg)

    try:
        result = pipeline.run(
            q,
            skip_generation=body.skip_generation,
            auto_ingest=body.auto_ingest,
            on_status=on_status,
            history=history_pairs,
            top_k_retrieve=body.top_k_retrieve,
            top_k_rerank=body.top_k_rerank,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Pipeline run failed")
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred. Please try again.",
        ) from exc

    out = result.to_dict()
    out["status_log"] = status_messages
    return out


@app.post("/api/chat/stream")
async def chat_stream(body: ChatRequest) -> StreamingResponse:
    """SSE endpoint — emits status, token, result, and done frames."""
    q = body.question.strip()
    if not q:
        raise HTTPException(status_code=400, detail="Empty question")

    history_pairs: List[dict[str, str]] = []
    i = 0
    msgs = body.history
    while i + 1 < len(msgs):
        if msgs[i].role == "user" and msgs[i + 1].role == "assistant":
            history_pairs.append(
                {"question": msgs[i].content, "answer": msgs[i + 1].content}
            )
            i += 2
        else:
            i += 1

    pipeline = get_pipeline()
    event_queue: _queue.Queue = _queue.Queue()

    def run_pipeline() -> None:
        def on_status(msg: str) -> None:
            event_queue.put({"event": "status", "data": msg})

        def on_token(token: str) -> None:
            event_queue.put({"event": "token", "data": token})

        def on_missing_drugs(drug_names: List[str]) -> None:
            # Only emit a conversational pre-answer when the LLM will actually run.
            if body.skip_generation:
                return
            pretty = ", ".join(f"**{n}**" for n in drug_names)
            text = (
                f"I don't have {pretty} in my knowledge base yet. "
                f"Give me a moment — I'm reading the FDA label and patient "
                f"education pages right now. I'll answer your question as soon "
                f"as I'm done learning."
            )
            event_queue.put({"event": "pre_answer", "data": text})

        try:
            result = pipeline.run(
                q,
                skip_generation=body.skip_generation,
                auto_ingest=body.auto_ingest,
                on_status=on_status,
                on_token=on_token if not body.skip_generation else None,
                on_missing_drugs=on_missing_drugs,
                history=history_pairs,
                top_k_retrieve=body.top_k_retrieve,
                top_k_rerank=body.top_k_rerank,
            )
            event_queue.put({"event": "result", "data": result.to_dict()})
        except Exception as exc:  # noqa: BLE001
            logger.exception("Pipeline stream failed")
            event_queue.put({"event": "error", "data": "An internal error occurred. Please try again."})
        finally:
            event_queue.put(None)  # sentinel — stream is done

    threading.Thread(target=run_pipeline, daemon=True).start()

    async def generate():
        try:
            while True:
                try:
                    item = event_queue.get_nowait()
                except _queue.Empty:
                    await asyncio.sleep(0.01)
                    continue

                if item is None:
                    break

                event_type = item["event"]
                data = json.dumps(item["data"], ensure_ascii=False)
                yield f"event: {event_type}\ndata: {data}\n\n"
        finally:
            yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# Optional: serve production Svelte build from ../frontend/build
_frontend_build = Path(__file__).resolve().parent.parent / "frontend" / "build"
if _frontend_build.is_dir():
    app.mount("/", StaticFiles(directory=str(_frontend_build), html=True), name="static")
