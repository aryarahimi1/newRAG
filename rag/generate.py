"""LLM generation via DeepSeek on OpenRouter.

We use the OpenAI Python SDK pointed at OpenRouter's OpenAI-compatible
endpoint. OpenRouter accepts optional HTTP-Referer / X-Title headers for
usage attribution; they are set from env vars when present.

The prompt forces strict grounding: the model must cite the passages it used
and must answer "I don't know" if the retrieved context doesn't cover the
question. This is the most important safety property for a medical demo.
"""

from __future__ import annotations

import logging
import os
import textwrap
from dataclasses import dataclass
from typing import List, Optional, Sequence

import httpx
from openai import OpenAI

from rag.retrieve import RetrievedChunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek/deepseek-v3.2-exp"
# OpenRouter OpenAI-compatible root (no trailing slash).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def _resolve_base_url(explicit: str) -> str:
    """Prefer OPENROUTER_BASE_URL env, then caller default."""
    u = (os.environ.get("OPENROUTER_BASE_URL") or explicit or OPENROUTER_BASE_URL).strip()
    return u.rstrip("/")

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are a cautious clinical-information assistant that helps people
    understand drug interactions using a curated corpus of drug-label and
    patient-education documents.

    Rules (follow all of them):
    1. Answer ONLY from the numbered CONTEXT passages supplied by the user
       message. Do NOT use outside knowledge.
    2. If the CONTEXT does not contain enough information to answer the
       question, respond with exactly: "I don't know based on the provided
       sources." and suggest the user consult a pharmacist or physician.
    3. Cite the passages you used inline using bracketed numbers like [1],
       [2] that match the passage numbers in the CONTEXT.
    4. Never invent drug names, dosages, interactions, or citation numbers.
    5. Include a one-sentence safety reminder that this is not medical
       advice and the user should confirm with a licensed clinician.
    6. Be concise. 3-6 sentences is usually enough.
    """
)


@dataclass
class GenerationResult:
    answer: str
    model: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def _format_context(chunks: Sequence[RetrievedChunk]) -> str:
    blocks = []
    for i, c in enumerate(chunks, start=1):
        drug = c.metadata.get("drug_name") or "unknown"
        section = c.metadata.get("section") or "general"
        source = c.metadata.get("source_url") or c.metadata.get("source") or "unknown"
        blocks.append(
            f"[{i}] drug: {drug} | section: {section} | source: {source}\n"
            f"{c.text.strip()}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, chunks: Sequence[RetrievedChunk]) -> str:
    if not chunks:
        return (
            f"QUESTION: {question}\n\n"
            "CONTEXT: (none — no passages were retrieved)\n\n"
            "Remember the rules in the system prompt."
        )
    return (
        f"QUESTION: {question}\n\n"
        f"CONTEXT:\n{_format_context(chunks)}\n\n"
        "Answer the question using only the CONTEXT above. "
        "Cite passages with [n]. If the CONTEXT is insufficient, say "
        "\"I don't know based on the provided sources.\""
    )


class DeepSeekGenerator:
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: str = OPENROUTER_BASE_URL,
        referer: Optional[str] = None,
        title: Optional[str] = None,
        timeout: float = 60.0,
    ):
        api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to your .env file."
            )

        self.model = model or os.environ.get("OPENROUTER_MODEL", DEFAULT_MODEL)
        resolved_base = _resolve_base_url(base_url)

        default_headers = {}
        ref = referer or os.environ.get("OPENROUTER_REFERER")
        ttl = title or os.environ.get("OPENROUTER_TITLE")
        if ref:
            default_headers["HTTP-Referer"] = ref
        if ttl:
            default_headers["X-Title"] = ttl

        # Custom httpx client avoids the SDK's default wrapper, which passes
        # `proxies=` into httpx and breaks on httpx 0.28+.
        # Keep `base_url` only on `OpenAI` — the SDK merges it into request URLs.
        _timeout = httpx.Timeout(timeout, connect=20.0)
        _http = httpx.Client(timeout=_timeout, follow_redirects=True)
        self._client = OpenAI(
            api_key=api_key,
            base_url=resolved_base,
            default_headers=default_headers or None,
            http_client=_http,
        )

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        temperature: float = 0.1,
        max_tokens: int = 600,
    ) -> GenerationResult:
        user_prompt = build_user_prompt(question, chunks)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            answer=answer.strip(),
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


_default_generator: Optional[DeepSeekGenerator] = None


def get_generator() -> DeepSeekGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = DeepSeekGenerator()
    return _default_generator
