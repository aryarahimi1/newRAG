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
from typing import Callable, List, Optional, Sequence

import httpx
from openai import OpenAI

from rag.retrieve import RetrievedChunk

# ---------------------------------------------------------------------------
# Safe user-facing error classes — these carry a sanitised message that is
# safe to surface to end users. The original exception is stored on .original
# for server-side logging only; it must never be serialised into a response.
# ---------------------------------------------------------------------------


class GenerationError(Exception):
    """Base class for generation errors that carry a safe user-facing message."""
    def __init__(self, safe_message: str, *, original: Optional[Exception] = None):
        super().__init__(safe_message)
        self.safe_message = safe_message
        self.original = original


class RateLimitGenerationError(GenerationError):
    pass


class AuthGenerationError(GenerationError):
    pass


class TimeoutGenerationError(GenerationError):
    pass


class ServiceGenerationError(GenerationError):
    pass

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
    You are a cautious clinical drug-information assistant that helps people
    understand medication-related questions using a curated corpus of
    drug-label, patient-education, and drug-information documents.

    Rules (follow all of them):
    1. Answer ONLY from the numbered CONTEXT passages supplied by the user
       message. Do NOT introduce facts, drug names, dosages, mechanisms, or
       warnings that are not stated in CONTEXT, and do not rely on outside
       medical knowledge.
    2. Synthesis across passages IS allowed when every step is grounded in
       CONTEXT. In particular:
       • If one passage warns about a drug class (e.g. MAOIs, sympathomimetics,
         serotonergic drugs, CNS stimulants, CYP2D6 inhibitors, anticoagulants)
         and another CONTEXT passage describes a specific drug that the same
         CONTEXT identifies as belonging to that class (or as having that
         mechanism), you may state the class-level relationship and cite BOTH
         passages.
       • You may combine warnings, contraindications, and adverse-reaction
         passages from different drugs in CONTEXT when the question is about
         using them together, as long as every claim is cited to a passage
         that supports it.
       • Do NOT assert a drug belongs to a class unless CONTEXT says so. If
         the class membership of one of the drugs is not stated in CONTEXT,
         do not draw the class-level conclusion — instead state plainly that
         the supplied passages don't establish the link.
    3. If — even after cross-passage synthesis — the CONTEXT does not contain
       enough information to answer the question, respond with exactly:
       "I don't know based on the provided sources." and suggest the user
       consult a pharmacist or physician.
    4. Cite every factual claim inline using bracketed numbers like [1], [2]
       that match passage numbers in CONTEXT. Never cite a number that does not
       appear in CONTEXT, and never invent drug names, dosages, interactions,
       or citation numbers.
    5. Structure your answer with sections the CONTEXT actually supports. Use
       clear headings or short lead sentences for each section you include.
       Omit any section entirely if CONTEXT does not support it — do not guess
       or fill gaps. Typical sections (only when grounded):
       • Direct answer — what the sources say about the question.
       • Mechanism / why — only if CONTEXT explains how or why.
       • Who may be at higher risk — only if CONTEXT mentions specific groups
         or situations (for example older adults, kidney impairment,
         concurrent diuretics or interacting drugs, dehydration, pregnancy, or
         similar); list only what the passages state.
       • Safer alternatives or mitigations — only if CONTEXT names them.
       • Red flags / when to seek a clinician urgently — only if CONTEXT
         describes warning signs, monitoring needs, or situations requiring
         professional care.
    6. Use short paragraphs or compact bullet lists when that improves
       clarity. Keep length proportional to how much CONTEXT supports: about
       150–400 words when passages cover the topic well; shorter when CONTEXT
       is thin. Do not pad or speculate to reach length.
    7. End with exactly one sentence reminding the user that this is
       educational information, not medical advice, and that they should
       confirm with a licensed clinician.
    8. If the user's message is a greeting, small talk, a social nicety, or
       anything clearly unrelated to a specific drug or medical question —
       such as "yo", "hello", "how is it going", "thanks", "what can you do",
       "what is this" — respond warmly and briefly WITHOUT saying
       "I don't know based on the provided sources." Instead, introduce
       yourself and describe what this platform does:

       This is a clinical drug-information assistant powered by a retrieval
       system grounded in FDA-approved drug labels (DailyMed SPL), NIH
       MedlinePlus patient education pages, and DrugBank data. It is designed
       to help patients, caregivers, and clinicians quickly understand:
         • Drug uses and label-supported indications
         • Drug–drug interactions and other clinically relevant precautions
         • Warnings, contraindications, and boxed warnings
         • Adverse reactions and who is at higher risk
         • Dosing guidance and administration notes
         • Pharmacokinetics, onset, duration, and mechanism of action when
           sources state them
         • Safe alternatives or mitigations when sources name them
         • When to seek urgent clinical attention

       Keep the intro to 2–4 sentences and end by inviting the user's real
       question. "I don't know based on the provided sources" is reserved
       strictly for genuine medical questions where the retrieved context
       cannot support an answer.
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
        "Answer using ONLY the CONTEXT above. You may combine information "
        "across passages (e.g. a class-level warning in one passage with a "
        "specific drug described in another) as long as every claim is "
        "supported by a passage you cite with [n]. Do not introduce drugs, "
        "classes, or mechanisms that are not in CONTEXT. If, even after "
        "combining passages, CONTEXT cannot answer the question, say "
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

    @staticmethod
    def _classify_openai_error(exc: Exception) -> GenerationError:
        """Map an openai SDK exception to a typed GenerationError with a safe message.

        The openai SDK raises subclasses of openai.APIError; we avoid importing
        them at module level to keep the optional-dependency surface small, so we
        inspect the class name and status code instead.
        """
        cls_name = type(exc).__name__
        status = getattr(exc, "status_code", None)

        if cls_name == "RateLimitError" or status == 429:
            return RateLimitGenerationError(
                "Service temporarily unavailable, please try again shortly.",
                original=exc,
            )
        if cls_name == "AuthenticationError" or status == 401:
            return AuthGenerationError(
                "The service is not properly configured. Please contact support.",
                original=exc,
            )
        if cls_name in ("APITimeoutError", "Timeout") or status == 408:
            return TimeoutGenerationError(
                "The request timed out. Please try again.",
                original=exc,
            )
        return ServiceGenerationError(
            "An error occurred while generating the response. Please try again.",
            original=exc,
        )

    def generate(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Optional[List[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
    ) -> GenerationResult:
        # Build message list: system prompt, then prior turns, then current question.
        # Prior turns are plain text (no CONTEXT block) so the model treats them as
        # conversation context rather than retrieval grounding.
        messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        max_turns = max(1, int(os.environ.get("RAG_LLM_HISTORY_TURNS", "12")))
        for turn in (history or [])[-max_turns:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": build_user_prompt(question, chunks)})

        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except GenerationError:
            raise  # already classified
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM API call failed in generate()")
            raise self._classify_openai_error(exc) from exc

        answer = resp.choices[0].message.content or ""
        usage = getattr(resp, "usage", None)
        return GenerationResult(
            answer=answer.strip(),
            model=self.model,
            prompt_tokens=getattr(usage, "prompt_tokens", None) if usage else None,
            completion_tokens=getattr(usage, "completion_tokens", None) if usage else None,
        )


    def generate_stream(
        self,
        question: str,
        chunks: Sequence[RetrievedChunk],
        history: Optional[List[dict]] = None,
        temperature: float = 0.1,
        max_tokens: int = 900,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> GenerationResult:
        """Same as generate() but streams tokens via on_token callback."""
        messages: List[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        max_turns = max(1, int(os.environ.get("RAG_LLM_HISTORY_TURNS", "12")))
        for turn in (history or [])[-max_turns:]:
            messages.append({"role": "user", "content": turn["question"]})
            messages.append({"role": "assistant", "content": turn["answer"]})
        messages.append({"role": "user", "content": build_user_prompt(question, chunks)})

        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            full_text = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    full_text += delta.content
                    if on_token is not None:
                        on_token(delta.content)
        except GenerationError:
            raise  # already classified
        except Exception as exc:  # noqa: BLE001
            logger.exception("LLM API call failed in generate_stream()")
            raise self._classify_openai_error(exc) from exc

        return GenerationResult(
            answer=full_text.strip(),
            model=self.model,
            prompt_tokens=None,
            completion_tokens=None,
        )


_default_generator: Optional[DeepSeekGenerator] = None


def get_generator() -> DeepSeekGenerator:
    global _default_generator
    if _default_generator is None:
        _default_generator = DeepSeekGenerator()
    return _default_generator
