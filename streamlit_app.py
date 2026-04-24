"""Streamlit UI — single-file demo that calls the RAG pipeline directly.

Run (uses project `.venv` — avoids Homebrew Python / broken TensorFlow):
    ./run.sh

Or, after `source .venv/bin/activate`:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import os

# Before any `rag.*` import — ensures transformers never probes TensorFlow.
import rag._transformers_env  # noqa: F401

from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(
    page_title="Drug Interaction RAG",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Cached loaders — keep heavy model init out of the per-request path.
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner="Loading embedding + retrieval stack…")
def _load_store():
    from rag.retrieve import get_store

    return get_store()


@st.cache_resource(show_spinner="Loading PII redactor…")
def _load_redactor():
    from rag.redact import get_redactor

    return get_redactor()


@st.cache_resource(show_spinner="Loading cross-encoder reranker…")
def _load_reranker():
    from rag.rerank import get_reranker

    return get_reranker()


@st.cache_resource(show_spinner="Loading RxNorm drug detector…")
def _load_detector():
    from rag.drug_detect import get_detector

    return get_detector()


@st.cache_resource(show_spinner=False)
def _load_pipeline():
    from rag.pipeline import RAGPipeline

    return RAGPipeline(
        store=_load_store(),
        redactor=_load_redactor(),
        reranker=_load_reranker(),
        detector=_load_detector(),
    )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

store = _load_store()
redactor = _load_redactor()
stats = store.stats()

with st.sidebar:
    st.markdown("### Corpus")
    st.metric("Chunks", f"{stats['n_chunks']:,}")
    st.metric("Drugs covered", stats["n_drugs"])
    st.metric("Unique source URLs", stats["n_sources"])
    st.caption(f"Embedding: `{stats['embedding_model']}`")
    st.caption(f"Collection: `{stats['collection']}`")
    _drug_list = stats.get("drugs") or []
    if _drug_list:
        with st.expander("Indexed drugs", expanded=False):
            st.write(", ".join(_drug_list))
    st.divider()

    st.markdown("### Pipeline")
    top_k_retrieve = st.slider("Retrieve top-k", 5, 50, 20, step=5)
    top_k_rerank = st.slider("Rerank top-k", 1, 10, 5)
    auto_ingest = st.toggle(
        "Auto-ingest unknown drugs (RxNorm)",
        value=True,
        help="If the user mentions a drug we haven't indexed yet, "
        "fetch it from DailyMed + MedlinePlus on the fly.",
    )
    skip_generation = st.toggle(
        "Skip LLM call (retrieval only)",
        value=not bool(os.environ.get("OPENROUTER_API_KEY")),
    )
    st.caption(f"PII backend: `{redactor.backend}`")
    st.caption(f"LLM model: `{os.environ.get('OPENROUTER_MODEL', 'deepseek/deepseek-v3.2-exp')}`")
    st.divider()
    st.markdown(
        "⚠️ **Educational demo only.** Answers are grounded in FDA "
        "DailyMed / MedlinePlus content, but this is not medical advice."
    )


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    # Each entry: {"role": "user"|"assistant", "content": str}
    st.session_state.messages = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# ---------------------------------------------------------------------------
# Sample questions in sidebar
# ---------------------------------------------------------------------------

sample_questions = [
    "Can I take ibuprofen with lisinopril for my blood pressure?",
    "Is it safe to combine warfarin and aspirin?",
    "I'm John Smith (john@example.com). Does metformin interact with alcohol?",
    "What happens if I take sertraline and tramadol together?",
    "Can I take omeprazole while on clopidogrel?",
    "Does rifampin reduce the effectiveness of warfarin?",
]

with st.sidebar:
    st.divider()
    st.markdown("### Sample questions")
    chosen_sample = st.selectbox(
        "Pick one to send",
        options=["(select)"] + sample_questions,
        key="sample_sel",
    )
    if st.button("Use sample", disabled=(chosen_sample == "(select)")):
        st.session_state["_pending"] = chosen_sample

    st.divider()
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.last_result = None
        st.rerun()

# ---------------------------------------------------------------------------
# Main pane
# ---------------------------------------------------------------------------

st.title("💊 Drug Interaction RAG")
st.caption(
    "Multi-turn conversation — ask follow-up questions naturally. "
    "Every question is PII-redacted, drug-detected, retrieved via hybrid "
    "BM25 + dense search, reranked, and answered by DeepSeek v3.2 with citations."
)


def _render_chunk_card(
    idx: int,
    chunk,
    ordinal: str,
    *,
    use_expander: bool = True,
    expand_first: bool = False,
):
    """Render one citation chunk."""
    meta = chunk.metadata or {}
    drug = meta.get("drug_name", "unknown")
    section = meta.get("section", "")
    source_url = meta.get("source_url", "")
    source = meta.get("source", "")
    title = f"{ordinal} · {drug} — {section} · score {chunk.score:.3f}"

    def _body() -> None:
        if source_url:
            st.markdown(f"**Source:** [{source or source_url}]({source_url})")
        st.write(chunk.text)

    if use_expander:
        with st.expander(title, expanded=expand_first):
            _body()
    else:
        st.markdown(f"**{title}**")
        _body()
        st.divider()


def _render_result_panels(result) -> None:
    """Render the four pipeline detail panels for a RAGResult."""
    # --- Redaction ---------------------------------------------------------
    st.subheader("1 · PII redaction")
    rc1, rc2 = st.columns(2)
    with rc1:
        st.markdown("**Original question**")
        st.code(result.redaction.original, language="text")
    with rc2:
        st.markdown("**Redacted (sent downstream)**")
        st.code(result.redaction.redacted, language="text")
    if result.redaction.had_pii:
        st.warning(
            f"Detected {len(result.redaction.entities)} PII entit"
            f"{'y' if len(result.redaction.entities) == 1 else 'ies'}: "
            + ", ".join(sorted({e["entity_type"] for e in result.redaction.entities}))
        )
    else:
        st.success("No PII detected.")

    # --- Drug detection ----------------------------------------------------
    st.subheader("2 · Drug detection & auto-ingest")
    if not result.detected_drugs:
        st.info("RxNorm didn't recognize any drug names. Retrieval runs over the full corpus.")
    else:
        cols = st.columns(min(len(result.detected_drugs), 4) or 1)
        for i, d in enumerate(result.detected_drugs):
            with cols[i % len(cols)]:
                st.metric(d.canonical, f"{d.score:.0f}/100", help=f"RxCUI {d.rxcui}")

    ai = result.auto_ingest
    if ai.skipped:
        st.caption("Auto-ingest disabled in the sidebar.")
    elif ai.missing:
        names = ", ".join(m.canonical for m in ai.missing)
        if ai.added_chunks > 0:
            st.success(f"Learned about **{names}** on the fly — added {ai.added_chunks} new chunks.")
        elif ai.error:
            st.warning(f"Auto-ingest note for {names}: {ai.error}")
        else:
            st.info(f"Tried to ingest **{names}** but no upstream documents were returned.")
    else:
        st.caption("All detected drugs are already in the corpus.")

    # --- Answer ------------------------------------------------------------
    st.subheader("3 · Answer")
    if result.error:
        st.error(f"Generation failed: {result.error}")
    elif result.generation:
        st.markdown(result.generation.answer)
        footer_bits = [f"model `{result.generation.model}`"]
        if result.generation.prompt_tokens is not None:
            footer_bits.append(
                f"{result.generation.prompt_tokens} prompt / "
                f"{result.generation.completion_tokens} completion tokens"
            )
        st.caption(" · ".join(footer_bits))
    else:
        st.info("LLM call skipped — toggle **Skip LLM call** off in the sidebar.")

    # --- Citations ---------------------------------------------------------
    st.subheader("4 · Citations (post-rerank)")
    if not result.reranked:
        st.info("No relevant passages were retrieved.")
    else:
        for i, c in enumerate(result.reranked):
            _render_chunk_card(i, c, ordinal=f"[{i + 1}]", expand_first=(i == 0))

    # --- Debug -------------------------------------------------------------
    with st.expander("Debug: all retrieved (pre-rerank) + timings", expanded=False):
        t = result.timing
        st.write(
            {
                "redact_ms": round(t.redact_ms, 1),
                "detect_ms": round(t.detect_ms, 1),
                "ingest_ms": round(t.ingest_ms, 1),
                "retrieve_ms": round(t.retrieve_ms, 1),
                "rerank_ms": round(t.rerank_ms, 1),
                "generate_ms": round(t.generate_ms, 1),
                "total_ms": round(t.total_ms, 1),
            }
        )
        for i, c in enumerate(result.retrieved):
            _render_chunk_card(i, c, ordinal=f"R{i + 1}", use_expander=False)


def _build_history() -> list:
    """Extract completed (question, answer) pairs from session messages for the generator."""
    msgs = st.session_state.messages
    history = []
    for i in range(0, len(msgs) - 1, 2):
        if msgs[i]["role"] == "user" and i + 1 < len(msgs) and msgs[i + 1]["role"] == "assistant":
            history.append({"question": msgs[i]["content"], "answer": msgs[i + 1]["content"]})
    return history


# ---------------------------------------------------------------------------
# Render conversation history
# ---------------------------------------------------------------------------

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Show detail panels for the most recent result (below the last assistant bubble).
if st.session_state.last_result:
    _render_result_panels(st.session_state.last_result)

# ---------------------------------------------------------------------------
# Handle new input
# ---------------------------------------------------------------------------

# Grab a pending sample question if the sidebar button was clicked.
question = st.session_state.pop("_pending", None)
# Chat input always anchors to the bottom of the page.
chat_input = st.chat_input("Ask about drug interactions…")
if chat_input:
    question = chat_input

if question and question.strip():
    question = question.strip()

    # Show the user bubble immediately.
    with st.chat_message("user"):
        st.markdown(question)

    pipeline = _load_pipeline()
    pipeline.top_k_retrieve = top_k_retrieve
    pipeline.top_k_rerank = top_k_rerank

    status_box = st.status("Running pipeline…", expanded=True)

    def _on_status(msg: str) -> None:
        status_box.write(f"• {msg}")

    history = _build_history()

    result = pipeline.run(
        question,
        skip_generation=skip_generation,
        auto_ingest=auto_ingest,
        on_status=_on_status,
        history=history,
    )
    status_box.update(label="Pipeline complete", state="complete", expanded=False)

    # Derive the assistant's text for the chat bubble.
    if result.generation:
        answer_text = result.generation.answer
    elif result.error:
        answer_text = f"_(generation failed: {result.error})_"
    else:
        answer_text = "_(LLM call skipped — retrieval only)_"

    with st.chat_message("assistant"):
        st.markdown(answer_text)

    # Persist to session state.
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer_text})
    st.session_state.last_result = result

    st.rerun()
