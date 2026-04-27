"""PK-aware query rewrite for the retriever.

Maps lay-person paraphrases of pharmacokinetic questions onto the SPL/PK
vocabulary used in FDA drug labels (Structured Product Labeling), so dense
and BM25 retrieval can hit the right "Pharmacokinetics" / "Clinical
Pharmacology" sections without a re-ingest.

Design notes
------------
- **Append, never substitute.** The user's original wording stays in the query
  so it still drives BM25 token overlap and dense-encoder semantics; canonical
  PK terms are appended.
- **Word-boundary regex, not substring matching.** Avoids triggering on the
  literal phrase appearing inside an unrelated sentence (e.g. "how long is
  the bottle").
- **One expansion per intent.** Multiple paraphrase patterns can fire for the
  same intent; the synonym block is added once.
- **Curated, auditable.** All intents and patterns live in one table — adding
  a new PK intent is a single tuple entry.
- **Plain-ASCII expansions.** They need to survive the BM25 tokenizer and the
  dense encoder cleanly, so no symbols beyond ``-``.

Run as a script for a fast smoke test::

    python -m rag.query_rewrite
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Tuple

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _PKIntent:
    """A single PK question intent with its paraphrase patterns and expansion."""

    name: str
    patterns: Tuple[re.Pattern[str], ...]
    expansion: str


def _p(*regexes: str) -> Tuple[re.Pattern[str], ...]:
    """Compile a list of case-insensitive paraphrase patterns."""
    return tuple(re.compile(r, re.IGNORECASE) for r in regexes)


# ---------------------------------------------------------------------------
# Intent table.
#
# Each entry should answer:
#   * "What does the user actually want to know?"  (intent)
#   * "What words appear in the SPL label that talk about that?" (expansion)
#
# Keep expansions compact (≈5–10 tokens). Long expansions dilute dense-vector
# retrieval and crowd out the user's actual question in BM25.
# ---------------------------------------------------------------------------

_PK_INTENTS: Tuple[_PKIntent, ...] = (
    # ---- "How long does <drug> stay in my body?" --------------------------
    # Maps to: elimination half-life, clearance, washout, terminal t1/2.
    _PKIntent(
        name="elimination_duration",
        patterns=_p(
            # how long does/will it stay/last/remain/persist in (my|your|the)
            # body / system / blood / plasma / urine / saliva / hair
            r"\bhow long\b[^?]*\b(?:stay|stays|stayed|last|lasts|lasted|"
            r"remain|remains|persist|persists)\b",
            r"\bhow long\b[^?]*\bin\s+(?:my|your|the|one'?s?)?\s*"
            r"(?:body|system|bloodstream|blood|plasma|urine|saliva|hair)\b",
            # how long until/before it's out / cleared / eliminated / gone
            r"\bhow long\b[^?]*\b(?:before|until)\b[^?]*\b"
            r"(?:out of|leave|leaves|gone|cleared|eliminated)\b",
            # detectability / drug-test framing
            r"\bhow long\b[^?]*\b(?:detect|detectable|show up|test positive)\b",
            # already-canonical wording — boost it anyway so cohorts of related
            # tokens (clearance, washout) come along.
            r"\b(?:wash[- ]?out|washout|clearance|elimination)\b",
            r"\bhalf[- ]?life\b",
        ),
        expansion=(
            "elimination half-life clearance terminal half-life t1/2 "
            "washout duration"
        ),
    ),
    # ---- "How long until <drug> works / kicks in?" ------------------------
    # Maps to: onset of action, time to peak (Tmax), absorption.
    _PKIntent(
        name="onset_tmax",
        patterns=_p(
            # how long to/until/before it works / kicks in / takes effect /
            # starts working / I feel relief
            r"\bhow long\b[^?]*\b(?:to|until|before)\b[^?]*\b"
            r"(?:work|works|kick[- ]?in|kicks[- ]?in|"
            r"take effect|takes effect|start working|starts working|"
            r"feel|relief|effective)\b",
            # how fast / how quickly / how soon does it work / act / kick in
            r"\bhow (?:fast|quickly|soon)\b[^?]*\b"
            r"(?:work|works|act|acts|kick|kicks|effect|effective|relief)\b",
            # when does/will it start/kick in/take effect
            r"\bwhen\b[^?]*\b(?:does|will|do|should)\b[^?]*\b"
            r"(?:start|kick|work|take effect|effective)\b",
            # already-canonical wording — boost it.
            r"\bonset\b",
            r"\b(?:time to peak|t\s*-?\s*max|peak\s+(?:plasma\s+)?"
            r"(?:concentration|level)s?)\b",
        ),
        expansion=(
            "onset of action time to peak Tmax peak plasma concentration "
            "absorption"
        ),
    ),
)


@dataclass(frozen=True)
class QueryRewrite:
    """Result of running :func:`expand_pk_query` over a user query."""

    original: str
    rewritten: str
    matched: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def changed(self) -> bool:
        return self.rewritten != self.original


def expand_pk_query(query: str) -> QueryRewrite:
    """Return ``query`` plus PK synonyms when a known intent is recognized.

    The user's wording is preserved verbatim (so it still drives BM25 token
    overlap and dense-encoder semantics); canonical PK terms are *appended*,
    never substituted.
    """
    if not query or not query.strip():
        return QueryRewrite(original=query, rewritten=query)

    matched: List[str] = []
    extras: List[str] = []
    seen: set[str] = set()

    for intent in _PK_INTENTS:
        if any(p.search(query) for p in intent.patterns):
            matched.append(intent.name)
            if intent.expansion not in seen:
                seen.add(intent.expansion)
                extras.append(intent.expansion)

    if not extras:
        return QueryRewrite(original=query, rewritten=query)

    rewritten = f"{query.rstrip()} {' '.join(extras)}"
    logger.debug("PK query rewrite: intents=%s rewritten=%r", matched, rewritten)
    return QueryRewrite(
        original=query, rewritten=rewritten, matched=tuple(matched)
    )


# ---------------------------------------------------------------------------
# Smoke tests — run with `python -m rag.query_rewrite`.
# Lightweight assertions instead of pulling in pytest. Covers the two intents
# the retriever depends on, plus a handful of negative cases.
# ---------------------------------------------------------------------------

_POSITIVE_CASES: Tuple[Tuple[str, str, str], ...] = (
    # (query, expected matched intent, expected substring in expansion)
    ("How long does Adderall stay in your body?", "elimination_duration", "half-life"),
    ("how long does it stay in plasma", "elimination_duration", "clearance"),
    ("How long until metoprolol is out of my system?", "elimination_duration", "washout"),
    ("How long can it be detected in urine?", "elimination_duration", "half-life"),
    ("What is the elimination half-life of warfarin?", "elimination_duration", "t1/2"),
    ("how long to work?", "onset_tmax", "Tmax"),
    ("How long until ibuprofen kicks in?", "onset_tmax", "onset of action"),
    ("how fast does it work", "onset_tmax", "Tmax"),
    ("When does this start working?", "onset_tmax", "onset of action"),
    ("What is the time to peak plasma concentration?", "onset_tmax", "absorption"),
)

_NEGATIVE_CASES: Tuple[str, ...] = (
    "",
    "   ",
    "What is the maximum daily dose of acetaminophen?",
    "Can I take this with food?",
    "Is this drug safe in pregnancy?",
    "How long is the bottle good for after opening?",  # "how long" w/o PK context
)


def _smoke_test() -> None:
    failures: List[str] = []

    for query, intent, needle in _POSITIVE_CASES:
        result = expand_pk_query(query)
        if intent not in result.matched:
            failures.append(
                f"[positive] {query!r}: expected intent {intent!r}, "
                f"got matched={result.matched!r}"
            )
        if needle.lower() not in result.rewritten.lower():
            failures.append(
                f"[positive] {query!r}: expansion missing {needle!r} "
                f"in {result.rewritten!r}"
            )
        if not result.rewritten.lower().startswith(query.strip().lower()):
            failures.append(
                f"[positive] {query!r}: original wording not preserved at "
                f"start of {result.rewritten!r}"
            )

    for query in _NEGATIVE_CASES:
        result = expand_pk_query(query)
        if result.changed:
            failures.append(
                f"[negative] {query!r}: unexpectedly rewritten to "
                f"{result.rewritten!r} (matched={result.matched!r})"
            )

    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print(
        f"ok — {len(_POSITIVE_CASES)} positive, "
        f"{len(_NEGATIVE_CASES)} negative cases passed"
    )


if __name__ == "__main__":
    _smoke_test()
