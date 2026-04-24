"""Detect drug mentions in a query using NIH's RxNorm API.

RxNorm is the authoritative drug terminology maintained by the U.S. National
Library of Medicine. The `approximateTerm` endpoint is free, no-auth, and
returns canonical drug names for brand names, generics, and common
misspellings:

    GET https://rxnav.nlm.nih.gov/REST/approximateTerm.json?term=Advil&maxEntries=1
    → { candidate: [{rxcui:"5640", name:"ibuprofen", score:"100", rank:"1"}] }

We tokenize the (redacted) query, build 1- and 2-gram candidates, skip
obvious non-drug tokens via a stopword list, and look each up in RxNorm.
Results are cached in-process so repeated queries in the same API process
don't hammer the API.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, List, Optional, Set

import requests

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"
DEFAULT_SCORE_THRESHOLD = 80  # RxNorm scores are 0-100; ≥80 is a confident match

# Common English words that are occasionally drug-like in RxNorm's fuzzy index.
# Cheap filter to save HTTP round-trips and avoid "tea" → tea-tree-oil matches.
_STOPWORDS: Set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "could",
    "do", "does", "doing", "don", "for", "from", "had", "has", "have",
    "having", "he", "her", "here", "hers", "him", "his", "how", "i",
    "if", "in", "into", "is", "it", "its", "just", "me", "my", "no",
    "not", "now", "of", "on", "only", "or", "other", "our", "out",
    "over", "own", "safe", "she", "should", "so", "some", "such",
    "take", "taking", "than", "that", "the", "their", "them", "then",
    "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "up", "use", "used", "using", "very", "was", "we", "well",
    "what", "when", "where", "which", "while", "who", "why", "will",
    "with", "would", "you", "your", "yours", "yourself",
    "medication", "medications", "medicine", "medicines", "drug", "drugs",
    "pill", "pills", "dose", "doses", "dosage", "prescription", "tablet",
    "tablets", "daily", "pressure", "blood", "interact", "interacts",
    "interaction", "interactions", "between", "risk", "safety",
    "morning", "evening", "night", "today", "tomorrow", "time",
    "person", "email_address", "phone_number",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]{2,}")


@dataclass(frozen=True)
class DrugMention:
    mention: str  # the surface form from the user's text
    canonical: str  # RxNorm canonical name, lowercased
    rxcui: str
    score: float  # 0-100


def _candidates(text: str) -> List[str]:
    """Extract 1- and 2-gram candidates from the query, minus stopwords and
    bracketed PII placeholders."""
    # Drop anything inside [BRACKETS] (e.g. [PERSON]) first.
    cleaned = re.sub(r"\[[A-Z_]+\]", " ", text)
    tokens = [t.lower() for t in _TOKEN_RE.findall(cleaned)]
    unigrams = [t for t in tokens if t not in _STOPWORDS]
    bigrams: List[str] = []
    for a, b in zip(tokens, tokens[1:]):
        if a in _STOPWORDS or b in _STOPWORDS:
            continue
        bigrams.append(f"{a} {b}")
    # Dedupe preserving order; prefer bigrams first (more specific).
    seen: Set[str] = set()
    out: List[str] = []
    for c in bigrams + unigrams:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


class DrugDetector:
    """RxNorm-backed drug mention detector with an in-memory cache."""

    def __init__(
        self,
        score_threshold: float = DEFAULT_SCORE_THRESHOLD,
        timeout: float = 6.0,
        max_candidates: int = 24,
    ):
        self.score_threshold = score_threshold
        self.timeout = timeout
        self.max_candidates = max_candidates
        self._cache: dict = {}  # term -> Optional[DrugMention]
        self._lock = Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "drug-rag-demo/0.1 (educational)"}
        )

    def detect(self, text: str) -> List[DrugMention]:
        if not text or not text.strip():
            return []
        mentions: List[DrugMention] = []
        seen_canonical: Set[str] = set()
        for cand in _candidates(text)[: self.max_candidates]:
            try:
                hit = self._lookup(cand)
            except Exception as exc:  # noqa: BLE001
                logger.debug("RxNorm lookup failed for %r: %s", cand, exc)
                continue
            if hit is None:
                continue
            if hit.canonical in seen_canonical:
                continue
            seen_canonical.add(hit.canonical)
            mentions.append(hit)
        return mentions

    def _lookup(self, term: str) -> Optional[DrugMention]:
        with self._lock:
            if term in self._cache:
                return self._cache[term]

        url = f"{RXNORM_BASE}/approximateTerm.json"
        params = {"term": term, "maxEntries": 1}
        r = self._session.get(url, params=params, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        candidates = (
            (data.get("approximateGroup") or {}).get("candidate") or []
        )
        result: Optional[DrugMention] = None
        if candidates:
            top = candidates[0]
            try:
                score = float(top.get("score", 0))
            except (TypeError, ValueError):
                score = 0.0
            name = (top.get("name") or "").strip().lower()
            rxcui = str(top.get("rxcui") or "")
            if score >= self.score_threshold and name and rxcui:
                result = DrugMention(
                    mention=term, canonical=name, rxcui=rxcui, score=score
                )

        with self._lock:
            self._cache[term] = result
        return result


_default_detector: Optional[DrugDetector] = None


def get_detector() -> DrugDetector:
    global _default_detector
    if _default_detector is None:
        _default_detector = DrugDetector()
    return _default_detector


def missing_drugs(
    mentions: Iterable[DrugMention], indexed: Iterable[str]
) -> List[DrugMention]:
    indexed_set = {d.lower() for d in indexed}
    return [m for m in mentions if m.canonical.lower() not in indexed_set]
