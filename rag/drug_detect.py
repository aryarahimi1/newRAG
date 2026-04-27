"""Detect drug mentions in a query using NIH's RxNorm API.

RxNorm is the authoritative drug terminology maintained by the U.S. National
Library of Medicine. The `approximateTerm` endpoint is free, no-auth, and
resolves brand names to ingredients where possible (`drugs.json` + IN relationships),
with `approximateTerm` as a fuzzy fallback (scores are implementation-defined floats, not 0–100).

We tokenize the (redacted) query, build 1- and 2-gram candidates, skip
obvious non-drug tokens via a stopword list, and look each up in RxNorm.
Results are cached in-process so repeated queries in the same API process
don't hammer the API.
"""

from __future__ import annotations

import logging
import os
import re
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterable, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

# Fuzzy score threshold — raised from 0.0 to 8.0 to cut false positives.
# `approximateTerm` scores are internal similarity floats (not 0–100), typically
# 10–20 for genuine drug matches. Override via env var for tuning without redeploy.
DEFAULT_APPROX_MIN_SCORE: float = float(
    os.environ.get("RXNORM_APPROX_MIN_SCORE", "8.0")
)

# Scores at or above this boundary are labelled "high" confidence; below is "low".
_HIGH_CONF_SCORE: float = float(
    os.environ.get("RXNORM_HIGH_CONF_SCORE", "12.0")
)

# Bounded LRU cache constants.
_CACHE_MAX_SIZE = 512
_CACHE_TTL_SECS = 86400  # 24 hours

# CamelCase-like pattern common in drug brand names (e.g. "BuSpar", "HumaLOG").
_BRAND_CAMEL_RE = re.compile(r"[A-Z][a-z]{2,}[A-Z]")

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
    mention: str          # the surface form from the user's text
    canonical: str        # preferred: RxNorm ingredient (IN) name when available, lowercased
    rxcui: str            # preferred: IN RxCUI when resolved; otherwise the matched concept
    score: float          # higher = more confident where applicable
    ingest_aliases: Tuple[str, ...] = ()  # extra strings to try on DailyMed / MedlinePlus
    # "high" = drugs.json exact match or approx score >= RXNORM_HIGH_CONF_SCORE
    # "low"  = approx score in [RXNORM_APPROX_MIN_SCORE, RXNORM_HIGH_CONF_SCORE)
    confidence: str = "high"


@dataclass
class RedactionFlag:
    """Audit record for a Presidio span that may have masked a drug name."""
    original_token: str
    presidio_entity: str
    presidio_score: float
    rxnorm_hit: bool
    rxcui: str           # empty string when rxnorm_hit is False
    warning: str


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
    """RxNorm-backed drug mention detector with a bounded LRU in-memory cache."""

    def __init__(
        self,
        approx_min_score: float = DEFAULT_APPROX_MIN_SCORE,
        timeout: float = 6.0,
        max_candidates: int = 24,
    ):
        self.approx_min_score = approx_min_score
        self.timeout = timeout
        self.max_candidates = max_candidates
        # OrderedDict used as a bounded LRU: eldest entry at front, newest at back.
        # Values are (Optional[DrugMention], float) tuples — result plus insert timestamp.
        self._cache: OrderedDict = OrderedDict()
        self._lock = Lock()
        self._session = requests.Session()
        self._session.headers.update(
            {"User-Agent": "medication-reference-demo/0.1 (educational)"}
        )

    # ------------------------------------------------------------------
    # Public API — must remain sync (called from threading.Thread context)
    # ------------------------------------------------------------------

    def detect(self, text: str) -> List[DrugMention]:  # MUST STAY SYNC
        """Run all candidate lookups in parallel via ThreadPoolExecutor.

        Workers are capped at min(max_candidates, 10) to stay rate-courteous
        toward the public RxNorm API.
        """
        if not text or not text.strip():
            return []

        cands = _candidates(text)[: self.max_candidates]
        if not cands:
            return []

        worker_count = min(len(cands), 10)
        results: dict[str, Optional[DrugMention]] = {}

        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            future_to_cand = {pool.submit(self._lookup, c): c for c in cands}
            for future in as_completed(future_to_cand):
                cand = future_to_cand[future]
                try:
                    results[cand] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("RxNorm lookup failed for %r: %s", cand, exc)
                    results[cand] = None

        # Re-impose original candidate order and deduplicate by canonical name.
        mentions: List[DrugMention] = []
        seen_canonical: Set[str] = set()
        for cand in cands:
            hit = results.get(cand)
            if hit is None:
                continue
            if hit.canonical in seen_canonical:
                continue
            seen_canonical.add(hit.canonical)
            mentions.append(hit)
        return mentions

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ingredient_for_rxcui(self, rxcui: str) -> Optional[Tuple[str, str]]:
        """Return (ingredient_rxcui, ingredient_name_lower) if RxNorm lists an IN."""
        if not rxcui:
            return None
        url = f"{RXNORM_BASE}/rxcui/{rxcui}/related.json"
        r = self._session.get(url, params={"tty": "IN"}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        groups = ((data.get("relatedGroup") or {}).get("conceptGroup")) or []
        for cg in groups:
            if (cg.get("tty") or "").upper() != "IN":
                continue
            for cp in cg.get("conceptProperties") or []:
                in_rxcui = str(cp.get("rxcui") or "").strip()
                name = (cp.get("name") or "").strip().lower()
                if in_rxcui and name:
                    return in_rxcui, name
        return None

    @staticmethod
    def _drugs_json_row_score(term_l: str, cp: dict) -> int:
        """Higher = better alignment between user term and this RxNorm row."""
        name = (cp.get("name") or "").strip().lower()
        syn = (cp.get("synonym") or "").strip().lower()
        if not name:
            return -1
        base = name.split("[", 1)[0].strip()
        if name == term_l:
            return 1000
        if base == term_l:
            return 990
        if base.startswith(term_l + " "):
            if " / " not in base:
                return 960
            return 450
        if f"[{term_l}]" in name:
            return 940
        if syn == term_l:
            return 920
        if syn.startswith(term_l + " "):
            return 900
        return -1

    def _lookup_drugs_json(self, term: str) -> Optional[DrugMention]:
        """Strong match via RxNorm name lookup (brands, many generics)."""
        term_l = term.strip().lower()
        url = f"{RXNORM_BASE}/drugs.json"
        r = self._session.get(url, params={"name": term}, timeout=self.timeout)
        r.raise_for_status()
        data = r.json() or {}
        groups = ((data.get("drugGroup") or {}).get("conceptGroup")) or []
        rows: List[Tuple[int, int, int, str, dict]] = []
        tty_rank = {"IN": 0, "SCD": 1, "BN": 2, "SBD": 3}.get
        for cg in groups:
            tty = (cg.get("tty") or "").upper()
            for cp in cg.get("conceptProperties") or []:
                score = self._drugs_json_row_score(term_l, cp)
                if score < 0:
                    continue
                rxcui = str(cp.get("rxcui") or "").strip()
                nm = (cp.get("name") or "").strip()
                if not rxcui or not nm:
                    continue
                tr = tty_rank(tty, 4)
                rows.append((score, tr, len(nm), tty, cp))

        if not rows:
            return None
        rows.sort(key=lambda x: (-x[0], x[1], x[2]))
        _, _, _, tty, best = rows[0]

        rxcui = str(best.get("rxcui") or "").strip()
        nm = (best.get("name") or "").strip()
        # drugs.json exact matches are always high confidence.
        if tty == "IN":
            return DrugMention(
                mention=term,
                canonical=nm.lower(),
                rxcui=rxcui,
                score=100.0,
                ingest_aliases=self._alias_tuple(term, nm),
                confidence="high",
            )

        ing = self._ingredient_for_rxcui(rxcui)
        if ing:
            in_rxcui, in_name = ing
            return DrugMention(
                mention=term,
                canonical=in_name,
                rxcui=in_rxcui,
                score=100.0,
                ingest_aliases=self._alias_tuple(term, nm, in_name),
                confidence="high",
            )
        return DrugMention(
            mention=term,
            canonical=nm.lower(),
            rxcui=rxcui,
            score=100.0,
            ingest_aliases=self._alias_tuple(term, nm),
            confidence="high",
        )

    @staticmethod
    def _alias_tuple(term: str, *extra_names: str) -> Tuple[str, ...]:
        """Brand / salt synonyms to try if the generic DailyMed query fails."""
        out: List[str] = []
        seen: Set[str] = set()
        for raw in (term,) + extra_names:
            s = (raw or "").strip().lower()
            if len(s) < 3 or s in seen:
                continue
            seen.add(s)
            out.append(s)
            # "lisdexamfetamine dimesylate 10 MG ... [Vyvanse]" → try short prefix
            if " mg " in s:
                short = s.split("[", 1)[0].strip()
                if short and short not in seen and len(short) >= 5:
                    seen.add(short)
                    out.append(short)
        return tuple(out)

    def _lookup_approximate(self, term: str) -> Optional[DrugMention]:
        """Fuzzy fallback; uses rank + score ordering (not a 0–100 scale)."""
        url = f"{RXNORM_BASE}/approximateTerm.json"
        r = self._session.get(
            url, params={"term": term, "maxEntries": 15}, timeout=self.timeout
        )
        r.raise_for_status()
        data = r.json() or {}
        raw = ((data.get("approximateGroup") or {}).get("candidate")) or []
        if isinstance(raw, dict):
            raw = [raw]
        candidates = [c for c in raw if c.get("name") and c.get("rxcui")]
        if not candidates:
            return None

        def sort_key(c: dict) -> Tuple[int, float]:
            try:
                rank = int(c.get("rank", 999))
            except (TypeError, ValueError):
                rank = 999
            try:
                score = float(c.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            return (rank, -score)

        candidates.sort(key=sort_key)
        top = candidates[0]
        try:
            score = float(top.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0

        # Reject anything below the configured threshold (default 8.0).
        if score < self.approx_min_score:
            return None

        rxcui = str(top.get("rxcui") or "").strip()
        base_name = (top.get("name") or "").strip()
        if not rxcui or not base_name:
            return None

        # Two-tier confidence: high >= RXNORM_HIGH_CONF_SCORE, low otherwise.
        confidence = "high" if score >= _HIGH_CONF_SCORE else "low"

        ing = self._ingredient_for_rxcui(rxcui)
        if ing:
            in_rxcui, in_name = ing
            return DrugMention(
                mention=term,
                canonical=in_name,
                rxcui=in_rxcui,
                score=min(99.0, max(50.0, score)),
                ingest_aliases=self._alias_tuple(term, base_name, in_name),
                confidence=confidence,
            )
        return DrugMention(
            mention=term,
            canonical=base_name.lower(),
            rxcui=rxcui,
            score=min(99.0, max(50.0, score)),
            ingest_aliases=self._alias_tuple(term, base_name),
            confidence=confidence,
        )

    def _lookup(self, term: str) -> Optional[DrugMention]:
        """Cache-backed lookup: try drugs.json first, fall back to approximateTerm.

        Cache is a bounded LRU (max 512 entries, 24-hour TTL). The same lock
        guards both read and write so concurrent threads sharing this instance
        don't race on the OrderedDict.
        """
        now = time.time()

        with self._lock:
            if term in self._cache:
                cached_result, ts = self._cache[term]
                if now - ts <= _CACHE_TTL_SECS:
                    # Move to end (most-recently-used position).
                    self._cache.move_to_end(term)
                    return cached_result
                # Entry expired — evict and fall through to a fresh lookup.
                del self._cache[term]

        result: Optional[DrugMention] = None
        try:
            result = self._lookup_drugs_json(term)
        except Exception as exc:  # noqa: BLE001
            logger.debug("RxNorm drugs.json failed for %r: %s", term, exc)
        if result is None:
            try:
                result = self._lookup_approximate(term)
            except Exception as exc:  # noqa: BLE001
                logger.debug("RxNorm approximateTerm failed for %r: %s", term, exc)

        with self._lock:
            # Another thread may have written the same key while we were fetching;
            # overwrite with the freshest result and ensure LRU invariant holds.
            self._cache[term] = (result, time.time())
            self._cache.move_to_end(term)
            # Evict oldest entry when over capacity.
            if len(self._cache) > _CACHE_MAX_SIZE:
                self._cache.popitem(last=False)

        return result


class RedactionAuditFilter:
    """Inspect Presidio recognizer results for tokens that may be drug names.

    Presidio's NER sometimes misclassifies drug brand names as PERSON / ORG /
    MISC, causing them to be redacted before the drug detector sees them. This
    filter audits low-confidence Presidio spans against RxNorm and emits a
    RedactionFlag for anything suspicious so callers can warn the user.
    """

    # Entity types that sometimes shadow drug names.
    _AUDIT_ENTITIES: Set[str] = {"PERSON", "ORG", "MISC"}
    # Presidio scores below this threshold trigger the audit.
    _CONF_GATE: float = 0.92
    # Token character-length range worth auditing.
    _MIN_TOKEN_LEN: int = 4
    _MAX_TOKEN_LEN: int = 15

    @staticmethod
    def audit(
        original_text: str,
        recognizer_results: list,
        detector: "DrugDetector",
    ) -> List[RedactionFlag]:
        """Return a list of RedactionFlags for spans that may mask drug names.

        Parameters
        ----------
        original_text:
            The raw (un-redacted) query text, used to extract span tokens by
            character offset.
        recognizer_results:
            List of Presidio RecognizerResult objects.  Each must expose
            `.entity_type` (str), `.score` (float), `.start` (int), `.end` (int).
            If the result also carries a `.text` attribute it is used directly;
            otherwise the token is sliced from `original_text`.
        detector:
            A DrugDetector instance whose `_lookup` method is used for RxNorm
            verification.
        """
        flags: List[RedactionFlag] = []

        for rr in recognizer_results:
            entity_type: str = getattr(rr, "entity_type", "") or ""
            score: float = float(getattr(rr, "score", 0.0))

            # Gate 2: only audit entity types that shadow drug names.
            if entity_type not in RedactionAuditFilter._AUDIT_ENTITIES:
                continue

            # Gate 3: only audit low-confidence Presidio decisions.
            if score >= RedactionAuditFilter._CONF_GATE:
                continue

            # Extract token by offset if .text is not populated.
            token: str = getattr(rr, "text", None) or ""
            if not token:
                start = int(getattr(rr, "start", 0))
                end = int(getattr(rr, "end", 0))
                token = original_text[start:end]

            token = token.strip()

            # Gate 1: length filter.
            if not (RedactionAuditFilter._MIN_TOKEN_LEN <= len(token) <= RedactionAuditFilter._MAX_TOKEN_LEN):
                continue

            # RxNorm lookup on the lower-cased token.
            drug_hit: Optional[DrugMention] = None
            try:
                drug_hit = detector._lookup(token.lower())
            except Exception as exc:  # noqa: BLE001
                logger.debug("RedactionAuditFilter RxNorm lookup failed for %r: %s", token, exc)

            if drug_hit is not None:
                flags.append(RedactionFlag(
                    original_token=token,
                    presidio_entity=entity_type,
                    presidio_score=score,
                    rxnorm_hit=True,
                    rxcui=drug_hit.rxcui,
                    warning=(
                        f"Presidio tagged '{token}' as {entity_type} "
                        f"(score={score:.2f}) but RxNorm matched it to "
                        f"'{drug_hit.canonical}' (rxcui={drug_hit.rxcui}). "
                        "Drug interaction analysis may be incomplete."
                    ),
                ))
            elif _BRAND_CAMEL_RE.search(token):
                # No RxNorm hit, but CamelCase pattern suggests a brand name.
                flags.append(RedactionFlag(
                    original_token=token,
                    presidio_entity=entity_type,
                    presidio_score=score,
                    rxnorm_hit=False,
                    rxcui="",
                    warning=(
                        f"Presidio tagged '{token}' as {entity_type} "
                        f"(score={score:.2f}). The token pattern resembles a "
                        "drug brand name but could not be verified via RxNorm."
                    ),
                ))

        return flags


# ---------------------------------------------------------------------------
# Module-level singleton helpers
# ---------------------------------------------------------------------------

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
