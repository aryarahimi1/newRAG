"""PII redaction layer.

Uses Microsoft Presidio when available, falls back to regex-only redaction if
presidio or its spaCy model are not installed. The regex fallback is
deliberately conservative so the demo still works on a fresh machine.

The redacted text (not the original) is what flows into embedding, retrieval,
and generation. Drug names are *not* considered PII and are preserved so
retrieval still works.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)

_PRESIDIO_ENTITIES = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_SSN",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "LOCATION",
    "DATE_TIME",
    "US_DRIVER_LICENSE",
    "MEDICAL_LICENSE",
]

_REGEX_PATTERNS = {
    "EMAIL_ADDRESS": re.compile(
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
    ),
    "PHONE_NUMBER": re.compile(
        r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"
    ),
    "US_SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "CREDIT_CARD": re.compile(r"\b(?:\d[ -]*?){13,16}\b"),
    # A tolerant "Name" regex that matches common patterns like
    # "I'm John Smith" or "my name is Jane Doe". Intentionally narrow to
    # avoid clobbering drug names (which are often capitalized).
    "PERSON": re.compile(
        r"(?:(?<=\bI am )|(?<=\bI'm )|(?<=\bmy name is )|(?<=\bthis is ))"
        r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?",
        flags=re.IGNORECASE,
    ),
}


@dataclass
class RedactionResult:
    """Output of a redaction pass."""

    original: str
    redacted: str
    entities: List[dict] = field(default_factory=list)

    @property
    def had_pii(self) -> bool:
        return len(self.entities) > 0


class PIIRedactor:
    """Wrapper around Presidio with a regex fallback."""

    def __init__(self, use_presidio: bool = True, language: str = "en"):
        self._language = language
        self._analyzer = None
        self._anonymizer = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                from presidio_analyzer.nlp_engine import NlpEngineProvider
                from presidio_anonymizer import AnonymizerEngine

                # Default Presidio config points at en_core_web_lg (~600MB). Pin sm instead.
                _nlp_cfg = {
                    "nlp_engine_name": "spacy",
                    "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
                }
                _nlp_engine = NlpEngineProvider(nlp_configuration=_nlp_cfg).create_engine()
                self._analyzer = AnalyzerEngine(nlp_engine=_nlp_engine)
                self._anonymizer = AnonymizerEngine()
                logger.info("Presidio engines loaded (spaCy en_core_web_sm)")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Presidio unavailable (%s). Falling back to regex redaction.", exc
                )
                self._analyzer = None
                self._anonymizer = None

    @property
    def backend(self) -> str:
        return "presidio" if self._analyzer is not None else "regex"

    def redact(self, text: str, entities: Optional[List[str]] = None) -> RedactionResult:
        if not text or not text.strip():
            return RedactionResult(original=text, redacted=text, entities=[])

        entities = entities or _PRESIDIO_ENTITIES

        if self._analyzer is not None and self._anonymizer is not None:
            try:
                return self._redact_presidio(text, entities)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Presidio failed (%s); falling back to regex.", exc)

        return self._redact_regex(text)

    def _redact_presidio(self, text: str, entities: List[str]) -> RedactionResult:
        from presidio_anonymizer.entities import OperatorConfig

        results = self._analyzer.analyze(
            text=text, entities=entities, language=self._language
        )
        operators = {
            e: OperatorConfig("replace", {"new_value": f"[{e}]"}) for e in entities
        }
        anonymized = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )
        entity_dicts = [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": round(float(r.score), 3),
                "text": text[r.start : r.end],
            }
            for r in results
        ]
        return RedactionResult(
            original=text, redacted=anonymized.text, entities=entity_dicts
        )

    def _redact_regex(self, text: str) -> RedactionResult:
        redacted = text
        entity_dicts: List[dict] = []
        offset_map: List[tuple[int, int, str, str]] = []
        for entity_type, pattern in _REGEX_PATTERNS.items():
            for m in pattern.finditer(text):
                offset_map.append((m.start(), m.end(), entity_type, m.group(0)))

        offset_map.sort(key=lambda x: x[0])
        last_end = 0
        pieces: List[str] = []
        for start, end, entity_type, matched in offset_map:
            if start < last_end:
                continue
            pieces.append(text[last_end:start])
            pieces.append(f"[{entity_type}]")
            entity_dicts.append(
                {
                    "entity_type": entity_type,
                    "start": start,
                    "end": end,
                    "score": 0.85,
                    "text": matched,
                }
            )
            last_end = end
        pieces.append(text[last_end:])
        redacted = "".join(pieces)
        return RedactionResult(original=text, redacted=redacted, entities=entity_dicts)


_default_redactor: Optional[PIIRedactor] = None


def get_redactor() -> PIIRedactor:
    global _default_redactor
    if _default_redactor is None:
        _default_redactor = PIIRedactor()
    return _default_redactor
