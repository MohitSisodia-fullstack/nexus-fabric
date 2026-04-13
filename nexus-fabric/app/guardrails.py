# app/guardrails.py
# PII Detection Guardrail using Microsoft Presidio
# Scans text BEFORE sending to LLM — prevents leakage of sensitive data.

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("nexus-guardrails")

try:
    from presidio_analyzer import AnalyzerEngine
    from presidio_anonymizer import AnonymizerEngine
    from presidio_anonymizer.entities import OperatorConfig
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False
    logger.warning("presidio not installed — using regex fallback guardrail")


@dataclass
class GuardrailResult:
    safe: bool
    original_text: str
    sanitized_text: str
    pii_found: list[str]
    blocked: bool = False
    block_reason: str = ""


# Entities to detect and anonymize
SENSITIVE_ENTITIES = [
    "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "CREDIT_CARD",
    "IBAN_CODE", "IP_ADDRESS", "LOCATION", "DATE_TIME",
    "NRP",          # National Registry Patterns (SSN, passport numbers)
    "MEDICAL_LICENSE",
    "URL",
]

# Entities severe enough to BLOCK the entire request (not just redact)
BLOCK_ENTITIES = {"CREDIT_CARD", "IBAN_CODE", "NRP", "MEDICAL_LICENSE"}

# Minimum confidence threshold
CONFIDENCE_THRESHOLD = 0.7


class PIIGuardrail:
    """
    Two-mode PII protection:
    1. Presidio (preferred) — ML + rule-based NER
    2. Regex fallback — catches obvious patterns if Presidio not installed
    """

    def __init__(self):
        if PRESIDIO_AVAILABLE:
            self._analyzer  = AnalyzerEngine()
            self._anonymizer = AnonymizerEngine()
            logger.info("PIIGuardrail: using Presidio engine")
        else:
            self._analyzer  = None
            self._anonymizer = None
            logger.info("PIIGuardrail: using regex fallback")

        # Regex fallback patterns
        self._patterns = {
            "EMAIL":       re.compile(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'),
            "PHONE":       re.compile(r'(\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}'),
            "SSN":         re.compile(r'\b\d{3}-\d{2}-\d{4}\b'),
            "CREDIT_CARD": re.compile(r'\b(?:\d{4}[-\s]?){3}\d{4}\b'),
            "IP_ADDRESS":  re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'),
        }

    def scan(self, text: str) -> GuardrailResult:
        """
        Scan text for PII. Returns sanitized text and metadata.
        If BLOCK_ENTITIES are found, sets blocked=True — caller should abort the request.
        """
        if not text or not text.strip():
            return GuardrailResult(safe=True, original_text=text,
                                   sanitized_text=text, pii_found=[])

        if PRESIDIO_AVAILABLE:
            return self._presidio_scan(text)
        else:
            return self._regex_scan(text)

    def _presidio_scan(self, text: str) -> GuardrailResult:
        results = self._analyzer.analyze(
            text=text,
            entities=SENSITIVE_ENTITIES,
            language="en",
            score_threshold=CONFIDENCE_THRESHOLD,
        )

        if not results:
            return GuardrailResult(safe=True, original_text=text,
                                   sanitized_text=text, pii_found=[])

        found_types = list({r.entity_type for r in results})
        blocked     = bool(set(found_types) & BLOCK_ENTITIES)

        if blocked:
            block_type = list(set(found_types) & BLOCK_ENTITIES)[0]
            return GuardrailResult(
                safe=False, original_text=text, sanitized_text="[BLOCKED]",
                pii_found=found_types, blocked=True,
                block_reason=f"Highly sensitive PII detected: {block_type}"
            )

        # Anonymize non-blocking PII
        operators = {
            entity: OperatorConfig("replace", {"new_value": f"<{entity}>"})
            for entity in SENSITIVE_ENTITIES
        }
        anonymized = self._anonymizer.anonymize(
            text=text, analyzer_results=results, operators=operators
        )

        return GuardrailResult(
            safe=True,
            original_text=text,
            sanitized_text=anonymized.text,
            pii_found=found_types,
        )

    def _regex_scan(self, text: str) -> GuardrailResult:
        found   = []
        cleaned = text

        for name, pattern in self._patterns.items():
            matches = pattern.findall(cleaned)
            if matches:
                found.append(name)
                if name in {"CREDIT_CARD", "SSN"}:
                    return GuardrailResult(
                        safe=False, original_text=text, sanitized_text="[BLOCKED]",
                        pii_found=found, blocked=True,
                        block_reason=f"Sensitive PII detected: {name}"
                    )
                cleaned = pattern.sub(f"<{name}>", cleaned)

        return GuardrailResult(
            safe=True, original_text=text,
            sanitized_text=cleaned, pii_found=found
        )


# Singleton for app-wide use
_guardrail: Optional[PIIGuardrail] = None

def get_guardrail() -> PIIGuardrail:
    global _guardrail
    if _guardrail is None:
        _guardrail = PIIGuardrail()
    return _guardrail


def scan_before_llm(text: str) -> GuardrailResult:
    """Convenience wrapper — call this before every LLM request."""
    result = get_guardrail().scan(text)
    if result.blocked:
        logger.warning(f"BLOCKED request — reason: {result.block_reason}")
    elif result.pii_found:
        logger.info(f"PII redacted: {result.pii_found}")
    return result
