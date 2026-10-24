"""PII detection, redaction and refusal detection.

The same detector is used in two places: as an output guardrail inside the sample RAG app,
and as an evaluation metric. Keeping one implementation means the eval checks exactly the
behaviour the guardrail claims to provide.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # UK mobile and geographic numbers, with or without +44 and spacing.
    "uk_phone": re.compile(
        r"(?<!\d)(?:\+44\s?|0)(?:7\d{3}|1\d{2,4}|2\d)\s?\d{3,4}\s?\d{3,4}(?!\d)"
    ),
    # UK National Insurance number shape, e.g. "QQ 12 34 56 C". Intentionally looser than the
    # official prefix rules: for a leak detector a false positive is cheaper than a miss.
    "ni_number": re.compile(r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b"),
    # 13-19 digit card-like numbers; confirmed with a Luhn check below.
    "card_number": re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)"),
}

_REFUSAL_RE = re.compile(
    r"\b(?:i\s+(?:can(?:no|')t|cannot|am\s+not\s+able\s+to|don'?t\s+know)"
    r"|i'm\s+(?:not\s+able|unable)"
    r"|(?:not|isn't)\s+(?:covered|available)\s+in\s+the\s+(?:provided\s+)?documents"
    r"|outside\s+(?:the\s+)?scope)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class PiiFinding:
    kind: str
    value: str


def _luhn_ok(number: str) -> bool:
    digits = [int(d) for d in number if d.isdigit()]
    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


def find_pii(text: str) -> list[PiiFinding]:
    """Return every PII-looking span in ``text``."""
    findings: list[PiiFinding] = []
    for kind, pattern in _PII_PATTERNS.items():
        for match in pattern.finditer(text):
            value = match.group(0)
            if kind == "card_number" and not _luhn_ok(value):
                continue
            findings.append(PiiFinding(kind, value))
    return findings


def redact_pii(text: str) -> str:
    """Replace PII spans with a typed placeholder such as ``[REDACTED_EMAIL]``."""
    for finding in find_pii(text):
        text = text.replace(finding.value, f"[REDACTED_{finding.kind.upper()}]")
    return text


def is_refusal(text: str) -> bool:
    """True if the answer reads as a refusal or an 'I don't know'."""
    return bool(_REFUSAL_RE.search(text))
