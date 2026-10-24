"""Small, dependency-free text helpers shared by the retriever, the mock model and the metrics.

These are deliberately simple. They need to be deterministic and fast so the whole suite
can run offline in CI; they are not trying to be a production NLP pipeline.
"""

from __future__ import annotations

import re
import unicodedata

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.:][0-9]+)?")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9£\"'(])")

_STOPWORD_TEXT = """
a about above after again all also am an and any are as at be because been before being
below between both but by can could did do does doing down during each few for from further
had has have having he her here hers him his how i if in into is it its itself just me more
most my no nor not of off on once only or other our out over own same she should so some such
than that the their them then there these they this those through to too under until up very
was we were what when where which while who whom why will with would you your yours
"""
STOPWORDS = frozenset(_STOPWORD_TEXT.split())


def normalise(text: str) -> str:
    """Lower-case, strip accents and punctuation, and collapse whitespace."""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower()
    text = re.sub(r"[^a-z0-9£%\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def stem(token: str) -> str:
    """A very light suffix stripper; enough to match 'refunds' with 'refund'."""
    if len(token) <= 4 or token.isdigit():
        return token
    for suffix in ("ing", "ies", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)] + ("y" if suffix == "ies" else "")
            break
    # Drop a trailing "e" so that "charge", "charges" and "charged" all meet at "charg".
    return token[:-1] if token.endswith("e") and len(token) > 4 else token


def tokens(text: str) -> list[str]:
    """All word tokens, lower-cased and unstemmed."""
    return _TOKEN_RE.findall(text.lower())


def content_terms(text: str) -> list[str]:
    """Stemmed tokens with stopwords removed; the unit most metrics compare on."""
    return [stem(t) for t in tokens(text) if t not in STOPWORDS]


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences. Good enough for short policy-style documents."""
    parts: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        block = " ".join(line.strip() for line in block.splitlines() if line.strip())
        if not block or block.startswith("#"):
            continue
        parts.extend(s.strip() for s in _SENTENCE_RE.split(block) if s.strip())
    return parts
