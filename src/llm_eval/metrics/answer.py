"""Answer-vs-reference metrics."""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from typing import Protocol

from llm_eval.text import content_terms, normalise


def exact_match(answer: str, reference: str) -> float:
    return float(normalise(answer) == normalise(reference))


def token_f1(answer: str, reference: str) -> float:
    """SQuAD-style token F1 over content terms."""
    predicted = Counter(content_terms(answer))
    expected = Counter(content_terms(reference))
    common = sum((predicted & expected).values())
    if not predicted or not expected or common == 0:
        return 0.0
    precision = common / sum(predicted.values())
    recall = common / sum(expected.values())
    return 2 * precision * recall / (precision + recall)


def fact_recall(answer: str, must_include: tuple[str, ...] | list[str]) -> float | None:
    """Share of required facts (short phrases such as '£10' or '28 days') present in the answer.

    This is the most useful single correctness signal for policy-style questions: it does not
    care about phrasing, only whether the facts a customer needs are there.
    """
    if not must_include:
        return None
    haystack = normalise(answer)
    found = sum(1 for fact in must_include if normalise(fact) in haystack)
    return found / len(must_include)


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class HashingEmbedder:
    """Deterministic bag-of-features embedding (stemmed words plus character 4-grams).

    It captures lexical and morphological similarity rather than true meaning, so treat the
    score as a cheap proxy. Anything with an ``embed`` method (for example a wrapper around
    a hosted embedding model) can be passed to ``semantic_similarity`` instead.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        self.dimensions = dimensions

    def _bucket(self, feature: str) -> int:
        return (
            int(hashlib.md5(feature.encode(), usedforsecurity=False).hexdigest(), 16)
            % self.dimensions
        )

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        terms = content_terms(text)
        for term in terms:
            vector[self._bucket(f"w:{term}")] += 1.0
        joined = " ".join(terms)
        for i in range(len(joined) - 3):
            vector[self._bucket(f"c:{joined[i : i + 4]}")] += 0.25
        return vector


_DEFAULT_EMBEDDER = HashingEmbedder()


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def semantic_similarity(answer: str, reference: str, embedder: Embedder | None = None) -> float:
    embedder = embedder or _DEFAULT_EMBEDDER
    return max(0.0, cosine(embedder.embed(answer), embedder.embed(reference)))
