"""Okapi BM25 over the chunk collection, implemented in plain Python.

A vector store would be the obvious production choice. BM25 keeps the sample app
dependency-free and fully deterministic, which matters more here: the harness is the
product, the RAG app is just something realistic to point it at.
"""

from __future__ import annotations

import math
from collections import Counter

from llm_eval.rag.corpus import Chunk
from llm_eval.text import content_terms


class BM25Retriever:
    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if not chunks:
            raise ValueError("BM25Retriever needs at least one chunk")
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self._term_freqs = [Counter(content_terms(c.text)) for c in chunks]
        self._lengths = [sum(tf.values()) for tf in self._term_freqs]
        self._avg_length = sum(self._lengths) / len(self._lengths)

        doc_freq: Counter[str] = Counter()
        for tf in self._term_freqs:
            doc_freq.update(tf.keys())
        n = len(chunks)
        self._idf = {
            term: math.log(1 + (n - df + 0.5) / (df + 0.5)) for term, df in doc_freq.items()
        }

    def score(self, query: str) -> list[float]:
        terms = content_terms(query)
        scores = []
        for tf, length in zip(self._term_freqs, self._lengths, strict=True):
            norm = self.k1 * (1 - self.b + self.b * length / self._avg_length)
            score = 0.0
            for term in terms:
                freq = tf.get(term, 0)
                if freq:
                    score += self._idf[term] * freq * (self.k1 + 1) / (freq + norm)
            scores.append(score)
        return scores

    def retrieve(self, query: str, top_k: int = 3) -> list[Chunk]:
        """Return up to ``top_k`` chunks with a positive score, best first."""
        scores = self.score(query)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [self.chunks[i] for i in ranked[:top_k] if scores[i] > 0]
