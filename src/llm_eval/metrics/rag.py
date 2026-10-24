"""Retrieval and grounding metrics for RAG answers."""

from __future__ import annotations

import re

from llm_eval.rag.corpus import Chunk
from llm_eval.safety import is_refusal
from llm_eval.text import content_terms, split_sentences

_PLACEHOLDER_RE = re.compile(r"\[REDACTED_[A-Z_]+\]")


def faithfulness(
    answer: str, contexts: list[Chunk], support_threshold: float = 0.75
) -> float | None:
    """Share of answer sentences that are supported by the retrieved context.

    A sentence counts as supported when at least ``support_threshold`` of its content terms
    appear in the context. This is a lexical approximation of the claim-level checks that
    LLM-based groundedness scorers do; it is cheap, deterministic and catches the most common
    failure, which is the model adding facts that were never retrieved.
    Returns ``None`` for refusals, where there is nothing to ground.
    """
    if not answer.strip() or is_refusal(answer):
        return None
    context_terms = {t for chunk in contexts for t in content_terms(chunk.text)}
    sentences = [s for s in split_sentences(_PLACEHOLDER_RE.sub(" ", answer)) if content_terms(s)]
    if not sentences:
        return None
    supported = 0
    for sentence in sentences:
        terms = content_terms(sentence)
        coverage = sum(1 for t in terms if t in context_terms) / len(terms)
        supported += coverage >= support_threshold
    return supported / len(sentences)


def context_recall(contexts: list[Chunk], reference_docs: tuple[str, ...]) -> float | None:
    """Share of the reference documents that appear in the retrieved chunks."""
    if not reference_docs:
        return None
    retrieved = {c.doc_id for c in contexts}
    return sum(1 for doc in reference_docs if doc in retrieved) / len(reference_docs)


def context_precision(contexts: list[Chunk], reference_docs: tuple[str, ...]) -> float | None:
    """Share of retrieved chunks that come from a reference document."""
    if not reference_docs:
        return None
    if not contexts:
        return 0.0
    relevant = set(reference_docs)
    return sum(1 for c in contexts if c.doc_id in relevant) / len(contexts)
