"""Offline, deterministic stand-in for an LLM.

The mock behaves like a cautious extractive model: it reads the ``<context>`` and
``<question>`` blocks from the prompt, returns the context sentences that best overlap with
the question, and refuses when nothing overlaps well enough. That is enough to exercise
every metric in the harness without network access or API keys, and it always gives the
same output for the same input, which is what a regression baseline needs.

A few knobs let you simulate a worse model so the regression gate can be demonstrated:
``hallucination_rate`` appends an unsupported claim to a deterministic subset of answers,
``max_sentences`` controls verbosity and ``extra_latency_ms`` simulates a slower endpoint.
"""

from __future__ import annotations

import hashlib
import re

from llm_eval.providers.base import Completion, LLMProvider, estimate_tokens
from llm_eval.text import content_terms, split_sentences

REFUSAL_TEXT = "I can't answer that from the provided documents."

# Plausible-sounding claims that are not supported by the sample corpus.
_FABRICATIONS = (
    "Passengers can also claim a free annual pass by writing to the managing director.",
    "This policy was introduced nationally by parliament in 1998.",
    "Weekend journeys are exempt and are handled by a separate partner company.",
    "Customers with a loyalty card receive double the stated amount.",
)

_BLOCK_RE = re.compile(r"<(context|question)>\s*(.*?)\s*</\1>", re.DOTALL | re.IGNORECASE)


def _unit_hash(*parts: str) -> float:
    """Map strings to a stable float in [0, 1)."""
    digest = hashlib.sha256("\x1f".join(parts).encode()).hexdigest()
    return int(digest[:12], 16) / float(16**12)


class MockProvider(LLMProvider):
    name = "mock"

    def __init__(
        self,
        model: str = "mock-extractive-v1",
        *,
        refusal_threshold: float = 0.4,
        max_sentences: int = 1,
        hallucination_rate: float = 0.0,
        extra_latency_ms: float = 0.0,
        seed: str = "0",
    ) -> None:
        super().__init__(model)
        if not 0.0 <= hallucination_rate <= 1.0:
            raise ValueError("hallucination_rate must be between 0 and 1")
        self.refusal_threshold = refusal_threshold
        self.max_sentences = max_sentences
        self.hallucination_rate = hallucination_rate
        self.extra_latency_ms = extra_latency_ms
        self.seed = seed

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        blocks = {name.lower(): body for name, body in _BLOCK_RE.findall(prompt)}
        answer = self._answer(blocks.get("question", ""), blocks.get("context", ""))

        input_tokens = estimate_tokens((system or "") + prompt)
        output_tokens = estimate_tokens(answer)
        # Simulated latency: a fixed overhead plus per-token costs, so it scales with prompt
        # size and answer length like a real endpoint but stays reproducible.
        latency_ms = 120.0 + 0.05 * input_tokens + 4.0 * output_tokens + self.extra_latency_ms
        return Completion(answer, self.model, input_tokens, output_tokens, round(latency_ms, 1))

    def _answer(self, question: str, context: str) -> str:
        question_terms = set(content_terms(question))
        if not question_terms or not context.strip():
            return REFUSAL_TEXT

        scored: list[tuple[float, int, str]] = []
        for position, sentence in enumerate(split_sentences(context)):
            overlap = question_terms & set(content_terms(sentence))
            score = len(overlap) / len(question_terms)
            scored.append((score, -position, sentence))
        scored.sort(reverse=True)

        if not scored or scored[0][0] < self.refusal_threshold:
            return REFUSAL_TEXT

        chosen = [s for score, _, s in scored[: self.max_sentences] if score > 0]
        answer = " ".join(chosen)

        if _unit_hash(self.seed, question) < self.hallucination_rate:
            index = int(_unit_hash("fabrication", question) * len(_FABRICATIONS))
            answer = f"{answer} {_FABRICATIONS[index]}"
        return answer
