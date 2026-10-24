"""LLM-as-judge scoring.

``LLMJudge`` sends a rubric prompt to any provider and parses a JSON verdict. ``MockJudge``
applies the same rubric with deterministic heuristics, so the pipeline (prompting, parsing,
normalising, reporting) is exercised offline and the judge scores in the baseline are
reproducible. Swap in a real judge with ``LLM_EVAL_JUDGE_PROVIDER``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol

from llm_eval.dataset import GoldenCase
from llm_eval.metrics.answer import fact_recall, semantic_similarity
from llm_eval.providers import LLMProvider, ProviderError, create_provider
from llm_eval.safety import is_refusal
from llm_eval.text import content_terms

JUDGE_SYSTEM = "You are a strict evaluator of customer support answers. Reply with JSON only."

JUDGE_PROMPT = """Score the ANSWER to the QUESTION against the REFERENCE on a 1-5 scale.

correctness: 5 = all facts in the reference are present and nothing contradicts it,
             1 = wrong or missing the key facts.
relevance:   5 = directly addresses the question, 1 = off-topic.

QUESTION: {question}
REFERENCE: {reference}
ANSWER: {answer}

Return JSON: {{"correctness": <1-5>, "relevance": <1-5>, "reason": "<one sentence>"}}"""


@dataclass(frozen=True)
class JudgeScore:
    correctness: float  # normalised to [0, 1]
    relevance: float  # normalised to [0, 1]
    reason: str


class Judge(Protocol):
    def score(self, case: GoldenCase, answer: str) -> JudgeScore: ...


def _normalise(score: float) -> float:
    """Map a 1-5 rubric score to [0, 1], clamping out-of-range values."""
    return round((min(max(score, 1.0), 5.0) - 1.0) / 4.0, 4)


class MockJudge:
    name = "mock"

    def score(self, case: GoldenCase, answer: str) -> JudgeScore:
        if is_refusal(answer):
            return JudgeScore(0.0, 0.0, "Answer is a refusal.")

        facts = fact_recall(answer, case.must_include)
        similarity = semantic_similarity(answer, case.reference_answer)
        blended = similarity if facts is None else 0.7 * facts + 0.3 * similarity
        correctness = round(1 + 4 * blended)

        question_terms = set(content_terms(case.question))
        overlap = len(question_terms & set(content_terms(answer))) / max(1, len(question_terms))
        relevance = round(1 + 4 * min(1.0, overlap / 0.6))

        reason = f"facts={facts if facts is not None else 'n/a'}, similarity={similarity:.2f}"
        return JudgeScore(_normalise(correctness), _normalise(relevance), reason)


class LLMJudge:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        self.name = provider.name

    def score(self, case: GoldenCase, answer: str) -> JudgeScore:
        prompt = JUDGE_PROMPT.format(
            question=case.question, reference=case.reference_answer or "(none)", answer=answer
        )
        completion = self.provider.complete(prompt, system=JUDGE_SYSTEM, max_tokens=200)
        return parse_verdict(completion.text)


def parse_verdict(text: str) -> JudgeScore:
    """Parse the judge's JSON, tolerating code fences or prose around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ProviderError(f"Judge did not return JSON: {text[:200]!r}")
    try:
        data = json.loads(match.group(0))
        return JudgeScore(
            _normalise(float(data["correctness"])),
            _normalise(float(data["relevance"])),
            str(data.get("reason", "")),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ProviderError(f"Could not parse judge verdict: {text[:200]!r}") from exc


def create_judge(name: str, model: str | None = None) -> Judge:
    if name == "mock":
        return MockJudge()
    return LLMJudge(create_provider(name, model))
