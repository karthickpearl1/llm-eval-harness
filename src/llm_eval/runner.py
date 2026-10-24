"""Run a golden dataset through the RAG app and score every case."""

from __future__ import annotations

import statistics
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from llm_eval import __version__
from llm_eval.config import EvalConfig
from llm_eval.dataset import GoldenCase, load_dataset
from llm_eval.metrics import (
    context_precision,
    context_recall,
    create_judge,
    exact_match,
    fact_recall,
    faithfulness,
    semantic_similarity,
    token_f1,
)
from llm_eval.metrics.judge import Judge
from llm_eval.providers import create_provider
from llm_eval.rag import BM25Retriever, RagApp, RagResponse, load_corpus
from llm_eval.safety import find_pii, is_refusal

SCHEMA_VERSION = 1

# Order matters: it is the column order in every report.
QUALITY_METRICS = (
    "exact_match",
    "token_f1",
    "semantic_similarity",
    "fact_recall",
    "faithfulness",
    "context_recall",
    "context_precision",
    "judge_correctness",
    "judge_relevance",
)


@dataclass
class CaseResult:
    id: str
    question: str
    expected_behaviour: str
    tags: list[str]
    answer: str
    retrieved: list[str]
    metrics: dict[str, float | None]
    refused: bool
    refusal_correct: bool
    pii_found: list[str]
    latency_ms: float
    retrieval_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    judge_reason: str = ""
    failures: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.failures

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "passed": self.passed}


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile; ``pct`` is 0-100."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def cost_for(model: str, input_tokens: int, output_tokens: int, pricing: dict[str, Any]) -> float:
    price = pricing.get(model)
    if not price:
        return 0.0
    return input_tokens / 1000 * float(price.get("input_per_1k", 0)) + output_tokens / 1000 * float(
        price.get("output_per_1k", 0)
    )


def score_case(
    case: GoldenCase,
    response: RagResponse,
    judge: Judge,
    config: EvalConfig,
) -> CaseResult:
    answer = response.answer
    refused = is_refusal(answer)
    if case.expected_behaviour == "answer":
        refusal_correct = not refused
    elif case.expected_behaviour == "refuse":
        refusal_correct = refused
    else:
        refusal_correct = True

    metrics: dict[str, float | None] = dict.fromkeys(QUALITY_METRICS)
    judge_reason = ""
    if case.expected_behaviour == "answer":
        verdict = judge.score(case, answer)
        judge_reason = verdict.reason
        metrics.update(
            exact_match=exact_match(answer, case.reference_answer),
            token_f1=token_f1(answer, case.reference_answer),
            semantic_similarity=semantic_similarity(answer, case.reference_answer),
            fact_recall=fact_recall(answer, case.must_include),
            faithfulness=faithfulness(answer, response.contexts),
            context_recall=context_recall(response.contexts, case.reference_contexts),
            context_precision=context_precision(response.contexts, case.reference_contexts),
            judge_correctness=verdict.correctness,
            judge_relevance=verdict.relevance,
        )
    metrics = {k: (round(v, 4) if v is not None else None) for k, v in metrics.items()}

    pii = sorted({f.kind for f in find_pii(answer)})
    completion = response.completion

    failures: list[str] = []
    if not refusal_correct:
        failures.append(
            "refused a question it should answer"
            if case.expected_behaviour == "answer"
            else "answered a question it should refuse"
        )
    if pii:
        failures.append(f"leaked PII: {', '.join(pii)}")
    for metric, minimum in config.case_thresholds.items():
        value = metrics.get(metric)
        if value is not None and value < minimum:
            failures.append(f"{metric} {value:.2f} < {minimum:.2f}")

    return CaseResult(
        id=case.id,
        question=case.question,
        expected_behaviour=case.expected_behaviour,
        tags=list(case.tags),
        answer=answer,
        retrieved=[c.chunk_id for c in response.contexts],
        metrics=metrics,
        refused=refused,
        refusal_correct=refusal_correct,
        pii_found=pii,
        latency_ms=completion.latency_ms,
        retrieval_ms=response.retrieval_ms,
        input_tokens=completion.input_tokens,
        output_tokens=completion.output_tokens,
        cost_usd=round(
            cost_for(
                completion.model, completion.input_tokens, completion.output_tokens, config.pricing
            ),
            8,
        ),
        judge_reason=judge_reason,
        failures=failures,
    )


def summarise(cases: list[CaseResult]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for metric in QUALITY_METRICS:
        values = [c.metrics[metric] for c in cases if c.metrics.get(metric) is not None]
        if values:
            summary[metric] = round(statistics.fmean(values), 4)

    latencies = [c.latency_ms for c in cases]
    summary.update(
        pass_rate=round(sum(c.passed for c in cases) / len(cases), 4),
        refusal_accuracy=round(sum(c.refusal_correct for c in cases) / len(cases), 4),
        pii_leak_rate=round(sum(bool(c.pii_found) for c in cases) / len(cases), 4),
        latency_p50_ms=round(percentile(latencies, 50), 1),
        latency_p95_ms=round(percentile(latencies, 95), 1),
        total_tokens=sum(c.input_tokens + c.output_tokens for c in cases),
        total_cost_usd=round(sum(c.cost_usd for c in cases), 6),
    )
    return summary


def build_app(config: EvalConfig) -> RagApp:
    provider = create_provider(
        config.provider.name, config.provider.model, **config.provider.options
    )
    retriever = BM25Retriever(load_corpus(config.corpus, config.paragraphs_per_chunk))
    return RagApp(provider, retriever, top_k=config.top_k, redact_pii=config.redact_pii)


def run_eval(
    config: EvalConfig, *, app: RagApp | None = None, judge: Judge | None = None
) -> dict[str, Any]:
    """Run every case and return a JSON-serialisable result document."""
    cases = load_dataset(config.dataset)
    app = app or build_app(config)
    judge = judge or create_judge(config.judge.name, config.judge.model)

    started = time.perf_counter()
    results = [score_case(case, app.ask(case.question), judge, config) for case in cases]
    duration = time.perf_counter() - started

    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "name": config.name,
            "config": str(config.source) if config.source else None,
            "provider": app.provider.name,
            "model": app.provider.model,
            "judge": getattr(judge, "name", type(judge).__name__),
            "top_k": config.top_k,
            "redact_pii": config.redact_pii,
            "dataset": config.dataset.name,
            "cases": len(results),
            "started_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "duration_s": round(duration, 3),
            "harness_version": __version__,
        },
        "summary": summarise(results),
        "cases": [r.to_dict() for r in results],
    }
