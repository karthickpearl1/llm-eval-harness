import copy

import pytest

from llm_eval.config import GateConfig
from llm_eval.gate import evaluate_gate


def result(cases=None, **summary):
    return {"summary": summary, "cases": cases or []}


def test_absolute_bounds():
    gate = GateConfig(min={"faithfulness": 0.9}, max={"pii_leak_rate": 0.0})
    assert evaluate_gate(result(faithfulness=0.95, pii_leak_rate=0.0), gate).passed
    failed = evaluate_gate(result(faithfulness=0.80, pii_leak_rate=0.1), gate)
    assert not failed.passed
    assert {c.metric for c in failed.checks if not c.passed} == {"faithfulness", "pii_leak_rate"}


def test_missing_metric_fails_closed():
    gate = GateConfig(min={"faithfulness": 0.9})
    check = evaluate_gate(result(), gate).checks[0]
    assert not check.passed
    assert "missing" in check.message


def test_max_drop_is_relative_to_baseline():
    gate = GateConfig(max_drop={"fact_recall": 0.05})
    baseline = result(fact_recall=0.90)
    assert evaluate_gate(result(fact_recall=0.86), gate, baseline).passed
    assert evaluate_gate(result(fact_recall=0.85), gate, baseline).passed  # boundary is allowed
    assert not evaluate_gate(result(fact_recall=0.84), gate, baseline).passed
    assert not evaluate_gate(result(), gate, baseline).passed


@pytest.mark.parametrize(
    ("base", "candidate", "passed"),
    [(100.0, 120.0, True), (100.0, 130.0, False), (0.0, 0.0, True), (0.0, 1.0, False)],
)
def test_max_increase_pct(base, candidate, passed):
    gate = GateConfig(max_increase_pct={"latency_p95_ms": 25})
    outcome = evaluate_gate(result(latency_p95_ms=candidate), gate, result(latency_p95_ms=base))
    assert outcome.passed is passed


def test_max_increase_pct_missing_metric():
    gate = GateConfig(max_increase_pct={"latency_p95_ms": 25})
    assert not evaluate_gate(result(), gate, result()).passed


def test_new_and_fixed_cases_are_tracked():
    baseline = result(cases=[{"id": "a", "passed": True}, {"id": "b", "passed": False}])
    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["passed"] = False
    candidate["cases"][1]["passed"] = True

    outcome = evaluate_gate(candidate, GateConfig(max_new_failures=0), baseline)
    assert outcome.new_failures == ["a"]
    assert outcome.fixed_cases == ["b"]
    assert not outcome.passed

    tolerant = evaluate_gate(candidate, GateConfig(max_new_failures=1), baseline)
    assert tolerant.passed
    assert tolerant.to_dict()["passed"] is True


def test_no_baseline_means_no_case_limit():
    outcome = evaluate_gate(result(), GateConfig(max_new_failures=0))
    assert outcome.max_new_failures is None
    assert outcome.passed
