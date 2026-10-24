import json

import pytest

from llm_eval.runner import QUALITY_METRICS, cost_for, percentile, run_eval


@pytest.fixture(scope="module")
def baseline_result(root):
    from llm_eval.config import load_config

    return run_eval(load_config(root / "eval.yaml"))


def test_run_produces_complete_result(baseline_result):
    run, summary, cases = (
        baseline_result["run"],
        baseline_result["summary"],
        baseline_result["cases"],
    )
    assert run["provider"] == "mock"
    assert run["cases"] == len(cases) == 22
    for metric in (*QUALITY_METRICS, "pass_rate", "pii_leak_rate", "latency_p95_ms"):
        assert metric in summary
    json.dumps(baseline_result)  # must be serialisable as-is


def test_run_matches_committed_baseline(baseline_result, root):
    """Guards against metric code changes silently shifting the baseline.

    If a change is intentional, regenerate baselines/baseline.json (make baseline) and
    review the diff in the pull request.
    """
    committed = json.loads((root / "baselines" / "baseline.json").read_text())
    assert baseline_result["summary"] == committed["summary"]
    assert [c["answer"] for c in baseline_result["cases"]] == [
        c["answer"] for c in committed["cases"]
    ]


def test_refusal_and_safety_cases(baseline_result):
    cases = {c["id"]: c for c in baseline_result["cases"]}
    assert cases["oos-weather"]["refused"]
    assert cases["oos-weather"]["passed"]
    probe = cases["pii-probe-contact"]
    assert probe["pii_found"] == []
    assert "[REDACTED_EMAIL]" in probe["answer"]
    # Metrics that only make sense for answerable questions are skipped, not zeroed.
    assert probe["metrics"]["fact_recall"] is None


def test_failures_explain_themselves(baseline_result):
    failing = [c for c in baseline_result["cases"] if not c["passed"]]
    assert failing, "the extractive mock has known weaknesses; the baseline should show them"
    for case in failing:
        assert case["failures"]


def test_degraded_candidate_leaks_pii_and_hallucinates(degraded_config):
    result = run_eval(degraded_config)
    cases = {c["id"]: c for c in result["cases"]}
    assert cases["pii-probe-contact"]["pii_found"] == ["email", "uk_phone"]
    assert any("leaked PII" in f for f in cases["pii-probe-contact"]["failures"])
    assert result["summary"]["faithfulness"] < 1.0


def test_percentile():
    assert percentile([], 95) == 0.0
    assert percentile([10.0], 95) == 10.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == 2.5
    assert percentile([1.0, 2.0, 3.0, 4.0], 100) == 4.0


def test_cost_for():
    pricing = {"m": {"input_per_1k": 1.0, "output_per_1k": 2.0}}
    assert cost_for("m", 1000, 500, pricing) == pytest.approx(2.0)
    assert cost_for("unknown", 1000, 500, pricing) == 0.0
