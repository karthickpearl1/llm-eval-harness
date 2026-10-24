import json

import pytest

from llm_eval.config import load_config
from llm_eval.gate import evaluate_gate
from llm_eval.report import fmt, fmt_delta, render_html, render_markdown, write_reports
from llm_eval.runner import run_eval


@pytest.fixture(scope="module")
def runs(root):
    baseline = run_eval(load_config(root / "eval.yaml"))
    degraded_config = load_config(root / "configs" / "candidate-degraded.yaml")
    degraded = run_eval(degraded_config)
    gate = evaluate_gate(degraded, degraded_config.gate, baseline)
    return baseline, degraded, gate


def test_markdown_single_run(runs):
    markdown = render_markdown(runs[0])
    assert markdown.startswith("# Eval report: baseline")
    assert "| pass_rate |" in markdown
    assert "Failing cases" in markdown


def test_markdown_comparison_shows_gate_and_deltas(runs):
    baseline, degraded, gate = runs
    markdown = render_markdown(degraded, gate, baseline)
    assert "**Quality gate: FAILED**" in markdown
    assert "| Metric | Baseline | Candidate | Change |" in markdown
    assert "newly failing cases" in markdown
    assert "▼" in markdown


def test_html_report_is_escaped_and_complete(runs):
    baseline, degraded, gate = runs
    html = render_html(degraded, gate, baseline)
    assert "Quality gate FAILED" in html
    assert "pii-probe-contact" in html
    assert "Compared with baseline" in html
    assert "<script" not in html


def test_write_reports(tmp_path, runs):
    baseline, degraded, gate = runs
    paths = write_reports(degraded, tmp_path / "out", gate, baseline)
    assert set(paths) == {"json", "markdown", "html", "gate"}
    assert json.loads(paths["gate"].read_text())["passed"] is False
    assert json.loads(paths["json"].read_text())["run"]["name"] == "candidate-degraded"


@pytest.mark.parametrize(
    ("value", "kind", "expected"),
    [
        (None, "ratio", "-"),
        (0.5, "pct", "50.0%"),
        (0.12345, "ratio", "0.123"),
        (1234.4, "ms", "1,234 ms"),
        (4137, "int", "4,137"),
        (0.00084, "usd", "$0.0008"),
    ],
)
def test_fmt(value, kind, expected):
    assert fmt(value, kind) == expected


@pytest.mark.parametrize(
    ("delta", "kind", "expected"),
    [
        (-0.273, "pct", "-27.3 pp"),
        (210.0, "ms", "+210 ms"),
        (-849, "int", "-849"),
        (0.000067, "usd", "+0.000067"),
        (0.157, "ratio", "+0.157"),
    ],
)
def test_fmt_delta(delta, kind, expected):
    assert fmt_delta(delta, kind) == expected
