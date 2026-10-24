"""Command-line interface.

Exit codes: 0 = success / gate passed, 1 = quality gate failed, 2 = usage or configuration error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from llm_eval import __version__
from llm_eval.config import load_config
from llm_eval.gate import GateResult, evaluate_gate
from llm_eval.providers import ProviderError
from llm_eval.report import render_markdown, summary_table, write_reports
from llm_eval.runner import run_eval

EXIT_OK, EXIT_GATE_FAILED, EXIT_ERROR = 0, 1, 2


def _colour(text: str, code: str) -> str:
    """Wrap text in an ANSI colour when writing to a terminal (respects NO_COLOR)."""
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _status(passed: bool) -> str:
    return _colour("PASS", "32") if passed else _colour("FAIL", "31")


def _load_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data or "cases" not in data:
        raise ValueError(f"{path} does not look like an llm-eval results file")
    return data


def _print_summary(result: dict[str, Any], baseline: dict[str, Any] | None = None) -> None:
    rows = summary_table(result, baseline)
    width = max(len(r["metric"]) for r in rows)
    if baseline is None:
        for r in rows:
            print(f"{r['metric']:<{width}}  {r['value']:>10}")
        return

    markers = {"better": _colour("  ▲", "32"), "worse": _colour("  ▼", "31")}
    print(f"{'metric':<{width}}  {'baseline':>10}  {'candidate':>10}  change")
    for r in rows:
        values = f"{r['baseline']:>10}  {r['value']:>10}"
        print(f"{r['metric']:<{width}}  {values}  {r['delta']}{markers.get(r['trend'], '')}")


def _print_gate(gate: GateResult) -> None:
    print()
    for check in gate.checks:
        print(f"  [{_status(check.passed)}] {check.rule:<16} {check.message}")
    if gate.new_failures:
        print(f"  [{_status(False)}] newly failing cases: {', '.join(gate.new_failures)}")
    if gate.fixed_cases:
        print(f"  [INFO] newly passing cases: {', '.join(gate.fixed_cases)}")
    verdict = _colour("PASSED", "1;32") if gate.passed else _colour("FAILED", "1;31")
    print(f"\nQuality gate: {verdict}")


def cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_eval(config)
    run = result["run"]
    print(
        f"Ran {run['cases']} cases with {run['provider']}/{run['model']} in {run['duration_s']}s\n"
    )
    _print_summary(result)

    gate = evaluate_gate(result, config.gate) if args.gate else None
    paths = write_reports(result, Path(args.out), gate)
    if gate is not None:
        _print_gate(gate)
    print(f"\nReports: {', '.join(str(p) for p in paths.values())}")
    return EXIT_GATE_FAILED if gate is not None and not gate.passed else EXIT_OK


def cmd_compare(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    baseline = _load_result(Path(args.baseline))
    candidate = _load_result(Path(args.candidate))

    gate = evaluate_gate(candidate, config.gate, baseline)
    print(f"Baseline:  {baseline['run']['name']} ({baseline['run']['model']})")
    print(f"Candidate: {candidate['run']['name']} ({candidate['run']['model']})\n")
    _print_summary(candidate, baseline)
    _print_gate(gate)

    paths = write_reports(candidate, Path(args.out), gate, baseline)
    if args.summary_md:
        with open(args.summary_md, "a", encoding="utf-8") as handle:
            handle.write(render_markdown(candidate, gate, baseline))
    print(f"\nReports: {', '.join(str(p) for p in paths.values())}")
    return EXIT_OK if gate.passed else EXIT_GATE_FAILED


def cmd_report(args: argparse.Namespace) -> int:
    result = _load_result(Path(args.results))
    paths = write_reports(result, Path(args.out))
    print(f"Reports: {', '.join(str(p) for p in paths.values())}")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval", description="Evaluate and regression-test LLM and RAG applications."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the golden dataset and write reports")
    run.add_argument("-c", "--config", default="eval.yaml", help="eval config (default: eval.yaml)")
    run.add_argument("-o", "--out", default="reports/latest", help="output directory")
    run.add_argument(
        "--gate", action="store_true", help="fail if absolute gate thresholds are not met"
    )
    run.set_defaults(func=cmd_run)

    compare = sub.add_parser("compare", help="compare a candidate run against a baseline")
    compare.add_argument("baseline", help="baseline results.json")
    compare.add_argument("candidate", help="candidate results.json")
    compare.add_argument(
        "-c", "--config", default="eval.yaml", help="config holding the gate thresholds"
    )
    compare.add_argument("-o", "--out", default="reports/compare", help="output directory")
    compare.add_argument(
        "--summary-md", help="append a Markdown summary to this file (e.g. $GITHUB_STEP_SUMMARY)"
    )
    compare.set_defaults(func=cmd_compare)

    report = sub.add_parser("report", help="re-render reports from a results.json")
    report.add_argument("results", help="results.json from a previous run")
    report.add_argument("-o", "--out", default="reports/rendered", help="output directory")
    report.set_defaults(func=cmd_report)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError, ProviderError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
