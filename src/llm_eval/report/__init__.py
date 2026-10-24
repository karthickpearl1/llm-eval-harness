"""JSON, Markdown and HTML reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, PackageLoader, select_autoescape

from llm_eval.gate import GateResult
from llm_eval.runner import QUALITY_METRICS

# Summary rows shown in reports, with how to format them and which direction is better.
SUMMARY_ROWS: tuple[tuple[str, str, str], ...] = (
    ("pass_rate", "pct", "up"),
    *((m, "ratio", "up") for m in QUALITY_METRICS),
    ("refusal_accuracy", "pct", "up"),
    ("pii_leak_rate", "pct", "down"),
    ("latency_p50_ms", "ms", "down"),
    ("latency_p95_ms", "ms", "down"),
    ("total_tokens", "int", "down"),
    ("total_cost_usd", "usd", "down"),
)


def fmt(value: Any, kind: str) -> str:
    if value is None:
        return "-"
    if kind == "pct":
        return f"{value * 100:.1f}%"
    if kind == "ms":
        return f"{value:,.0f} ms"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "usd":
        return f"${value:.4f}"
    return f"{value:.3f}"


def fmt_delta(delta: float, kind: str) -> str:
    if kind == "pct":
        return f"{delta * 100:+.1f} pp"
    if kind == "ms":
        return f"{delta:+,.0f} ms"
    if kind == "int":
        return f"{int(delta):+,}"
    if kind == "usd":
        return f"{delta:+.6f}"
    return f"{delta:+.3f}"


def summary_table(
    result: dict[str, Any], baseline: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """Rows of (metric, value, baseline, delta, verdict) used by both Markdown and HTML."""
    rows = []
    for metric, kind, better in SUMMARY_ROWS:
        value = result["summary"].get(metric)
        if value is None:
            continue
        row: dict[str, Any] = {"metric": metric, "value": fmt(value, kind), "trend": ""}
        if baseline is not None:
            base = baseline["summary"].get(metric)
            row["baseline"] = fmt(base, kind)
            if base is not None and value != base:
                delta = value - base
                row["delta"] = fmt_delta(delta, kind)
                improved = (delta > 0) == (better == "up")
                row["trend"] = "better" if improved else "worse"
            else:
                row["delta"] = "0"
        rows.append(row)
    return rows


def write_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def render_markdown(
    result: dict[str, Any],
    gate: GateResult | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    run = result["run"]
    lines = [
        f"# Eval report: {run['name']}",
        "",
        f"Provider `{run['provider']}` / model `{run['model']}` / judge `{run['judge']}` / "
        f"{run['cases']} cases from `{run['dataset']}`",
        "",
    ]
    if gate is not None:
        lines += [f"**Quality gate: {'PASSED' if gate.passed else 'FAILED'}**", ""]
        failed = [c for c in gate.checks if not c.passed]
        lines += [f"- :x: {c.message}" for c in failed]
        if gate.new_failures:
            lines.append(f"- :x: newly failing cases: {', '.join(gate.new_failures)}")
        if gate.fixed_cases:
            lines.append(f"- :white_check_mark: newly passing cases: {', '.join(gate.fixed_cases)}")
        lines.append("")

    rows = summary_table(result, baseline)
    if baseline is not None:
        lines += ["| Metric | Baseline | Candidate | Change |", "|---|---:|---:|---:|"]
        for r in rows:
            marker = {"better": " ▲", "worse": " ▼"}.get(r["trend"], "")
            lines.append(
                f"| {r['metric']} | {r['baseline']} | {r['value']} | {r['delta']}{marker} |"
            )
    else:
        lines += ["| Metric | Value |", "|---|---:|"]
        lines += [f"| {r['metric']} | {r['value']} |" for r in rows]

    failing = [c for c in result["cases"] if not c["passed"]]
    lines += ["", f"## Failing cases ({len(failing)})", ""]
    if not failing:
        lines.append("None.")
    for case in failing:
        lines.append(f"- **{case['id']}**: {'; '.join(case['failures'])}")
    return "\n".join(lines) + "\n"


_env = Environment(
    loader=PackageLoader("llm_eval.report", "templates"),
    autoescape=select_autoescape(["html", "j2"]),
    trim_blocks=True,
    lstrip_blocks=True,
)


def render_html(
    result: dict[str, Any],
    gate: GateResult | None = None,
    baseline: dict[str, Any] | None = None,
) -> str:
    template = _env.get_template("report.html.j2")
    return template.render(
        run=result["run"],
        rows=summary_table(result, baseline),
        cases=result["cases"],
        metrics=QUALITY_METRICS,
        gate=gate,
        baseline_run=baseline["run"] if baseline else None,
        fmt=fmt,
    )


def write_reports(
    result: dict[str, Any],
    out_dir: Path,
    gate: GateResult | None = None,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write results.json, report.md and report.html into ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": out_dir / "results.json",
        "markdown": out_dir / "report.md",
        "html": out_dir / "report.html",
    }
    write_json(result, paths["json"])
    paths["markdown"].write_text(render_markdown(result, gate, baseline), encoding="utf-8")
    paths["html"].write_text(render_html(result, gate, baseline), encoding="utf-8")
    if gate is not None:
        paths["gate"] = out_dir / "gate.json"
        write_json(gate.to_dict(), paths["gate"])
    return paths
