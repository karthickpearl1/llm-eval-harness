"""The regression gate: absolute floors plus baseline-vs-candidate comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from llm_eval.config import GateConfig


@dataclass(frozen=True)
class Check:
    rule: str
    metric: str
    candidate: float | None
    threshold: float
    baseline: float | None
    passed: bool
    message: str


@dataclass
class GateResult:
    checks: list[Check] = field(default_factory=list)
    new_failures: list[str] = field(default_factory=list)
    fixed_cases: list[str] = field(default_factory=list)
    max_new_failures: int | None = 0

    @property
    def passed(self) -> bool:
        too_many_new = (
            self.max_new_failures is not None and len(self.new_failures) > self.max_new_failures
        )
        return all(c.passed for c in self.checks) and not too_many_new

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "checks": [asdict(c) for c in self.checks],
            "new_failures": self.new_failures,
            "fixed_cases": self.fixed_cases,
            "max_new_failures": self.max_new_failures,
        }


def _missing(rule: str, metric: str, threshold: float) -> Check:
    return Check(rule, metric, None, threshold, None, False, f"{metric} missing from results")


def _absolute_checks(summary: dict[str, float], gate: GateConfig) -> list[Check]:
    checks = []
    for rule, bounds in (("min", gate.min), ("max", gate.max)):
        for metric, bound in bounds.items():
            value = summary.get(metric)
            if value is None:
                checks.append(_missing(rule, metric, bound))
                continue
            ok = value >= bound if rule == "min" else value <= bound
            op = (">=" if ok else "<") if rule == "min" else ("<=" if ok else ">")
            message = f"{metric} {value:.4g} {op} {bound:g}"
            checks.append(Check(rule, metric, value, bound, None, ok, message))
    return checks


def _pct_change(base: float, value: float) -> float:
    if base == 0:
        return 0.0 if value == 0 else float("inf")
    return (value - base) / base * 100


def _relative_checks(
    summary: dict[str, float], base_summary: dict[str, float], gate: GateConfig
) -> list[Check]:
    checks = []
    for metric, allowed in gate.max_drop.items():
        value, base = summary.get(metric), base_summary.get(metric)
        if value is None or base is None:
            checks.append(_missing("max_drop", metric, allowed))
            continue
        change = round(value - base, 10) + 0.0  # "+ 0.0" turns -0.0 into 0.0
        ok = -change <= allowed + 1e-9
        message = (
            f"{metric} {base:.4g} -> {value:.4g} (change {change:+.4g}, allowed drop {allowed:g})"
        )
        checks.append(Check("max_drop", metric, value, allowed, base, ok, message))

    for metric, allowed_pct in gate.max_increase_pct.items():
        value, base = summary.get(metric), base_summary.get(metric)
        if value is None or base is None:
            checks.append(_missing("max_increase_pct", metric, allowed_pct))
            continue
        change_pct = _pct_change(base, value)
        ok = change_pct <= allowed_pct + 1e-9
        message = (
            f"{metric} {base:.4g} -> {value:.4g} ({change_pct:+.1f}%, allowed +{allowed_pct:g}%)"
        )
        checks.append(Check("max_increase_pct", metric, value, allowed_pct, base, ok, message))
    return checks


def evaluate_gate(
    candidate: dict[str, Any],
    gate: GateConfig,
    baseline: dict[str, Any] | None = None,
) -> GateResult:
    """Check a candidate run against the gate; pass ``baseline`` to enable regression rules."""
    result = GateResult(
        checks=_absolute_checks(candidate["summary"], gate),
        max_new_failures=gate.max_new_failures if baseline else None,
    )
    if baseline is None:
        return result

    result.checks += _relative_checks(candidate["summary"], baseline["summary"], gate)

    was_passing = {c["id"]: c["passed"] for c in baseline["cases"]}
    for case in candidate["cases"]:
        before = was_passing.get(case["id"])
        if before and not case["passed"]:
            result.new_failures.append(case["id"])
        elif before is False and case["passed"]:
            result.fixed_cases.append(case["id"])
    return result
