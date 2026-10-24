"""Eval configuration: a YAML file, optionally extending another one.

``extends`` lets a candidate config state only what differs from the baseline, which keeps
experiments readable in code review.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

_PATH_KEYS = ("dataset", "corpus")


@dataclass
class ProviderConfig:
    name: str = "mock"
    model: str | None = None
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class GateConfig:
    """Thresholds for the regression gate.

    ``min``/``max`` are absolute bounds on the candidate's summary metrics; ``max_drop`` is
    the largest allowed fall versus the baseline; ``max_increase_pct`` is the largest allowed
    relative rise (used for latency and cost); ``max_new_failures`` caps cases that passed in
    the baseline but fail in the candidate.
    """

    min: dict[str, float] = field(default_factory=dict)
    max: dict[str, float] = field(default_factory=dict)
    max_drop: dict[str, float] = field(default_factory=dict)
    max_increase_pct: dict[str, float] = field(default_factory=dict)
    max_new_failures: int | None = 0


@dataclass
class EvalConfig:
    name: str
    dataset: Path
    corpus: Path
    provider: ProviderConfig
    judge: ProviderConfig
    top_k: int = 3
    paragraphs_per_chunk: int = 1
    redact_pii: bool = True
    case_thresholds: dict[str, float] = field(default_factory=dict)
    pricing: dict[str, dict[str, float]] = field(default_factory=dict)
    gate: GateConfig = field(default_factory=GateConfig)
    source: Path | None = None


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _read_raw(path: Path, seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Circular 'extends' in config: {path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    # Resolve paths against the file that declares them, so 'extends' works across folders.
    for key in _PATH_KEYS:
        if key in raw:
            raw[key] = str(path.parent / raw[key])
    parent = raw.pop("extends", None)
    if parent:
        base = _read_raw(path.parent / parent, (*seen, path))
        raw = _deep_merge(base, raw)
    return raw


def _provider(raw: dict[str, Any] | None, env_var: str) -> ProviderConfig:
    raw = dict(raw or {})
    configured = raw.get("name", "mock")
    override = os.environ.get(env_var)
    if override and override != configured:
        # Options such as the mock's degradation knobs belong to the configured provider
        # only, so they are dropped when the environment selects a different one.
        return ProviderConfig(name=override)
    return ProviderConfig(configured, raw.get("model"), dict(raw.get("options") or {}))


def load_config(path: str | Path) -> EvalConfig:
    path = Path(path)
    raw = _read_raw(path)
    try:
        dataset = Path(raw["dataset"])
        corpus = Path(raw["corpus"])
    except KeyError as exc:
        raise ValueError(f"Config {path} is missing required key {exc}") from None

    app = raw.get("app", {})
    gate = raw.get("gate", {})
    return EvalConfig(
        name=raw.get("name", path.stem),
        dataset=dataset,
        corpus=corpus,
        provider=_provider(raw.get("provider"), "LLM_EVAL_PROVIDER"),
        judge=_provider(raw.get("judge"), "LLM_EVAL_JUDGE_PROVIDER"),
        top_k=int(app.get("top_k", 3)),
        paragraphs_per_chunk=int(app.get("paragraphs_per_chunk", 1)),
        redact_pii=bool(app.get("redact_pii", True)),
        case_thresholds={k: float(v) for k, v in raw.get("case_thresholds", {}).items()},
        pricing=raw.get("pricing", {}),
        gate=GateConfig(
            min=gate.get("min", {}),
            max=gate.get("max", {}),
            max_drop=gate.get("max_drop", {}),
            max_increase_pct=gate.get("max_increase_pct", {}),
            max_new_failures=gate.get("max_new_failures", 0),
        ),
        source=path,
    )
