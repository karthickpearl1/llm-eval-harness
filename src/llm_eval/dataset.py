"""Golden dataset loading and validation (YAML or JSONL)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

BEHAVIOURS = ("answer", "refuse", "any")


@dataclass(frozen=True)
class GoldenCase:
    """One evaluation case.

    ``expected_behaviour`` is ``answer`` (should answer from the docs), ``refuse`` (out of
    scope, should decline) or ``any`` (either is fine, used for safety probes where the only
    requirement is that no personal data leaks).
    """

    id: str
    question: str
    reference_answer: str = ""
    must_include: tuple[str, ...] = ()
    reference_contexts: tuple[str, ...] = ()
    expected_behaviour: str = "answer"
    tags: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> GoldenCase:
        missing = [key for key in ("id", "question") if not raw.get(key)]
        if missing:
            raise ValueError(f"Case is missing required field(s) {missing}: {raw}")
        behaviour = raw.get("expected_behaviour", "answer")
        if behaviour not in BEHAVIOURS:
            raise ValueError(f"Case {raw['id']}: expected_behaviour must be one of {BEHAVIOURS}")
        if behaviour == "answer" and not raw.get("reference_answer"):
            raise ValueError(f"Case {raw['id']}: answer cases need a reference_answer")
        return cls(
            id=str(raw["id"]),
            question=str(raw["question"]),
            reference_answer=str(raw.get("reference_answer", "")),
            must_include=tuple(str(x) for x in raw.get("must_include", [])),
            reference_contexts=tuple(str(x) for x in raw.get("reference_contexts", [])),
            expected_behaviour=behaviour,
            tags=tuple(str(x) for x in raw.get("tags", [])),
        )


def load_dataset(path: str | Path) -> list[GoldenCase]:
    """Load cases from ``.yaml``/``.yml`` (a ``cases:`` list) or ``.jsonl`` (one per line)."""
    path = Path(path)
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        rows = data.get("cases", []) if isinstance(data, dict) else data
    elif path.suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        raise ValueError(f"Unsupported dataset format: {path.suffix} (use .yaml or .jsonl)")

    cases = [GoldenCase.from_dict(row) for row in rows]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {duplicates}")
    if not cases:
        raise ValueError(f"No cases found in {path}")
    return cases
