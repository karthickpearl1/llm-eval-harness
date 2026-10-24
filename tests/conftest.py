from __future__ import annotations

from pathlib import Path

import pytest

from llm_eval.config import load_config
from llm_eval.rag import BM25Retriever, load_corpus

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _no_provider_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's local provider settings from leaking into the tests."""
    for name in ("LLM_EVAL_PROVIDER", "LLM_EVAL_JUDGE_PROVIDER"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def chunks():
    return load_corpus(ROOT / "data" / "corpus")


@pytest.fixture(scope="session")
def retriever(chunks) -> BM25Retriever:
    return BM25Retriever(chunks)


@pytest.fixture
def baseline_config():
    return load_config(ROOT / "eval.yaml")


@pytest.fixture
def degraded_config():
    return load_config(ROOT / "configs" / "candidate-degraded.yaml")
