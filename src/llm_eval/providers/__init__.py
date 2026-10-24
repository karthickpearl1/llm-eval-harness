"""Provider registry.

Providers are chosen by name in the eval config and can be overridden with the
``LLM_EVAL_PROVIDER`` environment variable, so CI can run the same suite against the mock
on every pull request and against a real model on a schedule.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_eval.providers.base import Completion, LLMProvider, ProviderError, estimate_tokens
from llm_eval.providers.mock import MockProvider

__all__ = [
    "PROVIDERS",
    "Completion",
    "LLMProvider",
    "MockProvider",
    "ProviderError",
    "create_provider",
    "estimate_tokens",
]


def _openai(model: str | None, **options: Any) -> LLMProvider:
    from llm_eval.providers.openai_provider import OpenAIProvider

    return OpenAIProvider(model, **options)


def _azure(model: str | None, **options: Any) -> LLMProvider:
    from llm_eval.providers.openai_provider import AzureOpenAIProvider

    return AzureOpenAIProvider(model, **options)


def _bedrock(model: str | None, **options: Any) -> LLMProvider:
    from llm_eval.providers.bedrock import BedrockProvider

    return BedrockProvider(model, **options)


def _mock(model: str | None, **options: Any) -> LLMProvider:
    return MockProvider(model or "mock-extractive-v1", **options)


PROVIDERS: dict[str, Callable[..., LLMProvider]] = {
    "mock": _mock,
    "openai": _openai,
    "azure_openai": _azure,
    "bedrock": _bedrock,
}


def create_provider(name: str, model: str | None = None, **options: Any) -> LLMProvider:
    """Instantiate a provider by name."""
    try:
        factory = PROVIDERS[name]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise ProviderError(f"Unknown provider '{name}'. Known providers: {known}") from None
    return factory(model, **options)
