"""OpenAI and Azure OpenAI chat-completions providers.

The ``openai`` package is an optional extra (``pip install .[openai]``) and is imported
lazily, so the harness installs and runs without it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from llm_eval.providers.base import Completion, LLMProvider, ProviderError, estimate_tokens


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise ProviderError(f"Environment variable {name} is required for this provider")
    return value


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, model: str | None = None, *, client: Any | None = None) -> None:
        super().__init__(model or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini")
        self._client = client or self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError("Install the 'openai' extra: pip install '.[openai]'") from exc
        return OpenAI(api_key=_require_env("OPENAI_API_KEY"))

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        messages = [{"role": "system", "content": system}] if system else []
        messages.append({"role": "user", "content": prompt})

        started = time.perf_counter()
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise ProviderError(f"{self.name} call failed: {exc}") from exc
        latency_ms = (time.perf_counter() - started) * 1000

        text = response.choices[0].message.content or ""
        usage = getattr(response, "usage", None)
        return Completion(
            text=text,
            model=self.model,
            input_tokens=getattr(usage, "prompt_tokens", None) or estimate_tokens(prompt),
            output_tokens=getattr(usage, "completion_tokens", None) or estimate_tokens(text),
            latency_ms=round(latency_ms, 1),
        )


class AzureOpenAIProvider(OpenAIProvider):
    """Azure OpenAI uses the same API shape; the 'model' is the deployment name."""

    name = "azure_openai"

    def __init__(self, model: str | None = None, *, client: Any | None = None) -> None:
        deployment = model or os.environ.get("AZURE_OPENAI_DEPLOYMENT")
        if not deployment and client is None:
            raise ProviderError("Set AZURE_OPENAI_DEPLOYMENT or pass a model/deployment name")
        LLMProvider.__init__(self, deployment or "azure-deployment")
        self._client = client or self._build_client()

    def _build_client(self) -> Any:
        try:
            from openai import AzureOpenAI
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError("Install the 'openai' extra: pip install '.[openai]'") from exc
        return AzureOpenAI(
            azure_endpoint=_require_env("AZURE_OPENAI_ENDPOINT"),
            api_key=_require_env("AZURE_OPENAI_API_KEY"),
            api_version=os.environ.get("AZURE_OPENAI_API_VERSION") or "2024-06-01",
        )
