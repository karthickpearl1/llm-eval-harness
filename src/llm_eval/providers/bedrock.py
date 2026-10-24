"""AWS Bedrock provider using the model-agnostic Converse API.

Credentials come from the standard AWS chain (environment, profile, SSO or an assumed role),
so nothing secret is ever read from the harness config. ``boto3`` is an optional extra.
"""

from __future__ import annotations

import os
import time
from typing import Any

from llm_eval.providers.base import Completion, LLMProvider, ProviderError, estimate_tokens

DEFAULT_MODEL = "anthropic.claude-3-haiku-20240307-v1:0"


class BedrockProvider(LLMProvider):
    name = "bedrock"

    def __init__(self, model: str | None = None, *, client: Any | None = None) -> None:
        super().__init__(model or os.environ.get("BEDROCK_MODEL_ID") or DEFAULT_MODEL)
        self._client = client or self._build_client()

    @staticmethod
    def _build_client() -> Any:
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover - depends on optional extra
            raise ProviderError("Install the 'bedrock' extra: pip install '.[bedrock]'") from exc
        region = os.environ.get("AWS_REGION") or "eu-west-2"
        return boto3.client("bedrock-runtime", region_name=region)

    def complete(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Completion:
        request: dict[str, Any] = {
            "modelId": self.model,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {"maxTokens": max_tokens, "temperature": temperature},
        }
        if system:
            request["system"] = [{"text": system}]

        started = time.perf_counter()
        try:
            response = self._client.converse(**request)
        except Exception as exc:
            raise ProviderError(f"bedrock call failed: {exc}") from exc
        measured_ms = (time.perf_counter() - started) * 1000

        content = response["output"]["message"]["content"]
        text = "".join(part.get("text", "") for part in content)
        usage = response.get("usage", {})
        latency_ms = response.get("metrics", {}).get("latencyMs", measured_ms)
        return Completion(
            text=text,
            model=self.model,
            input_tokens=usage.get("inputTokens") or estimate_tokens(prompt),
            output_tokens=usage.get("outputTokens") or estimate_tokens(text),
            latency_ms=round(float(latency_ms), 1),
        )
