from types import SimpleNamespace

import pytest

from llm_eval.providers import MockProvider, ProviderError, create_provider, estimate_tokens
from llm_eval.providers.bedrock import BedrockProvider
from llm_eval.providers.mock import REFUSAL_TEXT
from llm_eval.providers.openai_provider import AzureOpenAIProvider, OpenAIProvider

PROMPT = """<context>
Bicycles travel free of charge. Each train carries a maximum of 2 full-size bicycles.
</context>
<question>
How many full-size bicycles can each train carry?
</question>"""


def test_mock_extracts_best_matching_sentence():
    completion = MockProvider().complete(PROMPT)
    assert completion.text == "Each train carries a maximum of 2 full-size bicycles."
    assert completion.model == "mock-extractive-v1"
    assert completion.input_tokens > completion.output_tokens > 0


def test_mock_is_deterministic():
    provider = MockProvider(hallucination_rate=0.5)
    assert provider.complete(PROMPT) == provider.complete(PROMPT)


def test_mock_refuses_without_relevant_context():
    prompt = "<context>Bicycles travel free.</context><question>Weather tomorrow?</question>"
    assert MockProvider().complete(prompt).text == REFUSAL_TEXT
    assert MockProvider().complete("no tags at all").text == REFUSAL_TEXT


def test_mock_hallucination_appends_unsupported_claim():
    clean = MockProvider().complete(PROMPT).text
    noisy = MockProvider(hallucination_rate=1.0).complete(PROMPT).text
    assert noisy.startswith(clean)
    assert len(noisy) > len(clean)


def test_mock_extra_latency_is_added():
    fast = MockProvider().complete(PROMPT).latency_ms
    slow = MockProvider(extra_latency_ms=100).complete(PROMPT).latency_ms
    assert slow == pytest.approx(fast + 100)


def test_mock_rejects_invalid_rate():
    with pytest.raises(ValueError, match="hallucination_rate"):
        MockProvider(hallucination_rate=1.5)


def test_estimate_tokens():
    assert estimate_tokens("") == 0
    assert estimate_tokens("abcd" * 10) == 10


def test_create_provider_unknown_name():
    with pytest.raises(ProviderError, match="Unknown provider"):
        create_provider("nope")


def test_create_provider_passes_options():
    provider = create_provider("mock", "custom", max_sentences=3)
    assert isinstance(provider, MockProvider)
    assert provider.model == "custom"
    assert provider.max_sentences == 3


class FakeChatClient:
    def __init__(self, text="Hello", usage=True, error=None):
        self.calls = []
        self._text, self._usage, self._error = text, usage, error
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        usage = SimpleNamespace(prompt_tokens=12, completion_tokens=3) if self._usage else None
        message = SimpleNamespace(content=self._text)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def test_openai_provider_maps_request_and_usage():
    client = FakeChatClient()
    completion = OpenAIProvider("gpt-test", client=client).complete("hi", system="be brief")
    assert completion.text == "Hello"
    assert (completion.input_tokens, completion.output_tokens) == (12, 3)
    request = client.calls[0]
    assert request["model"] == "gpt-test"
    assert request["messages"][0] == {"role": "system", "content": "be brief"}
    assert request["temperature"] == 0.0


def test_openai_provider_estimates_tokens_when_usage_missing():
    completion = OpenAIProvider("gpt-test", client=FakeChatClient(usage=False)).complete("x" * 40)
    assert completion.input_tokens == 10


def test_openai_provider_wraps_errors():
    provider = OpenAIProvider("gpt-test", client=FakeChatClient(error=RuntimeError("boom")))
    with pytest.raises(ProviderError, match="boom"):
        provider.complete("hi")


def test_openai_provider_requires_api_key(monkeypatch):
    pytest.importorskip("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderError, match="OPENAI_API_KEY"):
        OpenAIProvider()


def test_azure_provider_uses_deployment_name(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "my-deployment")
    provider = AzureOpenAIProvider(client=FakeChatClient())
    assert provider.name == "azure_openai"
    assert provider.model == "my-deployment"
    provider.complete("hi")


def test_azure_provider_requires_deployment(monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_DEPLOYMENT", raising=False)
    with pytest.raises(ProviderError, match="AZURE_OPENAI_DEPLOYMENT"):
        AzureOpenAIProvider()


class FakeBedrockClient:
    def __init__(self, error=None):
        self.calls = []
        self._error = error

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        if self._error:
            raise self._error
        return {
            "output": {"message": {"content": [{"text": "Hi "}, {"text": "there"}]}},
            "usage": {"inputTokens": 20, "outputTokens": 2},
            "metrics": {"latencyMs": 345},
        }


def test_bedrock_provider_uses_converse_api():
    client = FakeBedrockClient()
    completion = BedrockProvider("model-x", client=client).complete("hi", system="sys")
    assert completion.text == "Hi there"
    assert completion.latency_ms == 345.0
    assert (completion.input_tokens, completion.output_tokens) == (20, 2)
    request = client.calls[0]
    assert request["modelId"] == "model-x"
    assert request["system"] == [{"text": "sys"}]


def test_bedrock_provider_wraps_errors():
    provider = BedrockProvider("model-x", client=FakeBedrockClient(error=RuntimeError("denied")))
    with pytest.raises(ProviderError, match="denied"):
        provider.complete("hi")


def test_factory_builds_hosted_providers_with_injected_clients(monkeypatch):
    monkeypatch.setenv("AZURE_OPENAI_DEPLOYMENT", "dep")
    assert create_provider("openai", "m", client=FakeChatClient()).name == "openai"
    assert create_provider("azure_openai", client=FakeChatClient()).model == "dep"
    assert create_provider("bedrock", "b", client=FakeBedrockClient()).name == "bedrock"
