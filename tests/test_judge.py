import pytest

from llm_eval.dataset import GoldenCase
from llm_eval.metrics import LLMJudge, MockJudge, create_judge
from llm_eval.metrics.judge import parse_verdict
from llm_eval.providers import PROVIDERS, Completion, LLMProvider, ProviderError

CASE = GoldenCase(
    id="fee",
    question="How much is the refund administration fee?",
    reference_answer="An administration fee of £10 is deducted from each refund.",
    must_include=("£10",),
)


def test_mock_judge_rewards_correct_answers():
    good = MockJudge().score(CASE, "An administration fee of £10 is deducted from each refund.")
    bad = MockJudge().score(CASE, "Bicycles travel free of charge.")
    assert good.correctness > bad.correctness
    assert good.relevance > bad.relevance
    assert 0.0 <= bad.correctness <= good.correctness <= 1.0


def test_mock_judge_scores_refusal_as_zero():
    verdict = MockJudge().score(CASE, "I can't answer that from the provided documents.")
    assert (verdict.correctness, verdict.relevance) == (0.0, 0.0)


def test_mock_judge_without_required_facts_uses_similarity():
    case = GoldenCase(id="x", question="q?", reference_answer="Bikes are free.")
    assert MockJudge().score(case, "Bikes are free.").correctness == 1.0


@pytest.mark.parametrize(
    ("text", "correctness"),
    [
        ('{"correctness": 5, "relevance": 3, "reason": "ok"}', 1.0),
        ('```json\n{"correctness": 1, "relevance": 5}\n```', 0.0),
        ('Sure! {"correctness": 3, "relevance": 3}', 0.5),
        ('{"correctness": 9, "relevance": 3}', 1.0),  # clamped
    ],
)
def test_parse_verdict(text, correctness):
    assert parse_verdict(text).correctness == correctness


@pytest.mark.parametrize("text", ["no json here", '{"relevance": 3}', '{"correctness": "high"}'])
def test_parse_verdict_rejects_bad_output(text):
    with pytest.raises(ProviderError):
        parse_verdict(text)


class ScriptedProvider(LLMProvider):
    name = "scripted"

    def __init__(self, reply: str) -> None:
        super().__init__("scripted-1")
        self.reply = reply
        self.prompts: list[str] = []

    def complete(self, prompt, *, system=None, max_tokens=512, temperature=0.0):
        self.prompts.append(prompt)
        return Completion(self.reply, self.model, 10, 10, 1.0)


def test_llm_judge_sends_rubric_and_parses_reply():
    provider = ScriptedProvider('{"correctness": 4, "relevance": 5, "reason": "fine"}')
    verdict = LLMJudge(provider).score(CASE, "The fee is £10.")
    assert verdict.correctness == 0.75
    assert verdict.relevance == 1.0
    assert "REFERENCE: An administration fee" in provider.prompts[0]
    assert "ANSWER: The fee is £10." in provider.prompts[0]


def test_create_judge(monkeypatch):
    assert isinstance(create_judge("mock"), MockJudge)

    monkeypatch.setitem(PROVIDERS, "scripted", lambda model, **_: ScriptedProvider("{}"))
    judge = create_judge("scripted")
    assert isinstance(judge, LLMJudge)
    assert judge.name == "scripted"

    with pytest.raises(ProviderError):
        create_judge("unknown")
