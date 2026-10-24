import pytest

from llm_eval.metrics import (
    HashingEmbedder,
    context_precision,
    context_recall,
    exact_match,
    fact_recall,
    faithfulness,
    semantic_similarity,
    token_f1,
)
from llm_eval.rag import Chunk

CONTEXT = [
    Chunk("refunds#2", "refunds", "An administration fee of £10 is deducted from each refund."),
    Chunk("bikes#0", "bikes", "Bicycles travel free of charge on all services."),
]


def test_exact_match_ignores_case_and_punctuation():
    assert exact_match("The fee is £10.", "the fee is £10") == 1.0
    assert exact_match("The fee is £10", "The fee is £5") == 0.0


def test_token_f1():
    assert token_f1("fee is ten pounds", "fee is ten pounds") == pytest.approx(1.0)
    assert token_f1("refund fee", "refund admin fee") == pytest.approx(0.8)
    assert token_f1("", "anything") == 0.0
    assert token_f1("bicycles", "refund") == 0.0


def test_fact_recall():
    assert fact_recall("A £10 fee within 28 days", ["£10", "28 days"]) == 1.0
    assert fact_recall("A £10 fee", ["£10", "28 days"]) == 0.5
    assert fact_recall("anything", []) is None


def test_semantic_similarity_orders_sensibly():
    reference = "An administration fee of £10 is deducted from each refund."
    close = semantic_similarity("A £10 administration fee is deducted from refunds.", reference)
    far = semantic_similarity("Bicycles travel free of charge.", reference)
    assert 0.0 <= far < close <= 1.0
    assert semantic_similarity(reference, reference) == pytest.approx(1.0)


def test_semantic_similarity_accepts_custom_embedder():
    class Constant:
        def embed(self, text):
            return [1.0, 0.0]

    assert semantic_similarity("a", "b", embedder=Constant()) == pytest.approx(1.0)


def test_hashing_embedder_handles_empty_text():
    assert semantic_similarity("", "something", embedder=HashingEmbedder(64)) == 0.0


def test_faithfulness_flags_unsupported_sentences():
    grounded = "An administration fee of £10 is deducted from each refund."
    assert faithfulness(grounded, CONTEXT) == 1.0
    invented = grounded + " Loyalty card holders receive double compensation from the director."
    assert faithfulness(invented, CONTEXT) == 0.5


def test_faithfulness_not_applicable_to_refusals_or_empty_answers():
    assert faithfulness("I can't answer that from the provided documents.", CONTEXT) is None
    assert faithfulness("", CONTEXT) is None
    assert faithfulness("[REDACTED_EMAIL]", CONTEXT) is None


def test_faithfulness_ignores_redaction_placeholders():
    answer = "An administration fee of £10 is deducted from each refund [REDACTED_EMAIL]."
    assert faithfulness(answer, CONTEXT) == 1.0


def test_context_recall_and_precision():
    assert context_recall(CONTEXT, ("refunds",)) == 1.0
    assert context_recall(CONTEXT, ("refunds", "railcards")) == 0.5
    assert context_precision(CONTEXT, ("refunds",)) == 0.5
    assert context_precision([], ("refunds",)) == 0.0
    assert context_recall(CONTEXT, ()) is None
    assert context_precision(CONTEXT, ()) is None
