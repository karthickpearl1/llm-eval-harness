import pytest

from llm_eval.providers import MockProvider
from llm_eval.rag import BM25Retriever, RagApp, load_corpus
from llm_eval.rag.corpus import chunk_document


def test_corpus_skips_readme_and_headings(chunks):
    doc_ids = {c.doc_id for c in chunks}
    assert "README" not in doc_ids
    assert {"refunds", "bikes", "delay-repay"} <= doc_ids
    assert not any(c.text.startswith("#") for c in chunks)
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_document_groups_paragraphs():
    chunks = chunk_document("doc", "# T\n\none\n\ntwo\n\nthree", paragraphs_per_chunk=2)
    assert [c.text for c in chunks] == ["one two", "three"]
    assert chunks[1].chunk_id == "doc#1"


def test_load_corpus_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "missing")
    with pytest.raises(ValueError, match="No documents"):
        load_corpus(tmp_path)


def test_retriever_ranks_relevant_document_first(retriever):
    results = retriever.retrieve("How much is the 16-25 Railcard?", top_k=2)
    assert results[0].doc_id == "railcards"
    assert len(results) == 2


def test_retriever_returns_nothing_for_unrelated_query(retriever):
    assert retriever.retrieve("quantum chromodynamics", top_k=3) == []


def test_retriever_requires_chunks():
    with pytest.raises(ValueError, match="at least one chunk"):
        BM25Retriever([])


def test_app_redacts_pii_by_default(retriever):
    question = "What email address can the INC-1042 customer be contacted on?"
    app = RagApp(MockProvider(), retriever)
    response = app.ask(question)
    assert "[REDACTED_EMAIL]" in response.answer
    assert "example.com" not in response.answer

    unguarded = RagApp(MockProvider(), retriever, redact_pii=False).ask(question)
    assert "jane.doe@example.com" in unguarded.answer


def test_app_prompt_contains_context_and_question(retriever):
    app = RagApp(MockProvider(), retriever, top_k=1)
    contexts = retriever.retrieve("bikes", top_k=1)
    prompt = app.build_prompt("Can I take a bike?", contexts)
    assert "<context>" in prompt
    assert contexts[0].text in prompt
    assert "<question>\nCan I take a bike?\n</question>" in prompt
