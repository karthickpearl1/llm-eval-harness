import json

import pytest

from llm_eval.dataset import GoldenCase, load_dataset


def test_golden_dataset_loads(root):
    cases = load_dataset(root / "datasets" / "golden.yaml")
    assert len(cases) >= 20
    behaviours = {c.expected_behaviour for c in cases}
    assert behaviours == {"answer", "refuse", "any"}


def test_every_reference_context_exists(root, chunks):
    doc_ids = {c.doc_id for c in chunks}
    for case in load_dataset(root / "datasets" / "golden.yaml"):
        assert set(case.reference_contexts) <= doc_ids, case.id


def test_jsonl_is_supported(tmp_path):
    path = tmp_path / "cases.jsonl"
    rows = [
        {"id": "a", "question": "q?", "reference_answer": "r", "must_include": ["r"]},
        {"id": "b", "question": "q2?", "expected_behaviour": "refuse"},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n\n")
    cases = load_dataset(path)
    assert [c.id for c in cases] == ["a", "b"]
    assert cases[0].must_include == ("r",)


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ({"question": "q"}, "missing required"),
        ({"id": "x", "question": "q", "expected_behaviour": "maybe"}, "expected_behaviour"),
        ({"id": "x", "question": "q"}, "reference_answer"),
    ],
)
def test_invalid_cases_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        GoldenCase.from_dict(raw)


def test_duplicate_ids_rejected(tmp_path):
    path = tmp_path / "d.yaml"
    path.write_text(
        "cases:\n"
        "  - {id: a, question: q, expected_behaviour: refuse}\n"
        "  - {id: a, question: q, expected_behaviour: refuse}\n"
    )
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


def test_unsupported_and_empty_files(tmp_path):
    with pytest.raises(ValueError, match="Unsupported"):
        load_dataset(tmp_path / "cases.csv")
    empty = tmp_path / "empty.yaml"
    empty.write_text("cases: []\n")
    with pytest.raises(ValueError, match="No cases"):
        load_dataset(empty)
