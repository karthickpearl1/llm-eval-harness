from llm_eval.text import content_terms, normalise, split_sentences, stem


def test_normalise_strips_case_punctuation_and_accents():
    assert normalise("  Café, OPEN!  £10 ") == "cafe open £10"


def test_stem_conflates_simple_inflections():
    assert stem("refunds") == stem("refund")
    assert stem("charged") == stem("charges") == stem("charge")
    assert stem("policies") == "policy"
    assert stem("bike") == "bike"  # short words are left alone
    assert stem("2024") == "2024"


def test_content_terms_drop_stopwords():
    assert content_terms("What is the fee for a refund?") == ["fee", "refund"]


def test_split_sentences_skips_headings_and_joins_wrapped_lines():
    text = "# Title\n\nFirst sentence. Second\nsentence here.\n\nThird one!"
    assert split_sentences(text) == ["First sentence.", "Second sentence here.", "Third one!"]
