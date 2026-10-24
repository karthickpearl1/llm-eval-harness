import pytest

from llm_eval.safety import find_pii, is_refusal, redact_pii


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("write to jane.doe@example.com today", "email"),
        ("call 07700 900123", "uk_phone"),
        ("call +44 7700 900123", "uk_phone"),
        ("NI number QQ 12 34 56 C", "ni_number"),
        ("card 4111 1111 1111 1111", "card_number"),
    ],
)
def test_find_pii_detects_common_uk_identifiers(text, kind):
    assert kind in {f.kind for f in find_pii(text)}


def test_card_numbers_must_pass_luhn():
    assert find_pii("ref 4111 1111 1111 1112") == []


@pytest.mark.parametrize(
    "text",
    [
        "A delay of 120 minutes earns 100% of the return fare.",
        "The 08:15 service from Leeds left at 10:00 on 28 days notice for £12.",
    ],
)
def test_policy_text_is_not_flagged(text):
    assert find_pii(text) == []


def test_redact_pii_replaces_with_typed_placeholders():
    redacted = redact_pii("Contact jane.doe@example.com or 07700 900123.")
    assert redacted == "Contact [REDACTED_EMAIL] or [REDACTED_UK_PHONE]."
    assert find_pii(redacted) == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("I can't answer that from the provided documents.", True),
        ("I don't know.", True),
        ("I'm unable to help with that.", True),
        ("That is outside the scope of this assistant.", True),
        ("Bicycles travel free of charge.", False),
        ("Unused Advance tickets cannot be refunded.", False),
    ],
)
def test_is_refusal(text, expected):
    assert is_refusal(text) is expected
