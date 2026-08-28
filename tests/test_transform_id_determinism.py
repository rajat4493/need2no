"""Regression coverage for a real bug: MuPDF's trailer /ID can be written
as either a hex string (`<...>`) or a PDF literal string (`(...)`, with
backslash escapes and PDF-legal balanced unescaped parens) for either of
its two entries. The original fix only matched the hex form and silently
left MuPDF's own random ID in place whenever the literal-string form
showed up — discovered via a 400-iteration stress test that found 3
distinct output hashes for identical input, not by inspection.
"""

from __future__ import annotations

from n2n.transform import _make_id_deterministic


def test_hex_hex_form_is_normalized():
    original = b"trailer\n<</Size 7/Root 1 0 R/ID[<aabbccdd><11223344>]>>\n%%EOF"
    out = _make_id_deterministic(original)
    assert out != original
    assert out.count(b"/ID[<") == 1
    assert b"<aabbccdd>" not in out
    assert b"<11223344>" not in out


def test_hex_literal_form_is_normalized():
    # The exact failure shape found by the stress test: first entry hex,
    # second entry a literal string containing escaped/binary bytes.
    original = b"trailer\n<</Size 7/Root 1 0 R/ID[<C29B20C2B4C2BC56>(\\t\\337_7e\\0003\\n)]>>\n%%EOF"
    out = _make_id_deterministic(original)
    assert out != original
    assert b"(\\t\\337" not in out
    assert out.count(b"/ID[<") == 1


def test_literal_string_with_escaped_parens_is_parsed_correctly():
    # A literal string may contain balanced, unescaped parens, or escaped
    # ones — both are legal PDF syntax and must not confuse the scan for
    # where the string actually ends.
    original = b"trailer\n<</Size 7/ID[(ab(nested)cd)<aabbccdd>]>>\n%%EOF"
    out = _make_id_deterministic(original)
    assert out != original
    assert b"nested" not in out
    assert out.endswith(b">>\n%%EOF")


def test_literal_string_with_escaped_backslash_before_paren():
    # "\\)" is an escaped backslash followed by a real closing paren, not
    # an escaped paren — the scanner must not misread the escape.
    original = b"trailer\n<</Size 7/ID[(ab\\\\)<aabbccdd>]>>\n%%EOF"
    out = _make_id_deterministic(original)
    assert out != original


def test_no_id_field_leaves_bytes_unchanged():
    original = b"trailer\n<</Size 7/Root 1 0 R>>\n%%EOF"
    assert _make_id_deterministic(original) == original


def test_malformed_id_field_leaves_bytes_unchanged_rather_than_corrupting():
    original = b"trailer\n<</Size 7/ID[<unterminated]>>\n%%EOF"
    assert _make_id_deterministic(original) == original


def test_replacement_is_a_pure_function_of_the_surrounding_bytes():
    """Two documents whose ID differs only in encoding, but whose
    surrounding bytes are otherwise identical, must normalize to the same
    replacement — proving the digest is computed over the blanked
    (ID-removed) content, not accidentally over the ID's own bytes."""
    hex_form = b"trailer\n<</Size 7/Root 1 0 R/ID[<aabbccdd><11223344>]>>\n%%EOF"
    literal_form = b"trailer\n<</Size 7/Root 1 0 R/ID[(xy)(zz)]>>\n%%EOF"
    out_hex = _make_id_deterministic(hex_form)
    out_literal = _make_id_deterministic(literal_form)
    id_value_hex = out_hex.split(b"/ID")[1]
    id_value_literal = out_literal.split(b"/ID")[1]
    assert id_value_hex == id_value_literal
