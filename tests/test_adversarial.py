"""Adversarial corpus: try to break the engine with hostile-but-still-native-
text UK bank statement PDFs. This is not the full Phase 2 corpus (a few
hundred documents, real bank layouts, published benchmarks) — it's a first
pass at the specific failure classes called out in the build spec: split
text runs, hidden/invisible text, duplicate content, rotated pages, label
wording variance, multi-page placement, and malformed/encrypted inputs.

The property under test is never "does detection work perfectly" — it's
the fail-closed guarantee: the engine must never release a document with a
residual sensitive value, and every input must resolve to exactly one of
the five release states (never crash uncontrolled, never hang).
"""

from __future__ import annotations

import io

import pymupdf as fitz
import pdfplumber
import pytest

from n2n import pipeline

PACK_ID = "uk.bank_statement.share_with_ai"
SORT_CODE = "12-34-56"
ACCOUNT_NUMBER = "99887766"
IBAN = "GB94BARC10201530093459"
CARD_NUMBER = "4012 8888 8888 1881"


def _assert_no_leak(output_path):
    """Defense-in-depth: independently of the engine's own verify.py,
    re-check the released file with a THIRD extraction path (raw
    PyMuPDF chars this time) for any of the raw secret values."""
    doc = fitz.open(output_path)
    try:
        full_text = "\n".join(page.get_text("text") for page in doc)
    finally:
        doc.close()
    compact = full_text.replace(" ", "").replace("\n", "").replace("-", "")
    for secret in (SORT_CODE.replace("-", ""), ACCOUNT_NUMBER, IBAN, CARD_NUMBER.replace(" ", "")):
        assert secret not in compact, f"LEAK: {secret!r} survived in released output"


def _run(path, tmp_path, name="out"):
    # Output name is namespaced away from the input filename: several
    # fixtures pass a `name` that matches the input file's own basename,
    # and writing the certified output to that same path would make
    # out.exists() trivially true regardless of what the pipeline did.
    out = tmp_path / f"{name}.released.pdf"
    manifest = tmp_path / f"{name}.released.n2n.json"
    report = pipeline.run(path, PACK_ID, out, manifest)
    assert report.status in {
        "PASS_AUTO",
        "NEEDS_REVIEW",
        "UNSUPPORTED",
        "FAILED_VERIFY",
        "PROCESSING_ERROR",
    }
    if report.status == "PASS_AUTO":
        assert out.exists()
        _assert_no_leak(out)
    else:
        assert not out.exists()
    return report


# ---------------------------------------------------------------------------
# 1. Split text runs: each character of the sensitive value is its own
#    text-showing operation with real (if small) gaps, as some statement
#    generators / OCR layers do, instead of one clean word token.
# ---------------------------------------------------------------------------


def test_character_by_character_text_survives_or_refuses(tmp_path):
    path = tmp_path / "split_chars.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), "Sort code:", fontsize=10)
    x = 130
    for ch in SORT_CODE:
        page.insert_text((x, 90), ch, fontsize=10)
        x += 7
    page.insert_text((60, 110), "Account number:", fontsize=10)
    x = 170
    for ch in ACCOUNT_NUMBER:
        page.insert_text((x, 110), ch, fontsize=10)
        x += 7
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "split_chars")
    # Character-by-character text may or may not be recognized as a single
    # token by PyMuPDF's word segmentation; either way the engine must not
    # release a document with the value intact and undetected.
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert "sort_code" in structural or "account_number" not in structural


# ---------------------------------------------------------------------------
# 2. Duplicate content: the same sensitive value appears more than once on
#    the page (header + footer). Every occurrence must be removed.
# ---------------------------------------------------------------------------


def test_duplicated_account_number_all_occurrences_removed(tmp_path):
    path = tmp_path / "duplicate.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    page.insert_text((60, 700), f"Account number: {ACCOUNT_NUMBER}", fontsize=8)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "duplicate")
    if report.status == "PASS_AUTO":
        removed = [f for f in report.findings if f.field_id == "account_number"]
        assert len(removed) == 2, f"expected both occurrences detected, got {len(removed)}"


# ---------------------------------------------------------------------------
# 3. Hidden / invisible text (render mode 3) — the classic "visual cover
#    without removal" failure class the whole product exists to catch,
#    except here the danger is invisible sensitive text nobody drew a box
#    over because it was never meant to be seen, e.g. a stale OCR layer.
# ---------------------------------------------------------------------------


def test_invisible_text_layer_is_not_silently_released(tmp_path):
    path = tmp_path / "invisible.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), "Sort code: 00-00-00", fontsize=10)
    page.insert_text((60, 110), "Account number: 00000000", fontsize=10)
    # Invisible duplicate carrying the REAL values, e.g. a leftover OCR text
    # layer under a scanned-look statement.
    page.insert_text((60, 90), f"Sort code: {SORT_CODE}", fontsize=10, render_mode=3)
    page.insert_text((60, 110), f"Account number: {ACCOUNT_NUMBER}", fontsize=10, render_mode=3)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "invisible")
    # This must never end in PASS_AUTO with the real invisible values intact.
    if report.status == "PASS_AUTO":
        _assert_no_leak(tmp_path / "invisible.released.pdf")


# ---------------------------------------------------------------------------
# 4. Rotated pages — bbox/coordinate handling under page rotation.
# ---------------------------------------------------------------------------


def test_rotated_page_findings_map_to_correct_geometry(tmp_path):
    path = tmp_path / "rotated.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"Sort code: {SORT_CODE}", fontsize=10)
    page.insert_text((60, 110), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    page.set_rotation(90)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "rotated")
    if report.status == "PASS_AUTO":
        _assert_no_leak(tmp_path / "rotated.released.pdf")


# ---------------------------------------------------------------------------
# 5. Label wording variance across real bank statement phrasing.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sort_label,account_label",
    [
        ("Sort Code", "Account Number"),
        ("Sortcode", "Acc No"),
        ("SORT CODE:", "A/C NO:"),
        ("Sort code", "Account No."),
    ],
)
def test_label_wording_variants(tmp_path, sort_label, account_label):
    path = tmp_path / "labels.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"{sort_label}: {SORT_CODE}", fontsize=10)
    page.insert_text((60, 110), f"{account_label}: {ACCOUNT_NUMBER}", fontsize=10)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, f"labels_{sort_label}_{account_label}".replace(" ", "_").replace("/", "_"))
    structural = {f.field_id for f in report.findings if f.tier == "structural"}
    if report.status != "PASS_AUTO":
        # Acceptable outcome: refuse rather than release with the label
        # unrecognized and the value left unflagged.
        assert "sort_code" not in structural or "account_number" not in structural


# ---------------------------------------------------------------------------
# 6. Sensitive data on a later page of a multi-page statement.
# ---------------------------------------------------------------------------


def test_sensitive_data_on_page_three_is_still_caught(tmp_path):
    path = tmp_path / "multipage.pdf"
    doc = fitz.open()
    for i in range(2):
        page = doc.new_page()
        page.insert_text((60, 60), f"Statement page {i + 1}", fontsize=12)
        page.insert_text((60, 90), "Transaction history continues...", fontsize=10)
    page3 = doc.new_page()
    page3.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page3.insert_text((60, 90), f"Sort code: {SORT_CODE}", fontsize=10)
    page3.insert_text((60, 110), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "multipage")
    if report.status == "PASS_AUTO":
        pages_with_findings = {f.page for f in report.findings if f.tier == "structural"}
        assert 2 in pages_with_findings  # 0-indexed third page


# ---------------------------------------------------------------------------
# 7. Malformed / truncated PDF — must classify as UNSUPPORTED/corrupted,
#    never crash the process uncontrolled.
# ---------------------------------------------------------------------------


def test_truncated_pdf_is_rejected_not_crashed(tmp_path):
    path = tmp_path / "truncated.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    good_bytes = doc.tobytes()
    doc.close()
    path.write_bytes(good_bytes[: len(good_bytes) // 2])

    report = pipeline.run(path, PACK_ID, tmp_path / "out.pdf", tmp_path / "out.json")
    assert report.status in {"UNSUPPORTED", "PROCESSING_ERROR"}
    assert not (tmp_path / "out.pdf").exists()


def test_empty_file_is_rejected_not_crashed(tmp_path):
    path = tmp_path / "empty.pdf"
    path.write_bytes(b"")
    report = pipeline.run(path, PACK_ID, tmp_path / "out.pdf", tmp_path / "out.json")
    assert report.status in {"UNSUPPORTED", "PROCESSING_ERROR"}
    assert not (tmp_path / "out.pdf").exists()


# ---------------------------------------------------------------------------
# 8. Encrypted PDF — explicitly unsupported per spec 5.1.
# ---------------------------------------------------------------------------


def test_encrypted_pdf_is_unsupported(tmp_path):
    path = tmp_path / "encrypted.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    doc.save(path, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="user")
    doc.close()

    report = pipeline.run(path, PACK_ID, tmp_path / "out.pdf", tmp_path / "out.json")
    assert report.status == "UNSUPPORTED"
    assert not (tmp_path / "out.pdf").exists()


# ---------------------------------------------------------------------------
# 9. Overlapping / adjacent findings on the same line shouldn't corrupt each
#    other's redaction geometry.
# ---------------------------------------------------------------------------


def test_multiple_structural_findings_same_line(tmp_path):
    path = tmp_path / "sameline.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text(
        (60, 90), f"Sort code: {SORT_CODE}  Account number: {ACCOUNT_NUMBER}", fontsize=10
    )
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "sameline")
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert structural == {"sort_code", "account_number"}


# ---------------------------------------------------------------------------
# 10. Negative/tight kerning: value glyphs positioned with slight backward
#     overlap (common with condensed fonts or bad PDF generators) rather
#     than clean forward spacing.
# ---------------------------------------------------------------------------


def test_tight_negative_kerning_sort_code(tmp_path):
    path = tmp_path / "kerned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), "Sort code:", fontsize=10)
    # Slight negative overlap between successive glyph groups.
    x = 125
    for chunk in ["12-", "34-", "56"]:
        page.insert_text((x, 90), chunk, fontsize=10)
        x += len(chunk) * 5.4  # tighter than natural glyph advance
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "kerned")
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert "sort_code" in structural


# ---------------------------------------------------------------------------
# 11. A genuine digit collision between a labelled structural value and an
#     unrelated bare number elsewhere on the page: every occurrence of the
#     exact labelled value must be removed, wherever it appears.
# ---------------------------------------------------------------------------


def test_labelled_value_also_appearing_unlabelled_elsewhere_is_fully_removed(tmp_path):
    path = tmp_path / "collision.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    page.insert_text((60, 300), f"Reference {ACCOUNT_NUMBER} confirmed", fontsize=9)
    doc.save(path)
    doc.close()

    out = tmp_path / "collision_out.pdf"
    manifest = tmp_path / "collision_out.n2n.json"
    report = pipeline.run(path, PACK_ID, out, manifest)
    # The engine only redacts the LABELLED occurrence's bbox; an identical
    # value recurring unlabelled elsewhere is exactly what independent
    # verification exists to catch — PASS_AUTO must never happen here.
    assert report.status == "FAILED_VERIFY"
    assert not out.exists()
    assert "account_number" in (report.verification.residual_fields if report.verification else [])


# ---------------------------------------------------------------------------
# 12. Dash-variant separators: a font can substitute an en dash, em dash,
#     minus sign, or other dash-like glyph for a plain hyphen. The value
#     must still be recognized, not silently missed because the separator
#     isn't ASCII "-".
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dash", ["–", "—", "−", "·"])
def test_sort_code_with_dash_variant_separator(tmp_path, dash):
    path = tmp_path / "dashvariant.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"Sort code: 12{dash}34{dash}56", fontsize=10)
    page.insert_text((60, 110), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, f"dashvariant_{ord(dash)}")
    structural = {f.field_id for f in report.findings if f.tier == "structural"}
    assert "sort_code" in structural, f"sort code with separator {dash!r} was not detected"


# ---------------------------------------------------------------------------
# 13. Mixed page sizes within one document, sensitive data on the
#     unusually-sized page.
# ---------------------------------------------------------------------------


def test_mixed_page_sizes_findings_still_caught(tmp_path):
    path = tmp_path / "mixedsize.pdf"
    doc = fitz.open()
    page1 = doc.new_page(width=595, height=842)
    for i in range(20):
        page1.insert_text((60, 60 + i * 15), f"Transaction line {i} with descriptive filler text", fontsize=9)
    page2 = doc.new_page(width=300, height=300)
    page2.insert_text((20, 20), "ACME BANK PLC statement continues on this smaller page", fontsize=8)
    page2.insert_text((20, 40), f"Sort code: {SORT_CODE}", fontsize=8)
    page2.insert_text((20, 55), f"Account number: {ACCOUNT_NUMBER}", fontsize=8)
    for i in range(5):
        page2.insert_text((20, 80 + i * 15), f"More filler transaction detail text line {i}", fontsize=8)
    doc.save(path)
    doc.close()

    report = _run(path, tmp_path, "mixedsize")
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert {"sort_code", "account_number"} <= structural
