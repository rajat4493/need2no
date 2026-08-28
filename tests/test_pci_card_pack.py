from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from n2n import pipeline
from n2n.detectors.card_expiry import detect_card_expiry
from n2n.detectors.validators import luhn_valid, normalize_card_expiry
from n2n.models import TextSpan

PACK_ID = "pci.card_data.share_with_ai"


def _line(*texts, page=0, y=100.0):
    spans = []
    x = 0.0
    for text in texts:
        spans.append(TextSpan(text=text, bbox=(x, y, x + len(text) * 6, y + 12), page=page))
        x += len(text) * 6 + 4
    return spans


# ---------------------------------------------------------------------------
# Unit: validators
# ---------------------------------------------------------------------------


def test_amex_number_is_luhn_valid():
    assert luhn_valid("3782 822463 10005") is True


def test_card_expiry_rejects_invalid_month():
    assert normalize_card_expiry("13/29") is None
    assert normalize_card_expiry("00/29") is None


def test_card_expiry_accepts_valid_month_and_normalizes():
    assert normalize_card_expiry("03/29") == "03/29"
    assert normalize_card_expiry("03-29") == "03/29"
    assert normalize_card_expiry("0329") == "03/29"


def test_card_expiry_does_not_reject_a_past_date():
    # A statement from 2019 showing a card that expired in 2021 is still
    # real cardholder data — expiry validity is not a freshness check.
    assert normalize_card_expiry("01/21") == "01/21"


# ---------------------------------------------------------------------------
# Unit: detector
# ---------------------------------------------------------------------------


def test_bare_mm_yy_without_label_is_never_flagged():
    line = _line("Statement", "period:", "03/29")
    assert detect_card_expiry([line]) == []


def test_labelled_expiry_is_flagged():
    line = _line("Expiry", "date:", "03/29")
    findings = detect_card_expiry([line])
    assert len(findings) == 1
    assert findings[0].tier == "structural"
    assert findings[0].text == "03/29"


# ---------------------------------------------------------------------------
# End-to-end pipeline: PASS_AUTO / NEEDS_REVIEW
# ---------------------------------------------------------------------------


def _write_card_document(path: Path, *, include_name: bool = False, expiry_sep: str = "/") -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    if include_name:
        page.insert_text((60, y), "Jane Smith", fontsize=14)
        y += 20
    page.insert_text((60, y), "GLOBAL RETAIL LTD", fontsize=12)
    y += 25
    page.insert_text((60, y), "Order confirmation", fontsize=11)
    y += 20
    page.insert_text((60, y), "Card number: 4111 1111 1111 1111", fontsize=10)
    y += 18
    page.insert_text((60, y), f"Expiry date: 03{expiry_sep}29", fontsize=10)
    y += 18
    page.insert_text((60, y), "Amount charged: 49.99 GBP", fontsize=10)
    doc.save(path)
    doc.close()


def test_clean_card_document_reaches_pass_auto(tmp_path):
    path = tmp_path / "receipt.pdf"
    _write_card_document(path)
    out = tmp_path / "out.pdf"
    manifest = tmp_path / "out.n2n.json"
    report = pipeline.run(path, PACK_ID, out, manifest)

    assert report.status == "PASS_AUTO"
    assert out.exists()
    structural = {f.field_id for f in report.findings if f.tier == "structural"}
    assert structural == {"card_number", "card_expiry"}
    assert all(f.action == "removed" for f in report.findings)

    doc = fitz.open(out)
    text = doc[0].get_text("text")
    doc.close()
    assert "4111" not in text
    assert "03" not in text.replace("Card number:", "").replace("GLOBAL", "")  # crude but adequate
    assert "03/29" not in text


def test_card_document_with_name_forces_needs_review(tmp_path):
    path = tmp_path / "receipt_named.pdf"
    _write_card_document(path, include_name=True)
    out = tmp_path / "out.pdf"
    manifest = tmp_path / "out.n2n.json"
    report = pipeline.run(path, PACK_ID, out, manifest)

    assert report.status == "NEEDS_REVIEW"
    assert not out.exists()
    assert any("name_header" in r for r in report.reasons)


def test_deterministic_replay(tmp_path):
    path = tmp_path / "receipt.pdf"
    _write_card_document(path)
    out1, manifest1 = tmp_path / "out1.pdf", tmp_path / "out1.json"
    out2, manifest2 = tmp_path / "out2.pdf", tmp_path / "out2.json"
    r1 = pipeline.run(path, PACK_ID, out1, manifest1)
    r2 = pipeline.run(path, PACK_ID, out2, manifest2)
    assert r1.status == r2.status == "PASS_AUTO"
    assert r1.manifest["output_hash"] == r2.manifest["output_hash"]
    assert out1.read_bytes() == out2.read_bytes()


# ---------------------------------------------------------------------------
# Adversarial: reuse the failure classes found for the bank-statement pack
# ---------------------------------------------------------------------------


def test_amex_format_card_number_detected(tmp_path):
    path = tmp_path / "amex.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "GLOBAL RETAIL LTD", fontsize=12)
    page.insert_text((60, 90), "Card number: 3782 822463 10005", fontsize=10)
    doc.save(path)
    doc.close()

    out = tmp_path / "out.pdf"
    report = pipeline.run(path, PACK_ID, out, tmp_path / "out.json")
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert "card_number" in structural
        doc2 = fitz.open(out)
        assert "3782" not in doc2[0].get_text("text")
        doc2.close()


@pytest.mark.parametrize("sep", ["/", "-", "–"])
def test_expiry_separator_variants(tmp_path, sep):
    path = tmp_path / f"expiry_{ord(sep)}.pdf"
    _write_card_document(path, expiry_sep=sep)
    out = tmp_path / "out.pdf"
    report = pipeline.run(path, PACK_ID, out, tmp_path / "out.json")
    structural = {f.field_id for f in report.findings if f.tier == "structural"}
    assert "card_expiry" in structural, f"expiry with separator {sep!r} was not detected"


def test_split_token_expiry_not_silently_missed(tmp_path):
    path = tmp_path / "split_expiry.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "GLOBAL RETAIL LTD", fontsize=12)
    page.insert_text((60, 90), "Expiry date:", fontsize=10)
    x = 140
    for ch in "03/29":
        page.insert_text((x, 90), ch, fontsize=10)
        x += 7
    doc.save(path)
    doc.close()

    out = tmp_path / "out.pdf"
    report = pipeline.run(path, PACK_ID, out, tmp_path / "out.json")
    if report.status == "PASS_AUTO":
        structural = {f.field_id for f in report.findings if f.tier == "structural"}
        assert "card_expiry" in structural
        doc2 = fitz.open(out)
        assert "29" not in doc2[0].get_text("text").replace("GLOBAL RETAIL LTD", "")
        doc2.close()
