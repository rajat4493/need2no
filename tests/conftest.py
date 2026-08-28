from __future__ import annotations

from pathlib import Path

import pymupdf as fitz
import pytest


def _write_statement(
    path: Path,
    *,
    include_name: bool = True,
    sort_code: str = "12-34-56",
    account_number: str = "12345678",
    iban: str | None = None,
    card_number: str | None = None,
) -> None:
    doc = fitz.open()
    page = doc.new_page()

    y = 60
    if include_name:
        page.insert_text((60, y), "Jane Smith", fontsize=14)
        y += 20
        page.insert_text((60, y), "42 Example Street, London", fontsize=10)
        y += 30

    page.insert_text((60, y), "ACME BANK PLC", fontsize=12)
    y += 30
    page.insert_text((60, y), f"Sort code: {sort_code}", fontsize=10)
    y += 18
    page.insert_text((60, y), f"Account number: {account_number}", fontsize=10)
    y += 18

    if iban:
        page.insert_text((60, y), f"IBAN: {iban}", fontsize=10)
        y += 18

    if card_number:
        page.insert_text((60, y), f"Card: {card_number}", fontsize=10)
        y += 18

    y += 20
    page.insert_text((60, y), "Statement transactions", fontsize=11)
    y += 18
    page.insert_text((60, y), "01 Jan  Reference 87654321  Payment  -25.00", fontsize=9)
    y += 14
    page.insert_text((60, y), "02 Jan  Reference 11223344  Deposit  +100.00", fontsize=9)

    doc.save(path)
    doc.close()


@pytest.fixture
def clean_statement_pdf(tmp_path: Path) -> Path:
    """No free-text name header candidate above the header cutoff and no
    unresolved review-tier fields — should reach PASS_AUTO."""
    path = tmp_path / "clean_statement.pdf"
    _write_statement(
        path,
        include_name=False,
        sort_code="12-34-56",
        account_number="12345678",
        iban="GB29NWBK60161331926819",
        card_number="4111 1111 1111 1111",
    )
    return path


@pytest.fixture
def statement_with_name_pdf(tmp_path: Path) -> Path:
    """Has a header name candidate -> forces NEEDS_REVIEW."""
    path = tmp_path / "statement_with_name.pdf"
    _write_statement(
        path,
        include_name=True,
        sort_code="12-34-56",
        account_number="12345678",
    )
    return path


@pytest.fixture
def unsupported_scanned_pdf(tmp_path: Path) -> Path:
    """A page with an image and no text layer -> UNSUPPORTED."""
    path = tmp_path / "scanned.pdf"
    doc = fitz.open()
    page = doc.new_page()
    pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200))
    pix.set_rect(pix.irect, (200, 200, 200))
    page.insert_image(fitz.Rect(50, 50, 250, 250), pixmap=pix)
    doc.save(path)
    doc.close()
    return path


make_statement_pdf = _write_statement
