"""Closes a previously documented, unfixed known gap (see README's
adversarial-testing section): a font with a broken or missing ToUnicode
CMap, where extracted text doesn't reliably correspond to what's
rendered — the literal bug class behind the Epstein-files and Meta v. FTC
redaction failures cited in the build spec. Earlier this required a
"deliberately malformed embedded font", which wasn't feasible to build
with PyMuPDF's standard text insertion; pikepdf (direct PDF-structure
editing) and fontTools (a synthetic minimal font, built fully in-process
so this test has no external file dependency) make an actual repro
possible.
"""

from __future__ import annotations

import io
from pathlib import Path

import fitz
import pikepdf
import pytest
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen

from n2n import pipeline
from n2n.font_trust import check_font_trust

PACK_ID = "uk.bank_statement.share_with_ai"


def _build_minimal_ttf() -> bytes:
    fb = FontBuilder(1000, isTTF=True)
    glyph_order = [".notdef", "A", "B"]
    fb.setupGlyphOrder(glyph_order)
    fb.setupCharacterMap({65: "A", 66: "B"})

    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((0, 500))
    pen.lineTo((500, 500))
    pen.lineTo((500, 0))
    pen.closePath()
    glyph = pen.glyph()
    glyphs = {".notdef": TTGlyphPen(None).glyph(), "A": glyph, "B": glyph}

    fb.setupGlyf(glyphs)
    fb.setupHorizontalMetrics({name: (600, 0) for name in glyph_order})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupNameTable({"familyName": "TestFont", "styleName": "Regular"})
    fb.setupOS2()
    fb.setupPost()
    buf = io.BytesIO()
    fb.save(buf)
    return buf.getvalue()


@pytest.fixture(scope="module")
def minimal_ttf_path(tmp_path_factory) -> Path:
    path = tmp_path_factory.mktemp("fonts") / "minimal.ttf"
    path.write_bytes(_build_minimal_ttf())
    return path


def _embed_font_pdf(path: Path, ttf_path: Path, text: str = "AB") -> None:
    doc = fitz.open()
    page = doc.new_page()
    y = 60
    # The minimal test font only has glyphs for "A"/"B", but preflight's
    # sparse-text guard just counts characters — repeating the same two
    # characters is enough to clear it without needing a richer font.
    for _ in range(10):
        page.insert_text((60, y), text * 20, fontsize=10, fontname="F0", fontfile=str(ttf_path))
        y += 14
    doc.save(path)
    doc.close()


# ---------------------------------------------------------------------------
# Unit: check_font_trust
# ---------------------------------------------------------------------------


def test_standard_font_document_has_no_issues(clean_statement_pdf):
    # clean_statement_pdf (tests/conftest.py) uses PyMuPDF's default
    # base-14 font — no embedding, no trust risk.
    assert check_font_trust(clean_statement_pdf) == []


def test_well_formed_embedded_font_has_no_issues(tmp_path, minimal_ttf_path):
    path = tmp_path / "embedded_ok.pdf"
    _embed_font_pdf(path, minimal_ttf_path)
    assert check_font_trust(path) == []


def test_font_with_stripped_tounicode_is_flagged(tmp_path, minimal_ttf_path):
    path = tmp_path / "embedded.pdf"
    _embed_font_pdf(path, minimal_ttf_path)

    stripped = tmp_path / "no_tounicode.pdf"
    with pikepdf.open(path) as pdf:
        font = pdf.pages[0]["/Resources"]["/Font"]["/F0"]
        assert "/ToUnicode" in font  # PyMuPDF embeds one by default
        del font["/ToUnicode"]
        pdf.save(stripped)

    issues = check_font_trust(stripped)
    assert len(issues) == 1
    assert "ToUnicode" in issues[0].reason


def test_font_with_corrupted_program_is_flagged(tmp_path, minimal_ttf_path):
    path = tmp_path / "embedded.pdf"
    _embed_font_pdf(path, minimal_ttf_path)

    corrupted = tmp_path / "corrupt_fontfile.pdf"
    with pikepdf.open(path) as pdf:
        font = pdf.pages[0]["/Resources"]["/Font"]["/F0"]
        descriptor = font["/DescendantFonts"][0]["/FontDescriptor"]
        descriptor["/FontFile2"].write(b"not a real font program, truncated garbage")
        pdf.save(corrupted)

    issues = check_font_trust(corrupted)
    assert len(issues) == 1
    assert "failed to parse" in issues[0].reason


# ---------------------------------------------------------------------------
# End-to-end: pipeline routes an untrustworthy-font document to NEEDS_REVIEW,
# never PASS_AUTO, regardless of what the (unreliable) extracted text says.
# ---------------------------------------------------------------------------


def test_pipeline_refuses_document_with_no_tounicode_font(tmp_path, minimal_ttf_path):
    path = tmp_path / "embedded.pdf"
    _embed_font_pdf(path, minimal_ttf_path, text="AB")

    stripped = tmp_path / "no_tounicode.pdf"
    with pikepdf.open(path) as pdf:
        font = pdf.pages[0]["/Resources"]["/Font"]["/F0"]
        del font["/ToUnicode"]
        pdf.save(stripped)

    out = tmp_path / "out.pdf"
    manifest = tmp_path / "out.n2n.json"
    report = pipeline.run(stripped, PACK_ID, out, manifest)

    assert report.status == "NEEDS_REVIEW"
    assert not out.exists()
    assert any("ToUnicode" in r or "cannot be trusted" in r for r in report.reasons)
