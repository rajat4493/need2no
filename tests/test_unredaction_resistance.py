"""Tests N2N's output against the actual techniques used to un-redact
real-world documents — including the DOJ Epstein files release.

Two independent adversaries are used, not just our own assertions:

1. `x-ray` (freelawproject/x-ray) — a third-party, purpose-built
   bad-redaction detector: it finds rectangles, extracts the text
   underneath them, and reports what's readable. This is the same class
   of tool that demonstrated the Epstein redaction failures publicly.

2. A deep content-stream decoder (below) that decodes every text-showing
   operand in every PDF stream, *including hex-encoded strings*. This
   matters: PyMuPDF writes page text as hex (`<536f7274...>`), so a naive
   search of the raw file bytes for an ASCII string finds nothing even in
   a completely unredacted document. Every test here therefore runs a
   POSITIVE CONTROL first — a deliberately badly-redacted document that
   the adversary must successfully crack — so a passing result on N2N's
   output proves the tool actually works, rather than proving the test is
   blind.

The documented real-world failure mode is: a black rectangle drawn *over*
text, with the text never removed. Select-and-copy, or any of the tools
above, recovers it instantly.
"""

from __future__ import annotations

import re
from pathlib import Path

import pikepdf
import pymupdf as fitz
import pytest
import xray

from n2n import pipeline

PACK_ID = "uk.bank_statement.share_with_ai"
SORT_CODE = "12-34-56"
ACCOUNT_NUMBER = "99887766"


def _deep_extract_stream_text(path) -> str:
    """Decode every text-showing operand in every content stream,
    including hex strings and literal strings — the ground truth of what
    text the file still physically contains, independent of any
    extraction API's own interpretation."""
    found: list[str] = []
    with pikepdf.open(path) as pdf:
        for obj in pdf.objects:
            try:
                if not isinstance(obj, pikepdf.Stream):
                    continue
                data = bytes(obj.read_bytes())
            except Exception:
                continue
            for hexstr in re.findall(rb"<([0-9A-Fa-f\s]+)>", data):
                cleaned = re.sub(rb"\s", b"", hexstr)
                if len(cleaned) % 2:
                    continue
                try:
                    found.append(bytes.fromhex(cleaned.decode()).decode("latin-1"))
                except Exception:
                    pass
            for lit in re.findall(rb"\(((?:[^()\\]|\\.)*)\)", data):
                found.append(lit.decode("latin-1"))
    return "\n".join(found)


def _write_statement(path, *, cover_with_black_boxes: bool = False) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=12)
    page.insert_text((60, 90), f"Sort code: {SORT_CODE}", fontsize=10)
    page.insert_text((60, 110), f"Account number: {ACCOUNT_NUMBER}", fontsize=10)
    if cover_with_black_boxes:
        # The documented real-world failure: an opaque rectangle drawn
        # over the text, with the text left fully intact underneath.
        page.draw_rect(fitz.Rect(107, 78, 150, 94), color=(0, 0, 0), fill=(0, 0, 0))
        page.draw_rect(fitz.Rect(137, 98, 185, 114), color=(0, 0, 0), fill=(0, 0, 0))
    doc.save(path, deflate=True)
    doc.close()


# ---------------------------------------------------------------------------
# Positive controls — prove each adversary actually works before trusting a
# clean result from it.
# ---------------------------------------------------------------------------


def test_positive_control_xray_cracks_a_black_box_redaction(tmp_path):
    bad = tmp_path / "bad_redaction.pdf"
    _write_statement(bad, cover_with_black_boxes=True)

    findings = xray.inspect(str(bad))
    recovered = " ".join(
        item["text"] for page_items in findings.values() for item in page_items
    )
    assert findings, "x-ray failed to detect a known-bad redaction; adversary is not working"
    assert SORT_CODE in recovered or ACCOUNT_NUMBER in recovered


def test_positive_control_deep_extract_recovers_covered_text(tmp_path):
    bad = tmp_path / "bad_redaction.pdf"
    _write_statement(bad, cover_with_black_boxes=True)

    text = _deep_extract_stream_text(bad)
    assert SORT_CODE in text, "deep extractor is blind; test methodology invalid"
    assert ACCOUNT_NUMBER in text


def test_positive_control_plain_text_extraction_recovers_covered_text(tmp_path):
    """The simplest attack of all, and the one actually used on the
    Epstein files: select the text under the box and copy it."""
    bad = tmp_path / "bad_redaction.pdf"
    _write_statement(bad, cover_with_black_boxes=True)

    doc = fitz.open(bad)
    text = doc[0].get_text("text")
    doc.close()
    assert SORT_CODE in text
    assert ACCOUNT_NUMBER in text


# ---------------------------------------------------------------------------
# N2N's output against each adversary.
# ---------------------------------------------------------------------------


@pytest.fixture
def n2n_output(tmp_path):
    src = tmp_path / "source.pdf"
    _write_statement(src)
    out = tmp_path / "certified.pdf"
    report = pipeline.run(src, PACK_ID, out, tmp_path / "certified.n2n.json")
    assert report.status == "PASS_AUTO"
    return out


def test_xray_finds_no_bad_redaction_in_n2n_output(n2n_output):
    """The third-party detector that demonstrates the Epstein-class
    failure finds nothing to recover in a certified N2N output."""
    assert xray.inspect(str(n2n_output)) == {}


def test_deep_stream_decode_finds_no_sensitive_values(n2n_output):
    text = _deep_extract_stream_text(n2n_output)
    assert SORT_CODE not in text
    assert ACCOUNT_NUMBER not in text
    # The non-sensitive label survives — proving the document wasn't just
    # blanked wholesale, and that the extractor is still reading content.
    assert "Sort code" in text


def test_copy_paste_attack_recovers_nothing(n2n_output):
    doc = fitz.open(n2n_output)
    text = "\n".join(p.get_text("text") for p in doc)
    doc.close()
    assert SORT_CODE not in text
    assert ACCOUNT_NUMBER not in text


def test_no_sensitive_data_in_metadata_or_xmp(n2n_output):
    with pikepdf.open(n2n_output) as pdf:
        meta = str(dict(pdf.docinfo)) if pdf.docinfo is not None else ""
        try:
            xmp = str(pdf.open_metadata())
        except Exception:
            xmp = ""
    assert SORT_CODE not in meta and SORT_CODE not in xmp
    assert ACCOUNT_NUMBER not in meta and ACCOUNT_NUMBER not in xmp


def test_no_recoverable_prior_version_in_the_file(n2n_output):
    """A PDF can carry its own edit history (incremental updates), so an
    unredacted earlier revision can sometimes be recovered from inside
    the released file. N2N saves a flattened, garbage-collected rewrite;
    a single %%EOF confirms no prior revision is retained."""
    raw = n2n_output.read_bytes()
    assert raw.count(b"%%EOF") == 1


def test_certified_output_has_no_annotations_or_form_fields(n2n_output):
    """Annotations/form fields are another documented leak channel — a
    redaction annotation that was never applied, or a form field still
    holding its original value."""
    doc = fitz.open(n2n_output)
    try:
        for page in doc:
            assert page.first_annot is None
            assert page.first_widget is None
    finally:
        doc.close()


# ---------------------------------------------------------------------------
# PETS 2023 glyph-width side channel — "Story Beyond the Eye: Glyph
# Positions Break PDF Text Redaction". Even with the text genuinely
# excised, a redaction box whose width tracks the removed glyphs leaks
# the sum of their advance widths. In a proportional-figure font that was
# measured, before the fix, to narrow 10^8 candidate account numbers to
# exactly one for values like 11111111.
# ---------------------------------------------------------------------------

PROPORTIONAL_DIGIT_FONT = (
    "/mnt/skills/examples/canvas-design/canvas-fonts/ArsenalSC-Regular.ttf"
)


def _redaction_box_widths(path) -> list[float]:
    import re as _re

    with pikepdf.open(path) as pdf:
        for obj in pdf.objects:
            if not isinstance(obj, pikepdf.Stream):
                continue
            data = bytes(obj.read_bytes()).decode("latin-1")
            if " re" in data:
                return [
                    round(float(m[2]), 3)
                    for m in _re.findall(r"([\d.]+) ([\d.]+) ([\d.]+) ([\d.]+) re", data)
                ]
    return []


def _statement_in_font(path, account, font_path, size=11):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "ACME BANK PLC", fontsize=size + 1, fontname="AR", fontfile=font_path)
    page.insert_text((60, 90), "Sort code: 12-34-56", fontsize=size, fontname="AR", fontfile=font_path)
    page.insert_text((60, 115), f"Account number: {account}", fontsize=size, fontname="AR", fontfile=font_path)
    page.insert_text((60, 140), "Statement of account for the period shown", fontsize=size, fontname="AR", fontfile=font_path)
    doc.save(path)
    doc.close()


@pytest.mark.skipif(
    not Path(PROPORTIONAL_DIGIT_FONT).exists(),
    reason="proportional-digit font fixture not available in this environment",
)
@pytest.mark.parametrize("size", [9, 11, 14])
def test_redaction_box_width_does_not_vary_with_the_redacted_value(tmp_path, size):
    """The box width must be identical for every account number. If it
    varies, an attacker measuring it recovers the sum of the digits'
    advance widths and can narrow — sometimes uniquely determine — the
    value, without ever recovering a single character of text."""
    widths = set()
    for account in ["11111111", "00000000", "99999999", "10101010", "12345678"]:
        src = tmp_path / f"src_{account}_{size}.pdf"
        _statement_in_font(src, account, PROPORTIONAL_DIGIT_FONT, size)
        out = tmp_path / f"out_{account}_{size}.pdf"
        report = pipeline.run(src, PACK_ID, out, tmp_path / f"m_{account}_{size}.json")
        if report.status != "PASS_AUTO":
            continue
        boxes = _redaction_box_widths(out)
        assert boxes, "expected at least one drawn redaction box"
        widths.update(boxes)
    assert len(widths) == 1, (
        f"redaction box width varies with the redacted value {sorted(widths)} — "
        "this is the PETS 2023 glyph-width side channel"
    )


def test_drawn_box_never_exceeds_the_excised_region(tmp_path):
    """The constant-width box must stay inside the area whose text was
    actually removed. A box wider than the excision would visually cover
    text that is still present and extractable — recreating the exact
    'black box over live text' failure this product exists to prevent."""
    from n2n.transform import REDACTION_BOX_WIDTH, _fixed_width_box

    narrow = fitz.Rect(10, 10, 10 + (REDACTION_BOX_WIDTH / 2), 20)
    drawn = _fixed_width_box(narrow)
    assert drawn.x1 <= narrow.x1

    wide = fitz.Rect(10, 10, 10 + (REDACTION_BOX_WIDTH * 4), 20)
    drawn_wide = _fixed_width_box(wide)
    assert drawn_wide.x1 <= wide.x1
    assert round(drawn_wide.x1 - drawn_wide.x0, 6) == REDACTION_BOX_WIDTH
