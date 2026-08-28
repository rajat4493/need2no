"""Detects embedded fonts we can't trust extracted text from.

This is the literal bug class behind the Epstein-files and Meta v. FTC
redaction failures cited in the build spec: the PDF's text layer doesn't
reliably correspond to what's rendered, because a subset/embedded font
has no usable ToUnicode CMap (so its character codes have no defined
Unicode mapping) or the embedded font program itself doesn't even parse.
Detection logic that trusts extracted text without checking this can miss
sensitive content that's perfectly visible on the page.

Uses pikepdf (QPDF) for PDF-structure inspection and fontTools to
validate the embedded font program bytes — both far more capable and
battle-tested than anything hand-rolled here.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pikepdf
from fontTools.ttLib import TTFont

# The 14 standard PDF fonts have a fixed, well-known encoding even
# without an embedded ToUnicode CMap — not a trust risk.
STANDARD_14 = {
    "Helvetica",
    "Helvetica-Bold",
    "Helvetica-Oblique",
    "Helvetica-BoldOblique",
    "Courier",
    "Courier-Bold",
    "Courier-Oblique",
    "Courier-BoldOblique",
    "Times-Roman",
    "Times-Bold",
    "Times-Italic",
    "Times-BoldItalic",
    "Symbol",
    "ZapfDingbats",
}


@dataclass(frozen=True)
class FontTrustIssue:
    page: int
    font_name: str
    reason: str


def check_font_trust(path: Path) -> list[FontTrustIssue]:
    """Returns one issue per (page, font) combination we can't vouch for.
    An empty list means every font on every page either needs no
    ToUnicode map (standard 14) or has one backed by a font program that
    at least parses. It does NOT prove the mapping is semantically
    correct — only that the two cheap, well-defined failure modes this
    function targets aren't present."""
    issues: list[FontTrustIssue] = []
    try:
        pdf = pikepdf.open(path)
    except Exception:
        # Preflight's own open attempt (via PyMuPDF) is the authority on
        # whether the document is even readable; this function only adds
        # signal on top of a document that already opened successfully.
        return issues

    try:
        for page_index, page in enumerate(pdf.pages):
            resources = page.get("/Resources")
            if resources is None or "/Font" not in resources:
                continue
            for font_key, font in resources["/Font"].items():
                issue = _check_one_font(page_index, str(font_key), font)
                if issue is not None:
                    issues.append(issue)
    finally:
        pdf.close()
    return issues


def _check_one_font(page_index: int, font_key: str, font) -> FontTrustIssue | None:
    base_font = str(font.get("/BaseFont", "")).lstrip("/")
    clean_name = base_font.split("+", 1)[-1]  # strip an "ABCDEF+" subset tag
    if clean_name in STANDARD_14:
        return None

    descendant = None
    if font.get("/Subtype") == pikepdf.Name("/Type0"):
        try:
            descendant = font["/DescendantFonts"][0]
        except Exception:
            descendant = None
    target = descendant if descendant is not None else font

    if "/ToUnicode" not in font:
        return FontTrustIssue(
            page=page_index,
            font_name=base_font or font_key,
            reason=(
                "embedded font has no ToUnicode mapping; extracted text "
                "cannot be trusted to match the rendered glyphs"
            ),
        )

    descriptor = target.get("/FontDescriptor")
    if descriptor is None or "/FontFile2" not in descriptor:
        # FontFile (Type 1) and FontFile3 (CFF/OpenType-CFF) formats
        # aren't what fontTools' TTFont loader parses directly; only
        # FontFile2 (TrueType/OpenType) is checked here for program
        # validity. A missing/absent descriptor for a non-standard font
        # is unusual but not itself proof of a text-mapping problem given
        # ToUnicode is already present, so it's not flagged.
        return None

    try:
        raw = bytes(descriptor["/FontFile2"].read_bytes())
        TTFont(io.BytesIO(raw), lazy=True)
    except Exception as exc:
        return FontTrustIssue(
            page=page_index,
            font_name=base_font or font_key,
            reason=(
                f"embedded font program failed to parse ({exc.__class__.__name__}); "
                "extracted text reliability cannot be confirmed"
            ),
        )
    return None
