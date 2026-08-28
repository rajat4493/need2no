"""Irreversible removal of must-hide content — not visual covering.

Uses PyMuPDF's redaction annotations, which rewrite the page content
stream and strip the underlying text/glyphs (not just draw a box on top),
then strips metadata, form fields, annotations, embedded files, and
flattens the document so no prior incremental-update history survives.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from n2n.models import Finding

REDACT_FILL = (0, 0, 0)


def _skip_ws(data: bytes, i: int) -> int:
    while i < len(data) and data[i : i + 1].isspace():
        i += 1
    return i


def _find_pdf_string_end(data: bytes, start: int) -> int | None:
    """`start` points at the opening delimiter of a PDF string object
    (either `<` for a hex string or `(` for a literal string). Returns the
    index just past its matching closing delimiter, or None if the object
    is malformed. Handles a literal string's backslash escapes and
    (PDF-legal) balanced, unescaped nested parentheses — a plain regex
    can't do this correctly, and getting it wrong is exactly how the
    previous hex-only version of this function silently failed to match
    whenever MuPDF chose to emit an ID as a literal string instead."""
    if data[start : start + 1] == b"<":
        end = data.find(b">", start + 1)
        return None if end == -1 else end + 1
    if data[start : start + 1] == b"(":
        depth = 1
        i = start + 1
        while i < len(data):
            ch = data[i : i + 1]
            if ch == b"\\":
                i += 2
                continue
            if ch == b"(":
                depth += 1
            elif ch == b")":
                depth -= 1
                if depth == 0:
                    return i + 1
            i += 1
        return None
    return None


def _make_id_deterministic(pdf_bytes: bytes) -> bytes:
    """MuPDF stamps a trailer /ID on every save — the first entry is
    normally stable (tied to the input file's own original ID), but the
    second is regenerated fresh and random on every save, by PDF-spec
    design (it's meant to mark "this specific revision"). Left alone,
    that breaks the deterministic-replay guarantee (spec 5.7) for
    otherwise byte-identical redacted content. We replace both entries
    with one derived from the rest of the document's bytes instead —
    parsing them properly (see _find_pdf_string_end) since MuPDF can
    write either a hex `<...>` or literal `(...)` string for each entry,
    not always hex."""
    marker = pdf_bytes.rfind(b"/ID")
    if marker == -1:
        return pdf_bytes
    i = _skip_ws(pdf_bytes, marker + 3)
    if pdf_bytes[i : i + 1] != b"[":
        return pdf_bytes
    i = _skip_ws(pdf_bytes, i + 1)
    first_end = _find_pdf_string_end(pdf_bytes, i)
    if first_end is None:
        return pdf_bytes
    j = _skip_ws(pdf_bytes, first_end)
    second_end = _find_pdf_string_end(pdf_bytes, j)
    if second_end is None:
        return pdf_bytes
    k = _skip_ws(pdf_bytes, second_end)
    if pdf_bytes[k : k + 1] != b"]":
        return pdf_bytes
    array_end = k + 1

    blanked = pdf_bytes[:marker] + pdf_bytes[array_end:]
    digest = hashlib.sha256(blanked).hexdigest().encode("ascii")
    replacement = b"/ID[<" + digest + b"><" + digest + b">]"
    return pdf_bytes[:marker] + replacement + pdf_bytes[array_end:]


def redact_document(input_path: Path, findings_to_remove: list[Finding]) -> bytes:
    doc = fitz.open(input_path)
    try:
        by_page: dict[int, list[Finding]] = {}
        for finding in findings_to_remove:
            by_page.setdefault(finding.page, []).append(finding)

        for page_index, page_findings in by_page.items():
            page = doc[page_index]
            for finding in page_findings:
                rect = fitz.Rect(*finding.bbox)
                rect.x0 -= 1
                rect.y0 -= 1
                rect.x1 += 1
                rect.y1 += 1
                page.add_redact_annot(rect, fill=REDACT_FILL)
            page.apply_redactions(
                images=fitz.PDF_REDACT_IMAGE_REMOVE,
                graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED,
            )

        # Strip metadata.
        doc.set_metadata({})
        doc.set_xml_metadata("")

        # Strip form fields and annotations (redaction annots were already
        # consumed by apply_redactions; this removes anything else, e.g.
        # form widgets or reviewer comments that leaked through).
        for page in doc:
            widget = page.first_widget
            while widget is not None:
                nxt = widget.next
                page.delete_widget(widget)
                widget = nxt
            annot = page.first_annot
            while annot is not None:
                nxt = annot.next
                page.delete_annot(annot)
                annot = nxt

        # Strip embedded files/attachments.
        for name in list(doc.embfile_names()):
            doc.embfile_del(name)

        # Save without incremental update — a fresh, garbage-collected,
        # linearized rewrite flattens any prior incremental history rather
        # than appending to it.
        output_bytes = doc.tobytes(garbage=4, deflate=True, clean=True, incremental=False)
        return _make_id_deterministic(output_bytes)
    finally:
        doc.close()
