"""Irreversible removal of must-hide content — not visual covering.

Uses PyMuPDF's redaction annotations, which rewrite the page content
stream and strip the underlying text/glyphs (not just draw a box on top),
then strips metadata, form fields, annotations, embedded files, and
flattens the document so no prior incremental-update history survives.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import fitz  # PyMuPDF

from n2n.models import Finding

REDACT_FILL = (0, 0, 0)

# MuPDF stamps a randomized trailer /ID on every save, which would make
# byte-identical output impossible even for identical redacted content —
# breaking the deterministic-replay guarantee (spec 5.7). We replace it
# with an ID derived from the rest of the document's bytes instead.
_ID_RE = re.compile(rb"/ID\s*\[\s*<([0-9A-Fa-f]*)>\s*<([0-9A-Fa-f]*)>\s*\]")


def _make_id_deterministic(pdf_bytes: bytes) -> bytes:
    match = _ID_RE.search(pdf_bytes)
    if not match:
        return pdf_bytes
    blanked = pdf_bytes[: match.start()] + pdf_bytes[match.end() :]
    digest = hashlib.sha256(blanked).hexdigest().encode("ascii")
    replacement = b"/ID[<" + digest + b"><" + digest + b">]"
    return pdf_bytes[: match.start()] + replacement + pdf_bytes[match.end() :]


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
