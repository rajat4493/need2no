"""Primary extraction path (used for detection + transform), built on PyMuPDF.

This module is deliberately NOT reused by n2n/verify.py — verification must
walk a separate code path (see verify.py, which uses pdfplumber) so a bug
shared between "what we redacted" and "what we checked" can't hide a leak.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import fitz  # PyMuPDF

from n2n.models import DocumentInfo, TextSpan


@dataclass
class ExtractionResult:
    spans: list[TextSpan]
    document_info: DocumentInfo
    page_heights: list[float]


def extract_native(path: Path) -> ExtractionResult:
    doc = fitz.open(path)
    try:
        spans: list[TextSpan] = []
        page_heights: list[float] = []
        for page_index, page in enumerate(doc):
            page_heights.append(page.rect.height)
            words = page.get_text("words")  # x0, y0, x1, y1, word, block, line, word_no
            for x0, y0, x1, y1, word, *_ in words:
                if word.strip():
                    spans.append(TextSpan(text=word, bbox=(x0, y0, x1, y1), page=page_index))

        has_form_fields = any(page.first_widget is not None for page in doc)
        has_annotations = any(page.first_annot is not None for page in doc)
        has_embedded_files = doc.embfile_count() > 0
        has_incremental_history = _has_incremental_updates(path)

        info = DocumentInfo(
            classification="native_text_pdf",
            page_count=doc.page_count,
            extraction_methods=("native_text", "pdf_metadata_parse"),
            has_form_fields=has_form_fields,
            has_annotations=has_annotations,
            has_incremental_history=has_incremental_history,
            has_embedded_files=has_embedded_files,
        )
        return ExtractionResult(spans=spans, document_info=info, page_heights=page_heights)
    finally:
        doc.close()


def _has_incremental_updates(path: Path) -> bool:
    """A PDF with incremental updates has more than one %%EOF marker."""
    data = path.read_bytes()
    return data.count(b"%%EOF") > 1


def group_spans_into_lines(spans: list[TextSpan], y_tolerance: float = 3.0) -> list[list[TextSpan]]:
    """Group word spans on the same page into reading-order lines, for
    label-proximity detectors (e.g. 'Sort code: 12-34-56')."""
    by_page: dict[int, list[TextSpan]] = {}
    for span in spans:
        by_page.setdefault(span.page, []).append(span)

    lines: list[list[TextSpan]] = []
    for page, page_spans in by_page.items():
        page_spans = sorted(page_spans, key=lambda s: (s.bbox[1], s.bbox[0]))
        current: list[TextSpan] = []
        current_y: float | None = None
        for span in page_spans:
            y = span.bbox[1]
            if current_y is None or abs(y - current_y) <= y_tolerance:
                current.append(span)
                current_y = y if current_y is None else current_y
            else:
                lines.append(sorted(current, key=lambda s: s.bbox[0]))
                current = [span]
                current_y = y
        if current:
            lines.append(sorted(current, key=lambda s: s.bbox[0]))
    return lines
