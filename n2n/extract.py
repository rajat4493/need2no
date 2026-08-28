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
            for x0, y0, x1, y1, word, block_no, *_ in words:
                if word.strip():
                    spans.append(
                        TextSpan(text=word, bbox=(x0, y0, x1, y1), page=page_index, block=block_no)
                    )

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
                lines.extend(_split_overlapping_layers(current))
                current = [span]
                current_y = y
        if current:
            lines.extend(_split_overlapping_layers(current))
    return lines


def _split_overlapping_layers(bucket: list[TextSpan]) -> list[list[TextSpan]]:
    """A y-tolerance bucket normally IS one reading-order line. But two
    independent text layers stacked at the same coordinates — e.g. a
    visible line with an invisible OCR duplicate drawn directly under it —
    land in the same bucket too, and interleaving their tokens by x would
    corrupt both (a label from one layer next to a value from the other,
    so neither's label-proximity check matches). Detect that by looking
    for spans from different content-stream blocks whose x-ranges
    significantly overlap, and if found, split the bucket per block
    instead of treating it as one line."""
    bucket = sorted(bucket, key=lambda s: s.bbox[0])
    has_cross_block_overlap = False
    for i in range(len(bucket) - 1):
        a, b = bucket[i], bucket[i + 1]
        if a.block == b.block:
            continue
        overlap = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
        narrower_width = min(a.bbox[2] - a.bbox[0], b.bbox[2] - b.bbox[0])
        if narrower_width > 0 and overlap > 0.5 * narrower_width:
            has_cross_block_overlap = True
            break

    if not has_cross_block_overlap:
        return [bucket]

    by_block: dict[int, list[TextSpan]] = {}
    for span in bucket:
        by_block.setdefault(span.block, []).append(span)
    return [sorted(group, key=lambda s: s.bbox[0]) for group in by_block.values()]
