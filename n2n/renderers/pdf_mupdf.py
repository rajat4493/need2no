from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from n2n.models import DetectionResult
from n2n.renderers._textbbox import extract_word_entries, find_bbox_for_tokens


def apply_redactions(
    input_pdf: Path,
    detections: List[DetectionResult],
    output_pdf: Path,
) -> Path:
    """Redact detections by drawing black rectangles over matching word spans."""

    if not detections:
        return output_pdf

    word_cache = extract_word_entries(input_pdf)
    doc = fitz.open(str(input_pdf))
    pages_to_apply = set()

    for det in detections:
        page_index = det.span.page_index
        if page_index >= len(word_cache):
            continue

        span_text = (det.span.text or "").strip()
        if not span_text:
            continue

        tokens = span_text.split()
        bbox = _find_bbox_for_tokens(word_cache[page_index], tokens)
        if not bbox:
            continue

        page = doc[page_index]
        rect = fitz.Rect(*bbox)
        page.add_redact_annot(rect, fill=(0, 0, 0))
        pages_to_apply.add(page_index)

    for page_index in pages_to_apply:
        doc[page_index].apply_redactions()

    doc.save(str(output_pdf))
    doc.close()
    return output_pdf


__all__ = ["apply_redactions"]
