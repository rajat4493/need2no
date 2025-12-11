from __future__ import annotations

from pathlib import Path
from typing import List

import fitz

from n2n.models import DetectionResult
from n2n.renderers._textbbox import extract_word_entries, find_bbox_for_tokens


HIGHLIGHT_SUFFIX = "_highlighted"


def highlight_pdf(input_pdf: Path, detections: List[DetectionResult]) -> Path:
    """Produce a highlighted PDF showing detected spans."""

    output_path = input_pdf.with_name(f"{input_pdf.stem}{HIGHLIGHT_SUFFIX}.pdf")

    if not detections:
        # still copy original to keep workflow predictable
        doc = fitz.open(str(input_pdf))
        doc.save(str(output_path))
        doc.close()
        return output_path

    word_cache = extract_word_entries(input_pdf)
    doc = fitz.open(str(input_pdf))

    for det in detections:
        page_index = det.span.page_index
        if page_index >= len(word_cache):
            continue

        span_text = (det.span.text or "").strip()
        if not span_text:
            continue

        tokens = span_text.split()
        bbox = find_bbox_for_tokens(word_cache[page_index], tokens)
        if not bbox:
            continue

        page = doc[page_index]
        rect = fitz.Rect(*bbox)
        annot = page.add_highlight_annot(rect)
        annot.set_colors(stroke=(1.0, 1.0, 0.0))
        annot.set_opacity(0.4)
        annot.update()

    doc.save(str(output_path))
    doc.close()
    return output_path


__all__ = ["highlight_pdf", "HIGHLIGHT_SUFFIX"]
