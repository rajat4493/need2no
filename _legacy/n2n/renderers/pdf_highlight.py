from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import fitz
import pdfplumber

from n2n.models import DetectionResult

HIGHLIGHT_SUFFIX = "_highlighted"

BBox = Tuple[float, float, float, float]


def _build_bbox_from_words(words: Sequence[dict]) -> BBox:
    x0 = min(float(word["x0"]) for word in words)
    y0 = min(float(word["top"]) for word in words)
    x1 = max(float(word["x1"]) for word in words)
    y1 = max(float(word["bottom"]) for word in words)
    return (x0, y0, x1, y1)


def _find_word_sequences_for_text(words: Sequence[dict], target_text: str) -> List[BBox]:
    target = target_text.strip()
    if not target:
        return []

    clean_words = [word for word in words if (word.get("text") or "").strip()]
    matches: List[BBox] = []
    n = len(clean_words)

    for start in range(n):
        text_parts: List[str] = []
        for end in range(start, n):
            word_text = (clean_words[end].get("text") or "").strip()
            text_parts.append(word_text)
            candidate = " ".join(text_parts).strip()

            if candidate == target:
                matches.append(_build_bbox_from_words(clean_words[start : end + 1]))
                break

            if len(candidate) > len(target):
                break

    return matches


def highlight_pdf(input_pdf: Path, detections: List[DetectionResult]) -> Path:
    """Produce a highlighted PDF showing detected spans using real coordinates."""

    output_path = input_pdf.with_name(f"{input_pdf.stem}{HIGHLIGHT_SUFFIX}.pdf")

    doc = fitz.open(str(input_pdf))
    plumber_pdf = pdfplumber.open(str(input_pdf))

    try:
        if not detections:
            doc.save(str(output_path))
            return output_path

        for det in detections:
            page_index = det.span.page_index
            if page_index < 0 or page_index >= len(plumber_pdf.pages):
                continue

            span_text = (det.span.text or "").strip()
            if not span_text:
                continue

            words = plumber_pdf.pages[page_index].extract_words() or []
            matches = _find_word_sequences_for_text(words, span_text)
            if not matches:
                continue

            page = doc[page_index]
            for bbox in matches:
                rect = fitz.Rect(*bbox)
                annot = page.add_highlight_annot(rect)
                annot.set_colors(stroke=(1.0, 1.0, 0.0))
                annot.set_opacity(0.4)
                annot.update()
                det.span.bbox = bbox

        doc.save(str(output_path))
    finally:
        doc.close()
        plumber_pdf.close()

    return output_path


__all__ = ["highlight_pdf", "HIGHLIGHT_SUFFIX"]
