from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import fitz  # PyMuPDF
import pdfplumber

from n2n.models import DetectionResult

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


def _resolve_bboxes_for_detection(words: Sequence[dict], detection: DetectionResult) -> List[BBox]:
    bbox = detection.span.bbox
    if bbox and tuple(bbox) != (0.0, 0.0, 0.0, 0.0):
        return [bbox]

    span_text = (detection.span.text or "").strip()
    if not span_text:
        return []

    matches = _find_word_sequences_for_text(words, span_text)
    if matches:
        detection.span.bbox = matches[0]
    return matches


def apply_redactions(
    input_pdf: Path,
    detections: List[DetectionResult],
    output_pdf: Path,
) -> Path:
    """
    Apply real redactions: pdfplumber locates word-level bounding boxes for each
    detection and PyMuPDF draws solid redact annotations so the text is removed.
    """

    if not detections:
        return output_pdf

    doc = fitz.open(str(input_pdf))
    plumber_pdf = pdfplumber.open(str(input_pdf))

    try:
        pages_to_apply = set()

        for det in detections:
            page_index = det.span.page_index
            if page_index < 0 or page_index >= len(plumber_pdf.pages):
                continue

            words = plumber_pdf.pages[page_index].extract_words() or []
            bboxes = _resolve_bboxes_for_detection(words, det)
            if not bboxes:
                continue

            page = doc[page_index]
            for bbox in bboxes:
                rect = fitz.Rect(*bbox)
                page.add_redact_annot(rect, fill=(0, 0, 0))
                pages_to_apply.add(page_index)

        for page_index in pages_to_apply:
            doc[page_index].apply_redactions()

        doc.save(str(output_pdf))
    finally:
        doc.close()
        plumber_pdf.close()

    return output_pdf


__all__ = ["apply_redactions"]
