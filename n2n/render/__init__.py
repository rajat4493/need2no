from __future__ import annotations

from pathlib import Path
from typing import List

import fitz

from n2n.models import DetectionResult


def render_highlight(input_pdf: str | Path, detections: List[DetectionResult], out_pdf: str | Path) -> str:
    input_path = Path(input_pdf)
    out_path = Path(out_pdf)

    doc = fitz.open(str(input_path))
    try:
        for det in detections:
            page_index = det.page
            if page_index < 0 or page_index >= len(doc):
                continue
            page = doc[page_index]
            bbox = det.bbox
            if bbox == (0.0, 0.0, 0.0, 0.0) or bbox is None:
                continue
            rect = fitz.Rect(*bbox)
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(1, 1, 0))
            annot.set_opacity(0.35)
            annot.update()
        doc.save(str(out_path))
    finally:
        doc.close()

    return str(out_path)


def render_redact(input_pdf: str | Path, detections: List[DetectionResult], out_pdf: str | Path) -> str:
    input_path = Path(input_pdf)
    out_path = Path(out_pdf)

    doc = fitz.open(str(input_path))
    try:
        for det in detections:
            page_index = det.page
            if page_index < 0 or page_index >= len(doc):
                continue
            bbox = det.bbox
            if bbox == (0.0, 0.0, 0.0, 0.0) or bbox is None:
                continue
            page = doc[page_index]
            rect = fitz.Rect(*bbox)
            page.add_redact_annot(rect, fill=(0, 0, 0))

        for page in doc:
            page.apply_redactions()

        doc.save(str(out_path))
    finally:
        doc.close()

    return str(out_path)


__all__ = ["render_highlight", "render_redact"]
