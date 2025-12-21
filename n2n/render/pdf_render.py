from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Tuple

import fitz
from PIL import Image


@dataclass
class RenderBox:
    page: int
    bbox: Tuple[float, float, float, float]
    label: str = ""
    color: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    page_scale: float = 1.0


def render_highlight_from_boxes(
    input_path: str | Path,
    boxes: Sequence[RenderBox] | None,
    out_pdf: str | Path,
) -> str:
    doc, created = _load_document(Path(input_path))
    try:
        _draw_highlights(doc, boxes or [])
        doc.save(str(out_pdf))
    finally:
        if created:
            doc.close()
    return str(out_pdf)


def render_redact_from_boxes(
    input_path: str | Path,
    boxes: Sequence[RenderBox] | None,
    out_pdf: str | Path,
) -> str:
    doc, created = _load_document(Path(input_path))
    try:
        _apply_redactions(doc, boxes or [])
        doc.save(str(out_pdf))
    finally:
        if created:
            doc.close()
    return str(out_pdf)


def _draw_highlights(doc: fitz.Document, boxes: Sequence[RenderBox]) -> None:
    for box in boxes:
        if box.page < 0 or box.page >= len(doc):
            continue
        rect = _resolve_rect(doc, box)
        page = doc[box.page]
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=box.color, width=2.0)
        if box.label:
            page.insert_textbox(rect, box.label, fontsize=8, color=box.color, overlay=True)


def _apply_redactions(doc: fitz.Document, boxes: Sequence[RenderBox]) -> None:
    for box in boxes:
        if box.page < 0 or box.page >= len(doc):
            continue
        rect = _resolve_rect(doc, box)
        page = doc[box.page]
        shape = page.new_shape()
        shape.draw_rect(rect)
        shape.finish(color=None, fill=(0.0, 0.0, 0.0))


def _resolve_rect(doc: fitz.Document, box: RenderBox) -> fitz.Rect:
    if doc.is_pdf:
        scale = box.page_scale or 1.0
        coords = [coord / scale for coord in box.bbox]
    else:
        coords = box.bbox
    return fitz.Rect(*coords)


def _load_document(path: Path) -> tuple[fitz.Document, bool]:
    if path.suffix.lower() == ".pdf" and path.exists():
        return fitz.open(str(path)), False
    image = Image.open(str(path))
    width, height = image.size
    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(page.rect, filename=str(path))
    return doc, True


__all__ = ["RenderBox", "render_highlight_from_boxes", "render_redact_from_boxes"]
