from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import fitz
import numpy as np

from n2n import ENGINE_VERSION
from n2n.io import PageImage, prepare_input_images
from n2n.models import DecisionReason, DecisionReport, DetectionResult, TextSpan
from n2n.render.pdf_render import RenderBox
from n2n.vision.detect import Box, detect_objects, load_yolo_model


@dataclass
class ArtifactPaths:
    input_path: Path
    outdir: Path

    def highlight_path(self) -> Path:
        return self.outdir / f"{self.input_path.stem}_highlight.pdf"

    def redacted_path(self) -> Path:
        return self.outdir / f"{self.input_path.stem}_redacted.pdf"

    def report_path(self) -> Path:
        return self.outdir / f"{self.input_path.stem}_report.json"

    def ocr_text_path(self) -> Path:
        return self.outdir / f"{self.input_path.stem}_ocr_text.txt"

    def ocr_spans_path(self) -> Path:
        return self.outdir / f"{self.input_path.stem}_ocr_spans.json"


@dataclass
class PageContext:
    page: PageImage
    image: np.ndarray
    boxes: List[Box]



def load_page_contexts(
    input_path: Path,
    outdir: Path,
    model_path: Path,
    dpi: int = 350,
) -> Tuple[List[PageContext], Dict[str, object], Dict[str, object]]:
    pages, input_trace = prepare_input_images(input_path, outdir, dpi=dpi)
    contexts: List[PageContext] = []
    model, model_info = load_yolo_model(model_path)
    model_used = bool(model_info.get("model_used"))
    for page in pages:
        image = cv2.imread(str(page.path))
        if image is None:
            raise RuntimeError(f"Failed to load rendered page image: {page.path}")
        detections = detect_objects(image, model) if model else []
        contexts.append(PageContext(page=page, image=image, boxes=detections))
    vision_trace = {
        "weights_path": str(model_path),
        "model_used": model_used,
        "model_reason": model_info.get("reason", ""),
        "pages": [
            {
                "index": ctx.page.index,
                "width": ctx.page.width,
                "height": ctx.page.height,
                "box_count": len(ctx.boxes),
                "boxes": [
                    {
                        "label": box.label,
                        "conf": round(box.conf, 4),
                        "bbox": [round(v, 2) for v in box.as_tuple()],
                    }
                    for box in ctx.boxes
                ],
            }
            for ctx in contexts
        ],
    }
    return contexts, input_trace, vision_trace


def map_bbox_to_pdf_coords(bbox: Tuple[float, float, float, float], page: PageImage) -> Tuple[float, float, float, float]:
    scale = page.scale or 1.0
    if page.source == "image":
        return bbox
    return tuple(coord / scale for coord in bbox)


def build_report(
    pack_id: str,
    decision: str,
    reasons: Sequence[DecisionReason],
    detections: Sequence[DetectionResult],
    artifacts: Dict[str, str | None],
    trace: Dict[str, object],
    *,
    suggested_redactions: Sequence[Dict[str, object]] | None = None,
    action: str | None = None,
) -> DecisionReport:
    return DecisionReport(
        pack_id=pack_id,
        decision=decision,
        reasons=list(reasons),
        detections=list(detections),
        artifacts=artifacts,
        engine_version=ENGINE_VERSION,
        suggested_redactions=list(suggested_redactions or []),
        action=action,
        trace=trace,
    )


def write_report(report: DecisionReport, path: Path) -> None:
    report.artifacts["report_json"] = str(path)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def spans_to_payload(spans: Sequence[TextSpan]) -> List[Dict[str, object]]:
    payload: List[Dict[str, object]] = []
    for span in spans:
        payload.append(
            {
                "text": span.text,
                "bbox": list(span.bbox),
                "page": span.page,
                "source": span.source,
                "ocr_conf": span.ocr_conf,
            }
        )
    return payload


def render_pdf_to_image(pdf_path: Path, page_index: int, dpi: int = 350) -> np.ndarray:
    doc = fitz.open(str(pdf_path))
    try:
        page = doc[page_index]
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat)
        mode = "RGB" if pix.n < 4 else "RGBA"
        array = np.frombuffer(pix.samples, dtype=np.uint8)
        image = array.reshape(pix.height, pix.width, pix.n)
        if pix.n == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image
    finally:
        doc.close()


__all__ = [
    "ArtifactPaths",
    "PageContext",
    "RenderBox",
    "load_page_contexts",
    "map_bbox_to_pdf_coords",
    "build_report",
    "write_report",
    "spans_to_payload",
    "render_pdf_to_image",
]
