from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import fitz
import pdfplumber
from PIL import Image
import pytesseract
from pytesseract import Output

from n2n.models import TextSpan

ZOOM = 300 / 72
LINE_MERGE_TOLERANCE = 3.0


def extract_spans(
    input_file: str, ocr: bool = True, artifact_dir: str | Path | None = None
) -> Tuple[List[TextSpan], Dict[str, str], Dict[str, object]]:
    path = Path(input_file).expanduser().resolve()
    artifact_root = Path(artifact_dir).resolve() if artifact_dir else path.parent
    if artifact_dir:
        artifact_root.mkdir(parents=True, exist_ok=True)
    spans = _extract_pdf_text(path) if path.suffix.lower() == ".pdf" else []

    used_ocr = False
    ocr_text_parts: List[str] = []
    ocr_spans: List[TextSpan] = []

    if (not spans or _total_chars(spans) < 20) and ocr:
        used_ocr = True
        ocr_spans, text_blob = _run_ocr(path)
        spans = ocr_spans
        ocr_text_parts.append(text_blob)

    artifacts: Dict[str, str] = {}
    if used_ocr and ocr_spans:
        text_path = artifact_root / f"{path.stem}_ocr_text.txt"
        spans_path = artifact_root / f"{path.stem}_ocr_spans.json"
        text_path.write_text("\n".join(ocr_text_parts), encoding="utf-8")
        spans_payload = [
            {
                "text": span.text,
                "bbox": span.bbox,
                "page": span.page,
                "source": span.source,
                "ocr_conf": span.ocr_conf,
            }
            for span in ocr_spans
        ]
        spans_path.write_text(json.dumps(spans_payload, indent=2), encoding="utf-8")
        artifacts["ocr_text"] = str(text_path)
        artifacts["ocr_spans"] = str(spans_path)

    stats = {
        "used_ocr": used_ocr,
        "span_count": len(spans),
        "text_span_count": sum(1 for span in spans if (span.source or "") == "text"),
        "ocr_span_count": sum(1 for span in spans if (span.source or "").lower() == "ocr"),
    }

    return spans, artifacts, stats


def _extract_pdf_text(path: Path) -> List[TextSpan]:
    spans: List[TextSpan] = []
    if not path.exists():
        return spans

    with pdfplumber.open(str(path)) as pdf:
        for page_idx, page in enumerate(pdf.pages):
            words = page.extract_words() or []
            if words:
                spans.extend(_word_spans(words, page_idx))
                spans.extend(_line_spans(words, page_idx))
                continue

            text_content = (page.extract_text() or "").strip()
            for line in text_content.splitlines():
                line_text = line.strip()
                if line_text:
                    spans.append(
                        TextSpan(text=line_text, bbox=(0.0, 0.0, 0.0, 0.0), page=page_idx, source="text")
                    )
    return spans


def _word_spans(words: Sequence[dict], page_idx: int) -> List[TextSpan]:
    spans: List[TextSpan] = []
    for word in words:
        text = (word.get("text") or "").strip()
        if not text:
            continue
        bbox = (
            float(word.get("x0", 0.0)),
            float(word.get("top", 0.0)),
            float(word.get("x1", 0.0)),
            float(word.get("bottom", 0.0)),
        )
        spans.append(TextSpan(text=text, bbox=bbox, page=page_idx, source="text"))
    return spans


def _line_spans(words: Sequence[dict], page_idx: int) -> List[TextSpan]:
    spans: List[TextSpan] = []
    sorted_words = sorted(
        [
            {
                "text": (word.get("text") or "").strip(),
                "x0": float(word.get("x0", 0.0)),
                "x1": float(word.get("x1", 0.0)),
                "top": float(word.get("top", 0.0)),
                "bottom": float(word.get("bottom", 0.0)),
            }
            for word in words
            if (word.get("text") or "").strip()
        ],
        key=lambda w: (w["top"], w["x0"]),
    )
    if not sorted_words:
        return spans

    current_line = {
        "words": [],
        "x0": None,
        "x1": None,
        "top": None,
        "bottom": None,
    }

    def flush_line():
        if not current_line["words"]:
            return
        line_text = " ".join(entry["text"] for entry in current_line["words"])
        spans.append(
            TextSpan(
                text=line_text,
                bbox=(
                    current_line["x0"] or 0.0,
                    current_line["top"] or 0.0,
                    current_line["x1"] or 0.0,
                    current_line["bottom"] or 0.0,
                ),
                page=page_idx,
                source="text",
            )
        )
        current_line["words"] = []
        current_line["x0"] = None
        current_line["x1"] = None
        current_line["top"] = None
        current_line["bottom"] = None

    for word in sorted_words:
        if not current_line["words"]:
            current_line["words"].append(word)
            current_line["x0"] = word["x0"]
            current_line["x1"] = word["x1"]
            current_line["top"] = word["top"]
            current_line["bottom"] = word["bottom"]
            continue

        if abs(word["top"] - (current_line["top"] or 0.0)) <= LINE_MERGE_TOLERANCE:
            current_line["words"].append(word)
            current_line["x0"] = min(current_line["x0"], word["x0"])
            current_line["x1"] = max(current_line["x1"], word["x1"])
            current_line["top"] = min(current_line["top"], word["top"])
            current_line["bottom"] = max(current_line["bottom"], word["bottom"])
        else:
            flush_line()
            current_line["words"].append(word)
            current_line["x0"] = word["x0"]
            current_line["x1"] = word["x1"]
            current_line["top"] = word["top"]
            current_line["bottom"] = word["bottom"]

    flush_line()
    return spans


def _run_ocr(path: Path) -> Tuple[List[TextSpan], str]:
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        image = Image.open(str(path))
        spans, text_blob = _ocr_image(image, page_index=0, scale=1.0)
        return spans, text_blob

    doc = fitz.open(str(path))
    spans: List[TextSpan] = []
    texts: List[str] = []
    try:
        mat = fitz.Matrix(ZOOM, ZOOM)
        for page_index, page in enumerate(doc):
            pix = page.get_pixmap(matrix=mat)
            image = Image.frombytes("RGB" if pix.alpha == 0 else "RGBA", [pix.width, pix.height], pix.samples)
            page_spans, page_text = _ocr_image(image, page_index, scale=1 / ZOOM)
            spans.extend(page_spans)
            texts.append(page_text)
    finally:
        doc.close()
    return spans, "\n".join(texts)


def _ocr_image(image: Image.Image, page_index: int, scale: float) -> Tuple[List[TextSpan], str]:
    data = pytesseract.image_to_data(image, output_type=Output.DICT)
    spans: List[TextSpan] = []
    words: List[str] = []
    for idx, text in enumerate(data.get("text", [])):
        cleaned = (text or "").strip()
        if not cleaned:
            continue
        try:
            x = float(data.get("left", [0])[idx])
            y = float(data.get("top", [0])[idx])
            w = float(data.get("width", [0])[idx])
            h = float(data.get("height", [0])[idx])
            conf_raw = data.get("conf", [0])[idx]
            conf = float(conf_raw) / 100.0 if conf_raw not in {"", None} else 0.0
        except (ValueError, TypeError):
            continue
        bbox = (x * scale, y * scale, (x + w) * scale, (y + h) * scale)
        spans.append(TextSpan(text=cleaned, bbox=bbox, page=page_index, source="ocr", ocr_conf=conf))
        words.append(cleaned)
    return spans, " ".join(words)


def _total_chars(spans: List[TextSpan]) -> int:
    return sum(len(span.text.strip()) for span in spans)


__all__ = ["extract_spans"]
