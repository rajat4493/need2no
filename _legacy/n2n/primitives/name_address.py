from __future__ import annotations

import re
from typing import Dict, List, Tuple

import pdfplumber

from n2n.models import DetectionResult, ExtractionResult, PiiCategory, TextSpan
from n2n.primitives import register_primitive

CURRENCY_TOKEN_RE = re.compile(
    r"""^[£]?\s*
        -?\d{1,3}
        (,\d{3})*
        (\.\d{2})?
        \s*$""",
    re.VERBOSE,
)
NUMERIC_CHARS = set("0123456789.,-£$()")


def _normalize_region_bbox(region: Dict[str, object], page_width: float, page_height: float) -> Tuple[float, float, float, float]:
    x_range = region.get("x_range", (0.0, 1.0))
    y_range = region.get("y_range", (0.0, 1.0))
    x0 = float(x_range[0]) * page_width
    x1 = float(x_range[1]) * page_width
    y0 = float(y_range[0]) * page_height
    y1 = float(y_range[1]) * page_height
    return (x0, y0, x1, y1)


def _group_words_by_line(words: List[dict]) -> List[List[dict]]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda w: (w["top"], w["x0"]))
    lines: List[List[dict]] = []
    current_line: List[dict] = []
    current_top = None

    for word in sorted_words:
        top = float(word["top"])
        if current_line and current_top is not None and abs(top - current_top) > 2.0:
            lines.append(current_line)
            current_line = [word]
            current_top = top
        else:
            current_line.append(word)
            current_top = top if current_top is None else current_top

    if current_line:
        lines.append(current_line)

    return lines


def _line_text(words: List[dict]) -> str:
    return " ".join((word.get("text") or "").strip() for word in words if word.get("text"))


def _line_bbox(words: List[dict]) -> Tuple[float, float, float, float]:
    x0 = min(float(word["x0"]) for word in words)
    y0 = min(float(word["top"]) for word in words)
    x1 = max(float(word["x1"]) for word in words)
    y1 = max(float(word["bottom"]) for word in words)
    return (x0, y0, x1, y1)


@register_primitive("name_header")
def detect_name_header(
    extraction: ExtractionResult,
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    region_def = field_cfg.get("region_bounds") or field_cfg.get("region_def") or field_cfg.get("region")
    if not region_def:
        return []

    page_index = int(region_def.get("page", 0))
    if page_index >= len(extraction.pages):
        return []

    detections: List[DetectionResult] = []
    field_id = str(field_cfg.get("id", "account_name"))
    category_value = field_cfg.get("category", PiiCategory.CUSTOMER_IDENTITY)
    if isinstance(category_value, str):
        try:
            category = PiiCategory(category_value)
        except ValueError:
            category = PiiCategory.CUSTOMER_IDENTITY
    else:
        category = category_value
    primitive_name = field_cfg.get("primitive", "name_header")

    with pdfplumber.open(str(extraction.file_path)) as pdf:
        if page_index >= len(pdf.pages):
            return []

        page = pdf.pages[page_index]
        bbox = _normalize_region_bbox(region_def, page.width, page.height)
        cropped = page.crop(bbox)
        words = cropped.extract_words() or []
        for line_words in _group_words_by_line(words):
            text = _line_text(line_words).strip()
            if not _is_valid_header_line(text):
                continue

            span_bbox = _line_bbox(line_words)
            detections.append(
                DetectionResult(
                    field_id=field_id,
                    category=category,
                    primitive=primitive_name,
                    span=TextSpan(
                        page_index=page_index,
                        text=text,
                        bbox=span_bbox,
                    ),
                    confidence=1.0,
                    context=text,
                )
            )

            if field_id == "account_name":
                break

    return detections


def _is_valid_header_line(text: str) -> bool:
    if not text:
        return False

    if not any(ch.isalpha() for ch in text):
        return False

    tokens = text.split()
    currency_hits = sum(1 for tok in tokens if _looks_like_currency(tok))
    if currency_hits >= 2:
        return False

    stripped = "".join(ch for ch in text if not ch.isspace())
    if stripped:
        numeric_chars = sum(1 for ch in stripped if ch.isdigit() or ch in NUMERIC_CHARS)
        if numeric_chars / len(stripped) > 0.6:
            return False

    return True


def _looks_like_currency(token: str) -> bool:
    return bool(CURRENCY_TOKEN_RE.match(token.strip()))


__all__ = ["detect_name_header"]
