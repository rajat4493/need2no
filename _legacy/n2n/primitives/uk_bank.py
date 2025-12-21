from __future__ import annotations

import re
from typing import Dict, List, Sequence

from n2n.models import DetectionResult, PiiCategory, TextSpan
from n2n.primitives import register_primitive

SORT_CODE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{8}\b")
IBAN_PATTERN = re.compile(r"\bGB[0-9A-Z]{2}[0-9A-Z ]{10,30}\b")


def _line_has_context(text: str, keywords: Sequence[str]) -> bool:
    if not keywords:
        return False
    lower = text.lower()
    return any(keyword in lower for keyword in keywords)


def _resolve_category(field_cfg: Dict[str, object], default: PiiCategory) -> PiiCategory:
    raw = field_cfg.get("category")
    if isinstance(raw, PiiCategory):
        return raw
    if isinstance(raw, str):
        try:
            return PiiCategory(raw)
        except ValueError:
            return default
    return default


def _build_detection(
    span: TextSpan,
    match_text: str,
    field_id: str,
    category: PiiCategory,
    primitive: str,
) -> DetectionResult:
    return DetectionResult(
        field_id=field_id,
        category=category,
        primitive=primitive,
        span=TextSpan(
            page_index=span.page_index,
            text=match_text,
            bbox=span.bbox,
            source=span.source,
            ocr_confidence=span.ocr_confidence,
        ),
        confidence=1.0,
        context=span.text,
    )


def _pattern_detector(
    spans: List[TextSpan],
    field_cfg: Dict[str, object],
    pattern: re.Pattern[str],
    primitive: str,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    field_id = str(field_cfg.get("id", primitive))
    keywords = [str(k).lower() for k in field_cfg.get("context_keywords", []) if str(k).strip()]
    category = _resolve_category(field_cfg, PiiCategory.BANK_IDENTIFIERS)

    for span in spans:
        text = span.text or ""
        if keywords and not _line_has_context(text, keywords):
            continue
        for match in pattern.finditer(text):
            detections.append(
                _build_detection(
                    span=span,
                    match_text=match.group(),
                    field_id=field_id,
                    category=category,
                    primitive=primitive,
                )
            )
    return detections


@register_primitive("uk_sort_code")
def detect_uk_sort_code(spans: List[TextSpan], field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _pattern_detector(spans, field_cfg, SORT_CODE_PATTERN, "uk_sort_code")


@register_primitive("uk_account_number_8d")
def detect_uk_account_number_8d(spans: List[TextSpan], field_cfg: Dict[str, object]) -> List[DetectionResult]:
    detections = _pattern_detector(spans, field_cfg, ACCOUNT_NUMBER_PATTERN, "uk_account_number_8d")
    filtered: List[DetectionResult] = []
    for det in detections:
        text = det.span.text.strip()
        if any(ch in text for ch in ".,£$"):
            continue
        filtered.append(det)
    return filtered


@register_primitive("iban_gb")
def detect_iban_gb(spans: List[TextSpan], field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _pattern_detector(spans, field_cfg, IBAN_PATTERN, "iban_gb")


__all__ = [
    "detect_uk_sort_code",
    "detect_uk_account_number_8d",
    "detect_iban_gb",
]
