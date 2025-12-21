from __future__ import annotations

from typing import Dict, List

from api.primitives.card_pan import CardPanConfig, PrimitiveDetection, find_card_pans
from n2n.models import DetectionResult, PiiCategory, TextSpan
from n2n.primitives import register_primitive


def _filter_spans(spans: List[TextSpan], keywords: List[str]) -> List[TextSpan]:
    if not keywords:
        return spans
    filtered: List[TextSpan] = []
    for span in spans:
        text = (span.text or "").lower()
        if any(keyword in text for keyword in keywords):
            filtered.append(span)
    return filtered


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


def _convert_detection(det: PrimitiveDetection, field_id: str, category: PiiCategory) -> DetectionResult:
    return DetectionResult(
        field_id=field_id,
        category=category,
        primitive="card_pan",
        span=TextSpan(
            page_index=det.page,
            text=det.match_text,
            bbox=det.bbox,
            source=det.source,
        ),
        confidence=1.0,
        context=det.context,
        raw_text=det.raw,
        masked_text=det.text,
        source=det.source,
        validators=det.validators,
        severity=det.severity,
    )


@register_primitive("card_pan")
def detect_card_pan(
    spans: List[TextSpan],
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    keywords = [str(k).lower() for k in field_cfg.get("context_keywords", []) if str(k).strip()]
    scoped_spans = _filter_spans(spans, keywords)
    primitive_detections = find_card_pans(scoped_spans, CardPanConfig())
    if not primitive_detections:
        return []

    field_id = str(field_cfg.get("id", "card_pan"))
    category = _resolve_category(field_cfg, PiiCategory.CARD_NUMBERS)

    return [_convert_detection(det, field_id, category) for det in primitive_detections]


__all__ = ["detect_card_pan"]
