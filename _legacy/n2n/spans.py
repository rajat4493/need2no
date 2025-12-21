from __future__ import annotations

from typing import List

from n2n.models import ExtractionResult, TextSpan


def build_text_spans(extraction: ExtractionResult) -> List[TextSpan]:
    spans: List[TextSpan] = []
    span_source = (extraction.source or "text").lower()
    default_conf = extraction.quality_score if span_source == "ocr" else 1.0

    for page_index, page_text in enumerate(extraction.pages):
        if not page_text:
            continue
        for line in page_text.splitlines():
            text = line.strip()
            if not text:
                continue
            spans.append(
                TextSpan(
                    page_index=page_index,
                    text=text,
                    bbox=(0.0, 0.0, 0.0, 0.0),
                    source=span_source,
                    ocr_confidence=default_conf,
                )
            )
    extraction.spans = spans
    return spans


__all__ = ["build_text_spans"]
