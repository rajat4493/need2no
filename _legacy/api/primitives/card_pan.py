from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from n2n.models import TextSpan

PAN_PATTERN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
CONFUSABLE_TRANSLATION = str.maketrans(
    {
        "O": "0",
        "o": "0",
        "I": "1",
        "i": "1",
        "l": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "Z": "2",
        "z": "2",
    }
)


@dataclass
class CardPanConfig:
    ocr_conf_suspicion_threshold: float = 0.75


@dataclass
class PrimitiveDetection:
    field_id: str
    text: str
    raw: str
    page: int
    bbox: Tuple[float, float, float, float]
    source: str
    validators: List[str]
    severity: str
    match_text: str
    context: str


def _normalize(candidate: str) -> str:
    cleaned = candidate.strip()
    alnum = re.sub(r"[^0-9A-Za-z]", "", cleaned)
    letters = sum(1 for ch in alnum if ch.isalpha())
    if letters:
        letter_ratio = letters / max(1, len(alnum))
        if letter_ratio <= 0.25:
            cleaned = cleaned.translate(CONFUSABLE_TRANSLATION)
    return re.sub(r"[^\d]", "", cleaned)


def _luhn_checksum(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for index, char in enumerate(digits):
        value = int(char)
        if index % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask_pan(digits: str) -> str:
    if len(digits) <= 4:
        return digits
    masked = "*" * (len(digits) - 4) + digits[-4:]
    grouped = [masked[i : i + 4] for i in range(0, len(masked), 4)]
    return " ".join(grouped)


def _should_emit_suspicion(span: TextSpan, threshold: float) -> bool:
    if (span.source or "").lower() != "ocr":
        return False
    if span.ocr_confidence is None:
        return False
    return span.ocr_confidence < threshold


def find_card_pans(spans: Sequence[TextSpan], cfg: CardPanConfig | None = None) -> List[PrimitiveDetection]:
    """
    Detect possible PAN values from concrete text spans.
    """

    config = cfg or CardPanConfig()
    threshold = config.ocr_conf_suspicion_threshold
    if threshold <= 0.0:
        threshold = 0.0
    elif threshold >= 1.0:
        threshold = 1.0

    detections: List[PrimitiveDetection] = []

    for span in spans:
        original_text = span.text or ""
        translated = original_text.translate(CONFUSABLE_TRANSLATION)
        for match in PAN_PATTERN.finditer(translated):
            start, end = match.span()
            raw_candidate = original_text[start:end]
            normalized = _normalize(raw_candidate)
            if not (13 <= len(normalized) <= 19):
                continue
            if not normalized.isdigit():
                continue

            passes_luhn = _luhn_checksum(normalized)
            if not passes_luhn and not _should_emit_suspicion(span, threshold):
                continue

            severity = "hit" if passes_luhn else "suspicion"
            validators = ["luhn"] if passes_luhn else []
            detections.append(
                PrimitiveDetection(
                    field_id="card_pan",
                    text=_mask_pan(normalized),
                    raw=normalized,
                    page=span.page_index,
                    bbox=span.bbox,
                    source=span.source or "text",
                    validators=validators,
                    severity=severity,
                    match_text=raw_candidate,
                    context=text.strip(),
                )
            )

    return detections


def detect_card_pan(spans: Sequence[TextSpan]) -> List[PrimitiveDetection]:
    """
    Backwards-compatible alias for older integrations.
    """

    return find_card_pans(spans, CardPanConfig())


__all__ = ["CardPanConfig", "PrimitiveDetection", "find_card_pans", "detect_card_pan"]
