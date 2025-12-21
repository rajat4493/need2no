from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

from n2n.models import DetectionResult

EXPIRY_RE = re.compile(r"(0[1-9]|1[0-2])[\-/](\d{2,4})")


@dataclass
class ExpiryDetection:
    text: str
    raw: str
    severity: str
    validators: List[str]


def parse_expiry_from_text(text: str) -> ExpiryDetection | None:
    if not text:
        return None
    match = EXPIRY_RE.search(text)
    if not match:
        return None
    month = int(match.group(1))
    year_raw = match.group(2)
    year = int(year_raw)
    if year < 100:
        year += 2000
    try:
        datetime(year, month, 1)
    except ValueError:
        return None
    now = datetime.utcnow()
    validators = ["format_mm_yy"]
    severity = "hit"
    if year < now.year - 1:
        severity = "suspicion"
        validators.append("expired")
    return ExpiryDetection(
        text=f"{month:02d}/{year % 100:02d}",
        raw=match.group(0),
        severity=severity,
        validators=validators,
    )


def build_detection(field: str, detection: ExpiryDetection, bbox, page: int) -> DetectionResult:
    return DetectionResult(
        field_id=field,
        text=detection.text,
        raw=detection.raw,
        bbox=bbox,
        page=page,
        source="roi_ocr",
        validators=detection.validators,
        severity=detection.severity,
    )


__all__ = ["parse_expiry_from_text", "build_detection"]
