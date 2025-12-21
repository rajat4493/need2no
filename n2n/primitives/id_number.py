from __future__ import annotations

import re
from typing import List

from n2n.models import DetectionResult

ID_RE = re.compile(r"[A-Z0-9]{6,}")


def detect_id_number(text: str) -> str | None:
    if not text:
        return None
    match = ID_RE.search(text.replace(" ", ""))
    if not match:
        return None
    return match.group(0)


def build_detection(field: str, value: str, bbox, page: int) -> DetectionResult:
    return DetectionResult(
        field_id=field,
        text=value,
        raw=value,
        bbox=bbox,
        page=page,
        source="roi_ocr",
        validators=["pattern_alnum"],
        severity="suspicion",
    )


__all__ = ["detect_id_number", "build_detection"]
