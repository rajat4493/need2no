from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from n2n.models import DetectionResult

MRZ_LINE_RE = re.compile(r"^[A-Z0-9<]{30,44}$")


@dataclass
class MrzDetection:
    text: str
    raw: str
    severity: str
    validators: List[str]


def detect_mrz(text: str) -> MrzDetection | None:
    if not text:
        return None
    lines = [line.strip().replace(" ", "") for line in text.splitlines() if line.strip()]
    lines = [line.upper() for line in lines]
    matched = [line for line in lines if MRZ_LINE_RE.match(line)]
    if len(matched) < 2:
        return None
    mrz_block = "\n".join(matched[:3])
    validators = ["mrz_pattern"]
    return MrzDetection(text="\n".join(matched[:3]), raw=mrz_block, severity="hit", validators=validators)


def build_detection(field: str, detection: MrzDetection, bbox, page: int) -> DetectionResult:
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


__all__ = ["detect_mrz", "build_detection"]
