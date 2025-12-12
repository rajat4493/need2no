from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber

from n2n.models import DetectionResult, ExtractionResult, PiiCategory, TextSpan
from n2n.primitives import register_primitive

NI_PATTERN = re.compile(r"\b(?!BG)(?!GB)(?!NK)(?!KN)(?!TN)(?!NT)(?!ZZ)[A-CEGHJ-PR-TW-Z]{2}\d{6}[A-D]?\b")
NHS_PATTERN = re.compile(r"\b\d{3}\s?\d{3}\s?\d{4}\b")
DRIVING_PATTERN = re.compile(r"\b[A-Z9]{5}\d{6}[A-Z9]{2}\d{2}\b")
PASSPORT_PATTERN = re.compile(r"\b\d{9}\b")
POSTCODE_PATTERN = re.compile(r"\b([A-Z]{1,2}\d[A-Z\d]?)\s?(\d[A-Z]{2})\b")

ADDRESS_KEYWORDS = ["road", "street", "avenue", "lane", "close", "drive", "flat", "house", "no.", "building"]


def _iter_lines(extraction: ExtractionResult) -> Iterable[Tuple[int, str]]:
    for page_index, page_text in enumerate(extraction.pages):
        if not page_text:
            continue
        for line in page_text.splitlines():
            yield page_index, line


def _line_has_context(line: str, keywords: Sequence[str]) -> bool:
    if not keywords:
        return False
    lower = line.lower()
    return any(keyword in lower for keyword in keywords)


def _build_detection(
    page_index: int,
    text: str,
    field_cfg: Dict[str, object],
    primitive: str,
    category_default: PiiCategory,
    context_line: str,
) -> DetectionResult:
    field_id = str(field_cfg.get("id", primitive))
    category_value = field_cfg.get("category", category_default)
    if isinstance(category_value, str):
        try:
            category = PiiCategory(category_value)
        except ValueError:
            category = category_default
    else:
        category = category_value

    return DetectionResult(
        field_id=field_id,
        category=category,
        primitive=primitive,
        span=TextSpan(
            page_index=page_index,
            text=text,
            bbox=(0.0, 0.0, 0.0, 0.0),
        ),
        confidence=1.0,
        context=context_line.strip(),
    )


def _resolve_region_bounds(field_cfg: Dict[str, object]) -> Optional[Dict[str, object]]:
    region_cfg = field_cfg.get("region_bounds") or field_cfg.get("region_def")
    region_value = field_cfg.get("region")

    if isinstance(region_value, dict):
        region_cfg = region_value
    elif isinstance(region_value, str):
        # TODO: map named regions from profile definitions to coordinates.
        pass

    return region_cfg


def _candidate_lines(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> Iterable[Tuple[int, str]]:
    region = _resolve_region_bounds(field_cfg)
    if region:
        return _extract_region_lines(extraction, region)
    return _iter_lines(extraction)


def _require_context_keywords(field_cfg: Dict[str, object]) -> List[str]:
    raw_keywords = field_cfg.get("context_keywords") or []
    return [str(k).lower() for k in raw_keywords if str(k).strip()]


def _detect_regex_with_context(
    extraction: ExtractionResult,
    pattern: re.Pattern[str],
    field_cfg: Dict[str, object],
    primitive: str,
    category_default: PiiCategory,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    keywords = _require_context_keywords(field_cfg)
    if not keywords:
        return detections

    for page_index, line in _candidate_lines(extraction, field_cfg):
        if not _line_has_context(line, keywords):
            continue
        for match in pattern.finditer(line):
            detections.append(
                _build_detection(
                    page_index=page_index,
                    text=match.group(),
                    field_cfg=field_cfg,
                    primitive=primitive,
                    category_default=category_default,
                    context_line=line,
                )
            )
    return detections


@register_primitive("uk_ni_number")
def detect_uk_ni_number(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _detect_regex_with_context(
        extraction,
        NI_PATTERN,
        field_cfg,
        "uk_ni_number",
        PiiCategory.GOV_IDENTIFIERS,
    )


@register_primitive("uk_nhs_number")
def detect_uk_nhs_number(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _detect_regex_with_context(
        extraction,
        NHS_PATTERN,
        field_cfg,
        "uk_nhs_number",
        PiiCategory.GOV_IDENTIFIERS,
    )


@register_primitive("uk_driving_licence")
def detect_uk_driving_licence(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _detect_regex_with_context(
        extraction,
        DRIVING_PATTERN,
        field_cfg,
        "uk_driving_licence",
        PiiCategory.GOV_IDENTIFIERS,
    )


@register_primitive("uk_passport_number")
def detect_uk_passport_number(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    return _detect_regex_with_context(
        extraction,
        PASSPORT_PATTERN,
        field_cfg,
        "uk_passport_number",
        PiiCategory.GOV_IDENTIFIERS,
    )


def _extract_region_lines(extraction: ExtractionResult, region_def: Optional[Dict[str, object]]) -> List[Tuple[int, str]]:
    if not region_def:
        return []

    results: List[Tuple[int, str]] = []
    page_index = int(region_def.get("page", 0))
    if page_index >= len(extraction.pages):
        return []

    with pdfplumber.open(str(extraction.file_path)) as pdf:
        if page_index >= len(pdf.pages):
            return []
        page = pdf.pages[page_index]
        x_range = region_def.get("x_range", (0.0, 1.0))
        y_range = region_def.get("y_range", (0.0, 1.0))
        bbox = (
            float(x_range[0]) * page.width,
            float(y_range[0]) * page.height,
            float(x_range[1]) * page.width,
            float(y_range[1]) * page.height,
        )
        cropped = page.crop(bbox)
        words = cropped.extract_words() or []

        if not words:
            return []

        lines: Dict[float, List[dict]] = {}
        for word in words:
            top = float(word["top"])
            lines.setdefault(top, []).append(word)

        for top in sorted(lines.keys()):
            line_words = lines[top]
            text = " ".join((w.get("text") or "").strip() for w in line_words if w.get("text"))
            if not text:
                continue
            results.append((page_index, text))

    return results


@register_primitive("uk_postcode_enhanced")
def detect_uk_postcode_enhanced(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    region_def = field_cfg.get("region_def") or field_cfg.get("region")
    allow_unrestricted = bool(field_cfg.get("allow_unrestricted"))

    lines: List[Tuple[int, str]]
    if region_def:
        lines = _extract_region_lines(extraction, region_def)
    elif allow_unrestricted:
        lines = list(_iter_lines(extraction))
    else:
        return []

    for page_index, line in lines:
        for match in POSTCODE_PATTERN.finditer(line):
            detections.append(
                _build_detection(
                    page_index=page_index,
                    text=match.group().strip(),
                    field_cfg=field_cfg,
                    primitive="uk_postcode_enhanced",
                    category_default=PiiCategory.CUSTOMER_IDENTITY,
                    context_line=line,
                )
            )
    return detections


def _line_matches_address(text: str) -> bool:
    lower = text.lower()
    if POSTCODE_PATTERN.search(text):
        return True
    return any(keyword in lower for keyword in ADDRESS_KEYWORDS)


@register_primitive("uk_address_line")
def detect_uk_address_line(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> List[DetectionResult]:
    region_def = field_cfg.get("region_def") or field_cfg.get("region")
    if not region_def:
        return []

    lines = _extract_region_lines(extraction, region_def)
    detections: List[DetectionResult] = []
    for page_index, line in lines:
        if not _line_matches_address(line):
            continue

        detections.append(
            _build_detection(
                page_index=page_index,
                text=line.strip(),
                field_cfg=field_cfg,
                primitive="uk_address_line",
                category_default=PiiCategory.CUSTOMER_IDENTITY,
                context_line=line,
            )
        )

    return detections


__all__ = [
    "detect_uk_ni_number",
    "detect_uk_nhs_number",
    "detect_uk_driving_licence",
    "detect_uk_passport_number",
    "detect_uk_postcode_enhanced",
    "detect_uk_address_line",
]
