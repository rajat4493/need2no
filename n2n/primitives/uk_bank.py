import re
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import pdfplumber

from n2n.models import DetectionResult, ExtractionResult, PiiCategory, TextSpan
from n2n.primitives import register_primitive

SORT_CODE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{8}\b")
IBAN_PATTERN = re.compile(r"\bGB[0-9A-Z]{2}[0-9A-Z ]{10,30}\b")
CARD_PATTERN = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")


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


def _build_span(page_index: int, match_text: str) -> TextSpan:
    return TextSpan(
        page_index=page_index,
        text=match_text,
        bbox=(0.0, 0.0, 0.0, 0.0),
    )


def _extract_region_lines(extraction: ExtractionResult, region: Dict[str, object]) -> List[Tuple[int, str]]:
    page_index = int(region.get("page", 0))
    if page_index >= len(extraction.pages):
        return []

    with pdfplumber.open(str(extraction.file_path)) as pdf:
        if page_index >= len(pdf.pages):
            return []

        page = pdf.pages[page_index]
        x_range = region.get("x_range", (0.0, 1.0))
        y_range = region.get("y_range", (0.0, 1.0))
        bbox = (
            float(x_range[0]) * page.width,
            float(y_range[0]) * page.height,
            float(x_range[1]) * page.width,
            float(y_range[1]) * page.height,
        )

        cropped = page.crop(bbox)
        region_text = cropped.extract_text() or ""
        lines = [line.strip() for line in region_text.splitlines() if line.strip()]
        return [(page_index, line) for line in lines]


def _resolve_region_bounds(field_cfg: Dict[str, object]) -> Optional[Dict[str, object]]:
    region_cfg = field_cfg.get("region_bounds") or field_cfg.get("region_def")
    region_value = field_cfg.get("region")

    if isinstance(region_value, dict):
        region_cfg = region_value
    elif isinstance(region_value, str):
        # TODO: map region name to coordinates from profile config.
        pass

    return region_cfg


def _candidate_lines(extraction: ExtractionResult, field_cfg: Dict[str, object]) -> Iterable[Tuple[int, str]]:
    region = _resolve_region_bounds(field_cfg)
    if region:
        return _extract_region_lines(extraction, region)
    return _iter_lines(extraction)


def _require_context_keywords(field_cfg: Dict[str, object]) -> List[str]:
    raw_keywords = field_cfg.get("context_keywords") or []
    keywords = [str(k).lower() for k in raw_keywords if str(k).strip()]
    return keywords


def _account_number_filter(line: str, match: re.Match[str]) -> bool:
    start, end = match.span()
    snippet = line[max(0, start - 1) : min(len(line), end + 1)]
    noise_chars = {".", ",", "£", "$"}
    return not any(ch in snippet for ch in noise_chars)


def _detect_pattern(
    extraction: ExtractionResult,
    pattern: re.Pattern[str],
    field_cfg: Dict[str, object],
    primitive: str,
    default_category: PiiCategory,
    match_filter: Optional[Callable[[str, re.Match[str]], bool]] = None,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    field_id = str(field_cfg["id"])
    keywords = _require_context_keywords(field_cfg)
    if not keywords:
        return detections
    raw_category = field_cfg.get("category")
    category = raw_category if isinstance(raw_category, PiiCategory) else default_category

    for page_index, line in _candidate_lines(extraction, field_cfg):
        if not _line_has_context(line, keywords):
            continue

        for match in pattern.finditer(line):
            if match_filter and not match_filter(line, match):
                continue
            detections.append(
                DetectionResult(
                    field_id=field_id,
                    category=category,
                    primitive=primitive,
                    span=_build_span(page_index, match.group()),
                    confidence=1.0,
                    context=line.strip(),
                )
            )

    return detections


@register_primitive("uk_sort_code")
def detect_uk_sort_code(
    extraction: ExtractionResult,
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    return _detect_pattern(
        extraction=extraction,
        pattern=SORT_CODE_PATTERN,
        field_cfg=field_cfg,
        primitive="uk_sort_code",
        default_category=PiiCategory.BANK_IDENTIFIERS,
    )


@register_primitive("uk_account_number_8d")
def detect_uk_account_number_8d(
    extraction: ExtractionResult,
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    return _detect_pattern(
        extraction=extraction,
        pattern=ACCOUNT_NUMBER_PATTERN,
        field_cfg=field_cfg,
        primitive="uk_account_number_8d",
        default_category=PiiCategory.BANK_IDENTIFIERS,
        match_filter=_account_number_filter,
    )


@register_primitive("iban_gb")
def detect_iban_gb(
    extraction: ExtractionResult,
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    return _detect_pattern(
        extraction=extraction,
        pattern=IBAN_PATTERN,
        field_cfg=field_cfg,
        primitive="iban_gb",
        default_category=PiiCategory.BANK_IDENTIFIERS,
    )


@register_primitive("card_16")
def detect_card_16(
    extraction: ExtractionResult,
    field_cfg: Dict[str, object],
) -> List[DetectionResult]:
    return _detect_pattern(
        extraction=extraction,
        pattern=CARD_PATTERN,
        field_cfg=field_cfg,
        primitive="card_16",
        default_category=PiiCategory.CARD_NUMBERS,
    )


__all__ = [
    "detect_uk_sort_code",
    "detect_uk_account_number_8d",
    "detect_iban_gb",
    "detect_card_16",
]
