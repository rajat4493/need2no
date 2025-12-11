import re
from typing import Dict, Iterable, List, Sequence, Tuple

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
        return True
    lower = line.lower()
    return any(keyword in lower for keyword in keywords)


def _build_span(page_index: int, match_text: str) -> TextSpan:
    return TextSpan(
        page_index=page_index,
        text=match_text,
        bbox=(0.0, 0.0, 0.0, 0.0),
    )


def _detect_pattern(
    extraction: ExtractionResult,
    pattern: re.Pattern[str],
    field_cfg: Dict[str, object],
    primitive: str,
    default_category: PiiCategory,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    field_id = str(field_cfg["id"])
    keywords = [str(k).lower() for k in field_cfg.get("context_keywords", [])]
    raw_category = field_cfg.get("category")
    category = raw_category if isinstance(raw_category, PiiCategory) else default_category

    for page_index, line in _iter_lines(extraction):
        if not _line_has_context(line, keywords):
            continue

        for match in pattern.finditer(line):
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
