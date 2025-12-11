from typing import Dict, List, Optional

from n2n.models import DetectionResult, ExtractionResult, PiiCategory
from n2n.primitives.name_address import detect_name_header
from n2n.primitives.uk_bank import (
    detect_card_16,
    detect_iban_gb,
    detect_uk_account_number_8d,
    detect_uk_sort_code,
)

SORT_CODE_CFG = {
    "id": "sort_code",
    "context_keywords": ["sort code", "sort-code", "sc"],
    "category": PiiCategory.BANK_IDENTIFIERS,
}

ACCOUNT_CFG = {
    "id": "account_number",
    "context_keywords": ["account number", "account no", "a/c no", "acc no"],
    "category": PiiCategory.BANK_IDENTIFIERS,
}

IBAN_CFG = {
    "id": "iban_gb",
    "context_keywords": ["iban"],
    "category": PiiCategory.BANK_IDENTIFIERS,
}

CARD_CFG = {
    "id": "card_16",
    "context_keywords": ["card number", "debit card", "credit card", "card ending"],
    "category": PiiCategory.CARD_NUMBERS,
}


def _build_account_name_cfg(profile: Optional[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not profile:
        return None

    regions = profile.get("regions", {})
    fields = profile.get("fields", [])
    header_region = regions.get("header_block")
    if not header_region:
        return None

    for field in fields:
        if field.get("primitive") == "name_header":
            cfg = {
                "id": field.get("id", "account_name"),
                "category": PiiCategory.CUSTOMER_IDENTITY,
                "primitive": "name_header",
                "region_def": header_region,
            }
            return cfg

    return None


def detect_pii_uk_bank_statement(
    extraction: ExtractionResult,
    profile: Optional[Dict[str, object]] = None,
) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    detections.extend(detect_uk_sort_code(extraction, SORT_CODE_CFG))
    detections.extend(detect_uk_account_number_8d(extraction, ACCOUNT_CFG))
    detections.extend(detect_iban_gb(extraction, IBAN_CFG))
    detections.extend(detect_card_16(extraction, CARD_CFG))

    account_name_cfg = _build_account_name_cfg(profile)
    if account_name_cfg:
        detections.extend(detect_name_header(extraction, account_name_cfg))

    return detections
