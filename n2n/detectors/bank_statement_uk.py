from typing import List

from n2n.models import DetectionResult, ExtractionResult, PiiCategory
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


def detect_pii_uk_bank_statement(extraction: ExtractionResult) -> List[DetectionResult]:
    detections: List[DetectionResult] = []
    detections.extend(detect_uk_sort_code(extraction, SORT_CODE_CFG))
    detections.extend(detect_uk_account_number_8d(extraction, ACCOUNT_CFG))
    detections.extend(detect_iban_gb(extraction, IBAN_CFG))
    detections.extend(detect_card_16(extraction, CARD_CFG))
    return detections
