import re
from typing import List

from n2n.models import DetectionResult, ExtractionResult, PiiType, TextSpan

# Regex patterns
SORT_CODE_PATTERN = re.compile(r"\b\d{2}-\d{2}-\d{2}\b")
ACCOUNT_NUMBER_PATTERN = re.compile(r"\b\d{8}\b")
IBAN_PATTERN = re.compile(r"\bGB[0-9A-Z]{2}[0-9A-Z ]{10,30}\b")
CARD_PATTERN = re.compile(r"\b(?:\d{4}[- ]?){3}\d{4}\b")

# Context keywords
SORT_CODE_KEYWORDS = ["sort code", "sort-code", "sc"]
ACCOUNT_KEYWORDS = ["account number", "account no", "a/c no", "acc no"]
IBAN_KEYWORDS = ["iban"]
CARD_KEYWORDS = ["card number", "debit card", "credit card", "card ending"]


def _has_context(line: str, keywords) -> bool:
    lower = line.lower()
    return any(k in lower for k in keywords)


def _build_span(page_index: int, text: str) -> TextSpan:
    # For now we don't have real coordinates – we’ll set a dummy bbox.
    # Later you’ll wire this to word-level positions from pdfplumber / PyMuPDF.
    return TextSpan(
        page_index=page_index,
        text=text,
        bbox=(0.0, 0.0, 0.0, 0.0),
    )


def detect_pii_uk_bank_statement(extraction: ExtractionResult) -> List[DetectionResult]:
    """
    v0.1 detector with strict rule:
    - Only mark a detection when regex AND context keyword are present in the same line.
    - Confidence is set to 1.0 when conditions are met.
    """
    detections: List[DetectionResult] = []

    for page_index, page_text in enumerate(extraction.pages):
        if not page_text:
            continue

        lines = page_text.splitlines()

        for line in lines:
            # Sort Code
            if _has_context(line, SORT_CODE_KEYWORDS):
                for m in SORT_CODE_PATTERN.finditer(line):
                    detections.append(
                        DetectionResult(
                            pii_type=PiiType.SORT_CODE,
                            span=_build_span(page_index, m.group()),
                            confidence=1.0,
                            context=line.strip(),
                        )
                    )

            # Account Number
            if _has_context(line, ACCOUNT_KEYWORDS):
                for m in ACCOUNT_NUMBER_PATTERN.finditer(line):
                    detections.append(
                        DetectionResult(
                            pii_type=PiiType.ACCOUNT_NUMBER,
                            span=_build_span(page_index, m.group()),
                            confidence=1.0,
                            context=line.strip(),
                        )
                    )

            # IBAN
            if _has_context(line, IBAN_KEYWORDS):
                for m in IBAN_PATTERN.finditer(line):
                    detections.append(
                        DetectionResult(
                            pii_type=PiiType.IBAN,
                            span=_build_span(page_index, m.group()),
                            confidence=1.0,
                            context=line.strip(),
                        )
                    )

            # Card Number
            if _has_context(line, CARD_KEYWORDS):
                for m in CARD_PATTERN.finditer(line):
                    detections.append(
                        DetectionResult(
                            pii_type=PiiType.CARD_NUMBER,
                            span=_build_span(page_index, m.group()),
                            confidence=1.0,
                            context=line.strip(),
                        )
                    )

    # Name + Address header: we’ll handle in a separate pass later
    # For v0.1 skeleton we can leave it unimplemented or very strict.

    return detections
