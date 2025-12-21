from pathlib import Path
from typing import List

import pdfplumber

from n2n.models import ExtractionResult


KEYWORDS = ["sort code", "account number", "iban", "statement", "account"]


def _estimate_quality(pages: List[str]) -> float:
    joined = "\n".join(pages)
    length = len(joined.strip())
    if length == 0:
        return 0.0

    # length-based score: 0.0 -> 0 chars, 1.0 -> >= 2000 chars
    length_score = min(1.0, length / 2000.0)

    lower_text = joined.lower()
    keyword_boost = 0.1 if any(keyword in lower_text for keyword in KEYWORDS) else 0.0

    return float(min(1.0, length_score + keyword_boost))


def extract_text_with_quality(file_path: Path) -> ExtractionResult:
    pages: List[str] = []

    with pdfplumber.open(str(file_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    quality_score = _estimate_quality(pages)

    return ExtractionResult(
        file_path=file_path,
        quality_score=quality_score,
        pages=pages,
        source="text",
    )
