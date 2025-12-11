from pathlib import Path
from typing import List

import pdfplumber

from n2n.models import ExtractionResult


def _estimate_quality(pages: List[str]) -> float:
    """
    Very naive heuristic for now:
    - If there is "Account" or "Sort Code" etc anywhere, bump score.
    - If pages are very short / mostly whitespace, lower score.

    You can replace this later with a better layout/char coverage metric.
    """
    joined = "\n".join(pages)
    length = len(joined.strip())
    if length == 0:
        return 0.0

    base = min(1.0, length / 2000.0)  # if we have ~2000+ chars, treat as decent

    keywords = ["sort code", "account number", "iban", "statement"]
    boost = any(k.lower() in joined.lower() for k in keywords)

    quality = base + (0.1 if boost else 0.0)
    return float(min(1.0, quality))


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
    )
