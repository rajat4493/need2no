from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class PiiType(str, Enum):
    SORT_CODE = "sort_code"
    ACCOUNT_NUMBER = "account_number"
    IBAN = "iban"
    CARD_NUMBER = "card_number"
    NAME_ADDRESS_HEADER = "name_address_header"


@dataclass
class TextSpan:
    page_index: int
    text: str
    bbox: Tuple[float, float, float, float]  # (x0, y0, x1, y1)


@dataclass
class DetectionResult:
    pii_type: PiiType
    span: TextSpan
    confidence: float  # 0–1 but must be 1.0 for “strict mode” in v0.1
    context: str       # e.g., surrounding text line


@dataclass
class ExtractionResult:
    file_path: Path
    quality_score: float
    pages: List[str]  # simple text per page for now
    # later: you can include per-word coordinates for precise redaction


@dataclass
class RedactionOutcome:
    input_path: Path
    output_path: Optional[Path]
    redactions_applied: int
    reason: Optional[str] = None  # e.g. "quality_too_low", "no_pii_found"
