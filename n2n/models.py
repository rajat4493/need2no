from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple


class PiiCategory(str, Enum):
    CUSTOMER_IDENTITY = "customer_identity"
    BANK_IDENTIFIERS = "bank_identifiers"
    CARD_NUMBERS = "card_numbers"
    TRANSACTION_DETAILS = "transaction_details"


@dataclass
class TextSpan:
    page_index: int
    text: str
    bbox: Tuple[float, float, float, float]


@dataclass
class DetectionResult:
    field_id: str
    category: PiiCategory
    primitive: str
    span: TextSpan
    confidence: float
    context: str


@dataclass
class ExtractionResult:
    file_path: Path
    quality_score: float
    pages: List[str]


@dataclass
class RedactionOutcome:
    input_path: Path
    output_path: Optional[Path]
    redactions_applied: int
    reason: Optional[str] = None
