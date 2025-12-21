from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class TextSpan:
    text: str
    bbox: Tuple[float, float, float, float]
    page: int
    source: str = "text"
    ocr_conf: Optional[float] = None


@dataclass
class DetectionResult:
    field_id: str
    text: str
    raw: str
    bbox: Tuple[float, float, float, float]
    page: int
    source: str
    validators: List[str]
    severity: str


@dataclass
class DecisionReason:
    code: str
    description: str


@dataclass
class DecisionReport:
    pack_id: str
    decision: str
    reasons: List[DecisionReason]
    detections: List[DetectionResult]
    artifacts: Dict[str, Optional[str]]
    engine_version: str
    suggested_redactions: List[Dict[str, object]] = field(default_factory=list)
    action: Optional[str] = None
    trace: Optional[Dict[str, object]] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "pack_id": self.pack_id,
            "decision": self.decision,
            "reasons": [reason.__dict__ for reason in self.reasons],
            "detections": [det.__dict__ for det in self.detections],
            "artifacts": self.artifacts,
            "engine_version": self.engine_version,
            "suggested_redactions": self.suggested_redactions,
            "action": self.action,
            "trace": self.trace or {},
        }
