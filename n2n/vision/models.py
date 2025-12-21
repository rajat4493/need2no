from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Box:
    label: str
    conf: float
    page: int
    x1: float
    y1: float
    x2: float
    y2: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass
class VisionResult:
    page_images: List[str] = field(default_factory=list)
    boxes: List[Box] = field(default_factory=list)
    trace: Dict[str, object] = field(default_factory=dict)
