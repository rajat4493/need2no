from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import List, Protocol, Sequence, Tuple

BBox = Tuple[float, float, float, float]


class BackendUnavailable(RuntimeError):
    """Raised when an OCR backend cannot run on the current platform."""


@dataclass
class OCRWord:
    text: str
    bbox: Tuple[float, float, float, float]
    confidence: float


@dataclass
class OCRResult:
    text: str
    avg_conf: float
    words: Sequence[OCRWord]
    engine: str
    elapsed_ms: float

    def as_dict(self) -> dict:
        return {
            "text": self.text,
            "avg_conf": self.avg_conf,
            "engine": self.engine,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "words": [
                {"text": word.text, "bbox": list(word.bbox), "confidence": word.confidence}
                for word in self.words
            ],
        }


@dataclass
class OCRConfig:
    psm: int = 6
    lang: str = "eng"
    whitelist_digits: bool = False
    extra_whitelist: str = ""


class OCRBackend(Protocol):
    name: str

    def is_available(self) -> bool: ...

    def ocr_roi(self, image, roi_bbox: BBox, config: OCRConfig) -> OCRResult: ...


__all__ = ["OCRBackend", "OCRConfig", "OCRResult", "OCRWord", "BackendUnavailable", "BBox"]
