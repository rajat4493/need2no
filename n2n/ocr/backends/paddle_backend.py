from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np

from n2n.ocr.backends.base import (
    BBox,
    BackendUnavailable,
    OCRBackend,
    OCRConfig,
    OCRResult,
    OCRWord,
)

_PADDLE_CACHE = {}


class PaddleBackend(OCRBackend):
    name = "paddleocr"

    def is_available(self) -> bool:
        try:
            from paddleocr import PaddleOCR  # noqa: F401
        except Exception:
            return False
        return True

    def _client(self, lang: str):
        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise BackendUnavailable(f"PaddleOCR not installed: {exc}") from exc
        key = lang or "en"
        if key not in _PADDLE_CACHE:
            _PADDLE_CACHE[key] = PaddleOCR(lang=key, show_log=False, det=False, use_angle_cls=False)
        return _PADDLE_CACHE[key]

    def ocr_roi(self, image: np.ndarray, roi_bbox: BBox, config: OCRConfig) -> OCRResult:
        x1, y1, x2, y2 = _normalize_bbox(roi_bbox, image.shape)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(text="", avg_conf=0.0, words=[], engine=self.name, elapsed_ms=0.0)
        client = self._client(config.lang or "en")
        t0 = time.perf_counter()
        try:
            result = client.ocr(roi, det=False, rec=True, cls=False)
        except Exception as exc:
            raise BackendUnavailable(f"PaddleOCR failed: {exc}") from exc
        elapsed = (time.perf_counter() - t0) * 1000.0
        words: List[OCRWord] = []
        confidences = []
        texts = []
        for entry in result:
            if not entry:
                continue
            text, conf = entry[0]
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            confidence = max(0.0, min(float(conf), 1.0))
            confidences.append(confidence)
            texts.append(cleaned)
            words.append(
                OCRWord(
                    text=cleaned,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence,
                )
            )
        avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
        return OCRResult(
            text=" ".join(texts),
            avg_conf=avg_conf,
            words=words,
            engine=self.name,
            elapsed_ms=elapsed,
        )


def _normalize_bbox(box: BBox, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
    height, width = shape[:2]
    x1 = max(0, min(int(box[0]), width - 1))
    y1 = max(0, min(int(box[1]), height - 1))
    x2 = max(x1 + 1, min(int(box[2]), width))
    y2 = max(y1 + 1, min(int(box[3]), height))
    return x1, y1, x2, y2


__all__ = ["PaddleBackend"]
