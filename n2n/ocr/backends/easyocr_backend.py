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

_EASY_CACHE = {}


class EasyOCRBackend(OCRBackend):
    name = "easyocr"

    def is_available(self) -> bool:
        try:
            import easyocr  # noqa: F401
        except Exception:
            return False
        return True

    def _client(self, lang: str):
        try:
            import easyocr
        except Exception as exc:
            raise BackendUnavailable(f"EasyOCR not installed: {exc}") from exc
        languages = [lang or "en"]
        key = tuple(languages)
        if key not in _EASY_CACHE:
            _EASY_CACHE[key] = easyocr.Reader(languages, gpu=False)
        return _EASY_CACHE[key]

    def ocr_roi(self, image: np.ndarray, roi_bbox: BBox, config: OCRConfig) -> OCRResult:
        x1, y1, x2, y2 = _normalize_bbox(roi_bbox, image.shape)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(text="", avg_conf=0.0, words=[], engine=self.name, elapsed_ms=0.0)
        reader = self._client(config.lang or "en")
        t0 = time.perf_counter()
        try:
            result = reader.readtext(roi, detail=1, paragraph=False)
        except Exception as exc:
            raise BackendUnavailable(f"EasyOCR failed: {exc}") from exc
        elapsed = (time.perf_counter() - t0) * 1000.0
        words: List[OCRWord] = []
        confidences = []
        texts = []
        for bbox, text, conf in result:
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            confidence = max(0.0, min(float(conf), 1.0))
            confidences.append(confidence)
            texts.append(cleaned)
            if isinstance(bbox, np.ndarray):
                xs = bbox[:, 0]
                ys = bbox[:, 1]
                abs_bbox: Tuple[float, float, float, float] = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            else:
                abs_bbox = (x1, y1, x2, y2)
            words.append(OCRWord(text=cleaned, bbox=abs_bbox, confidence=confidence))
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


__all__ = ["EasyOCRBackend"]
