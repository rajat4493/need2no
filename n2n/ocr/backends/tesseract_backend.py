from __future__ import annotations

import time
from typing import List, Tuple

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from n2n.ocr.backends.base import (
    BBox,
    OCRBackend,
    OCRConfig,
    OCRResult,
    OCRWord,
)


class TesseractBackend(OCRBackend):
    name = "tesseract"

    def is_available(self) -> bool:
        return True

    def ocr_roi(self, image: np.ndarray, roi_bbox: BBox, config: OCRConfig) -> OCRResult:
        x1, y1, x2, y2 = _normalize_bbox(roi_bbox, image.shape)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(text="", avg_conf=0.0, words=[], engine=self.name, elapsed_ms=0.0)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        cfg_parts = [f"--psm {config.psm}"]
        if config.lang:
            cfg_parts.append(f"-l {config.lang}")
        whitelist = ""
        if config.whitelist_digits:
            whitelist = "0123456789"
        if config.extra_whitelist:
            whitelist += config.extra_whitelist
        if whitelist:
            cfg_parts.append(f'-c tessedit_char_whitelist="{whitelist}"')
        t0 = time.perf_counter()
        data = pytesseract.image_to_data(gray, output_type=Output.DICT, config=" ".join(cfg_parts))
        elapsed = (time.perf_counter() - t0) * 1000.0
        words: List[OCRWord] = []
        confidences: List[float] = []
        texts: List[str] = []
        for idx, text in enumerate(data.get("text", [])):
            cleaned = (text or "").strip()
            if not cleaned:
                continue
            try:
                conf_raw = float(data.get("conf", [0])[idx])
            except (ValueError, TypeError):
                conf_raw = 0.0
            left = int(data.get("left", [0])[idx])
            top = int(data.get("top", [0])[idx])
            width_box = int(data.get("width", [0])[idx])
            height_box = int(data.get("height", [0])[idx])
            abs_bbox: Tuple[float, float, float, float] = (
                x1 + left,
                y1 + top,
                x1 + left + width_box,
                y1 + top + height_box,
            )
            confidence = max(0.0, min(conf_raw / 100.0, 1.0))
            words.append(OCRWord(text=cleaned, bbox=abs_bbox, confidence=confidence))
            confidences.append(confidence)
            texts.append(cleaned)
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


__all__ = ["TesseractBackend"]
