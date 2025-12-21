from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from n2n.models import TextSpan
from n2n.ocr.backends.base import OCRConfig
from n2n.ocr.backends.tesseract_backend import TesseractBackend

_BACKEND = TesseractBackend()

_MODE_CONFIGS: Dict[str, OCRConfig] = {
    "pan_digits": OCRConfig(psm=7, lang="eng", whitelist_digits=True),
    "expiry": OCRConfig(psm=7, lang="eng", whitelist_digits=True, extra_whitelist="/"),
    "mrz": OCRConfig(psm=6, lang="eng", whitelist_digits=False, extra_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<"),
    "name": OCRConfig(psm=7, lang="eng", whitelist_digits=False),
    "id_alnum": OCRConfig(psm=7, lang="eng", whitelist_digits=False, extra_whitelist="0123456789"),
}


def ocr_roi(
    image: np.ndarray,
    roi_box: Tuple[float, float, float, float] | None,
    mode: str = "pan_digits",
) -> Tuple[str, Dict[str, float], List[TextSpan]]:
    h, w = image.shape[:2]
    if roi_box is None:
        roi_box = (0, 0, w, h)
    config = _MODE_CONFIGS.get(mode, OCRConfig(psm=6))
    result = _BACKEND.ocr_roi(image, roi_box, config)
    spans: List[TextSpan] = [
        TextSpan(text=word.text, bbox=word.bbox, page=0, source="ocr_roi", ocr_conf=word.confidence)
        for word in result.words
    ]
    stats = {"avg_conf": round(result.avg_conf, 4), "min_conf": round(min((word.confidence for word in result.words), default=0.0), 4)}
    return result.text, stats, spans


__all__ = ["ocr_roi"]
