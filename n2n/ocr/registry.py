from __future__ import annotations

import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np

from n2n.ocr.backends.apple_vision_backend import AppleVisionBackend
from n2n.ocr.backends.base import (
    BBox,
    BackendUnavailable,
    OCRBackend,
    OCRConfig,
    OCRResult,
    OCRWord,
)
from n2n.ocr.backends.easyocr_backend import EasyOCRBackend
from n2n.ocr.backends.paddle_backend import PaddleBackend
from n2n.ocr.backends.tesseract_backend import TesseractBackend

_FACTORIES: Dict[str, callable] = {
    "tesseract": TesseractBackend,
    "apple": AppleVisionBackend,
    "paddle": PaddleBackend,
    "easy": EasyOCRBackend,
}

_VALID_MODES = {"auto", "combo", "tesseract", "apple", "paddle", "easy"}


def resolve_backend_mode(cli_value: str | None) -> str:
    env_mode = os.getenv("N2N_OCR_BACKEND")
    mode = (cli_value or env_mode or "auto").lower()
    return mode if mode in _VALID_MODES else "auto"


def _sequence_for_mode(mode: str) -> List[str]:
    if mode == "tesseract":
        return ["tesseract"]
    if mode == "apple":
        return ["apple"]
    if mode == "paddle":
        return ["paddle"]
    if mode == "easy":
        return ["easy"]
    # default/auto/combo pipeline: Apple Vision -> PaddleOCR -> Tesseract
    order: List[str] = ["apple", "paddle", "tesseract"]
    return order


def get_backends_for_mode(mode: str) -> List[OCRBackend]:
    order = _sequence_for_mode(mode)
    backends: List[OCRBackend] = []
    for name in order:
        factory = _FACTORIES.get(name)
        if not factory:
            continue
        backend = factory()
        if backend.is_available():
            backends.append(backend)
    if not backends:
        backends.append(TesseractBackend())
    return backends


def run_ocr_backends(
    image: np.ndarray,
    roi_bbox: BBox,
    config: OCRConfig,
    mode: str,
) -> Tuple[Sequence[OCRResult], List[dict]]:
    attempts: List[dict] = []
    results: List[OCRResult] = []
    for backend in get_backends_for_mode(mode):
        start = time.perf_counter()
        try:
            result = backend.ocr_roi(image, roi_bbox, config)
            elapsed = result.elapsed_ms or (time.perf_counter() - start) * 1000.0
            attempt = {
                "engine": backend.name,
                "success": True,
                "text_preview": (result.text or "")[:40],
                "avg_conf": result.avg_conf,
                "elapsed_ms": round(elapsed, 2),
            }
            attempts.append(attempt)
            results.append(result)
        except BackendUnavailable as exc:
            attempts.append(
                {
                    "engine": backend.name,
                    "success": False,
                    "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                }
            )
            continue
        except Exception as exc:  # pragma: no cover - defensive against backend crashes
            attempts.append(
                {
                    "engine": backend.name,
                    "success": False,
                    "error": str(exc),
                    "elapsed_ms": round((time.perf_counter() - start) * 1000.0, 2),
                }
            )
            continue
    if not results:
        results.append(
            OCRResult(text="", avg_conf=0.0, words=[], engine="none", elapsed_ms=0.0)
        )
    return results, attempts


__all__ = [
    "BackendUnavailable",
    "OCRConfig",
    "OCRResult",
    "OCRWord",
    "resolve_backend_mode",
    "get_backends_for_mode",
    "run_ocr_backends",
]
