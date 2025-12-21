from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List

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


class AppleVisionBackend(OCRBackend):
    name = "apple_vision"

    def __init__(self, binary_path: Path | None = None) -> None:
        default_path = Path("tools/apple_vision_ocr/.build/release/AppleVisionOCR")
        self.binary_path = Path(binary_path) if binary_path else default_path

    def is_available(self) -> bool:
        return sys.platform == "darwin" and self.binary_path.exists()

    def ocr_roi(self, image: np.ndarray, roi_bbox: BBox, config: OCRConfig) -> OCRResult:
        if not self.is_available():
            raise BackendUnavailable(
                f"Apple Vision backend not available. Build the helper via `swift build -c release` in {self.binary_path.parent.parent}"
            )
        x1, y1, x2, y2 = _normalize_bbox(roi_bbox, image.shape)
        roi = image[y1:y2, x1:x2]
        if roi.size == 0:
            return OCRResult(text="", avg_conf=0.0, words=[], engine=self.name, elapsed_ms=0.0)
        temp_dir = tempfile.mkdtemp(prefix="n2n_apple_vision_")
        img_path = Path(temp_dir) / "roi.png"
        cv2.imwrite(str(img_path), roi)
        cmd = [
            str(self.binary_path),
            "--image",
            str(img_path),
            "--roi",
            f"{x1},{y1},{x2},{y2}",
            "--lang",
            config.lang or "en",
            "--digits-only",
            "1" if config.whitelist_digits else "0",
        ]
        t0 = time.perf_counter()
        try:
            completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as exc:
            raise BackendUnavailable(f"Apple Vision helper failed: {exc.stderr.strip()}") from exc
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass
            try:
                os.rmdir(temp_dir)
            except OSError:
                pass
        elapsed = (time.perf_counter() - t0) * 1000.0
        payload = json.loads(completed.stdout or "{}")
        words = [
            OCRWord(
                text=entry["text"],
                bbox=tuple(entry.get("bbox", [])) or (x1, y1, x2, y2),
                confidence=float(entry.get("confidence", 0.0)),
            )
            for entry in payload.get("words", [])
        ]
        avg_conf = float(payload.get("avg_conf", 0.0))
        return OCRResult(
            text=payload.get("text", ""),
            avg_conf=avg_conf,
            words=words,
            engine=self.name,
            elapsed_ms=elapsed,
        )


def _normalize_bbox(box: BBox, shape: tuple[int, int, int]) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1 = max(0, min(int(box[0]), width - 1))
    y1 = max(0, min(int(box[1]), height - 1))
    x2 = max(x1 + 1, min(int(box[2]), width))
    y2 = max(y1 + 1, min(int(box[3]), height))
    return x1, y1, x2, y2


__all__ = ["AppleVisionBackend"]
