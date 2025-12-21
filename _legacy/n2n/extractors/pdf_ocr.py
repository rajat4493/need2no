from __future__ import annotations

from pathlib import Path
from typing import List

import fitz  # PyMuPDF
from PIL import Image
import pytesseract

from n2n.models import ExtractionResult

# Note: pytesseract requires the Tesseract binary to be installed on the host.
TESSERACT_CONFIG = "--oem 3 --psm 6"


def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Placeholder for future preprocessing (deskew/denoise). Currently returns the image as-is.
    """

    return image


def _pixmap_to_pil(pixmap: fitz.Pixmap) -> Image.Image:
    mode = "RGB" if pixmap.alpha == 0 else "RGBA"
    image = Image.frombytes(mode, [pixmap.width, pixmap.height], pixmap.samples)
    return image


def _estimate_quality(pages: List[str]) -> float:
    total_chars = sum(len(page) for page in pages)
    base = min(1.0, total_chars / 2000.0)

    joined_lower = " ".join(pages).lower()
    keywords = ["sort code", "account number", "iban", "statement"]
    if any(keyword in joined_lower for keyword in keywords):
        base = min(1.0, base + 0.1)

    return float(base)


def extract_text_with_quality_ocr(file_path: Path) -> ExtractionResult:
    pages: List[str] = []
    doc = fitz.open(str(file_path))
    try:
        zoom = 300 / 72
        matrix = fitz.Matrix(zoom, zoom)

        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix)
            image = _pixmap_to_pil(pixmap)
            processed = preprocess_image(image)
            data = pytesseract.image_to_data(
                processed, output_type=pytesseract.Output.DICT, config=TESSERACT_CONFIG
            )
            words = [word.strip() for word in data.get("text", []) if word and word.strip()]
            page_text = " ".join(words)
            pages.append(page_text)
    finally:
        doc.close()

    quality_score = _estimate_quality(pages)

    return ExtractionResult(
        file_path=file_path,
        quality_score=quality_score,
        pages=pages,
        source="ocr",
    )


__all__ = ["extract_text_with_quality_ocr"]
