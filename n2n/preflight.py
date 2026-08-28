"""Classify the input document before any detection/redaction is attempted.

Phase 1 supports exactly one class: native_text_pdf. Everything else is
rejected immediately with UNSUPPORTED — no partial processing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

SUPPORTED_CLASSES = frozenset({"native_text_pdf"})


@dataclass(frozen=True)
class PreflightResult:
    classification: str
    reason: str
    supported: bool


def classify(path: Path) -> PreflightResult:
    if not path.exists():
        return PreflightResult("missing", "Input file does not exist.", False)
    return classify_bytes(path.read_bytes())


def classify_bytes(data: bytes) -> PreflightResult:
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - any open failure means corrupted/unreadable
        return PreflightResult("corrupted", f"Could not open document: {exc}", False)

    try:
        if doc.is_encrypted:
            return PreflightResult(
                "encrypted_unsupported",
                "Document is encrypted; Phase 1 does not support encrypted inputs.",
                False,
            )

        if doc.page_count == 0:
            return PreflightResult("missing_pages", "Document has no pages.", False)

        total_chars = 0
        total_images = 0
        for page in doc:
            total_chars += len(page.get_text("text").strip())
            total_images += len(page.get_images(full=True))

        if total_chars == 0:
            if total_images > 0:
                return PreflightResult(
                    "scanned_pdf",
                    "Document appears to be scanned/image-only; Phase 1 requires native text (OCR fallback lands in Phase 3).",
                    False,
                )
            return PreflightResult(
                "empty",
                "Document has no extractable text or images.",
                False,
            )

        # Hybrid: substantial text alongside full-page images (a photographed
        # page saved as a searchable PDF can still have very sparse text) is
        # out of scope for Phase 1 to avoid silently mis-trusting an OCR layer
        # we didn't generate ourselves.
        avg_chars_per_page = total_chars / doc.page_count
        if avg_chars_per_page < 40:
            return PreflightResult(
                "hybrid_low_text",
                "Document has very little native text per page; treating as unsupported rather than guessing whether it's a reliable text layer.",
                False,
            )

        return PreflightResult("native_text_pdf", "Native text PDF.", True)
    finally:
        doc.close()
