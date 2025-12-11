from __future__ import annotations

from pathlib import Path
from typing import List, Sequence, Tuple

import pdfplumber

WordEntry = Tuple[str, Tuple[float, float, float, float]]


def extract_word_entries(pdf_path: Path) -> List[List[WordEntry]]:
    pages: List[List[WordEntry]] = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            page_words: List[WordEntry] = []
            for word in page.extract_words():
                text = (word.get("text") or "").strip()
                if not text:
                    continue
                bbox = (
                    float(word["x0"]),
                    float(word["top"]),
                    float(word["x1"]),
                    float(word["bottom"]),
                )
                page_words.append((text, bbox))
            pages.append(page_words)
    return pages


def _combine_bboxes(bboxes: Sequence[Tuple[float, float, float, float]]) -> Tuple[float, float, float, float]:
    x0 = min(b[0] for b in bboxes)
    y0 = min(b[1] for b in bboxes)
    x1 = max(b[2] for b in bboxes)
    y1 = max(b[3] for b in bboxes)
    return (x0, y0, x1, y1)


def find_bbox_for_tokens(words: Sequence[WordEntry], tokens: Sequence[str]) -> Tuple[float, float, float, float] | None:
    token_len = len(tokens)
    if token_len == 0:
        return None

    if token_len == 1:
        target = tokens[0]
        for text, bbox in words:
            if text == target:
                return bbox
        return None

    for idx in range(len(words) - token_len + 1):
        window = words[idx : idx + token_len]
        if all(window[j][0] == tokens[j] for j in range(token_len)):
            return _combine_bboxes([entry[1] for entry in window])
    return None


def find_bbox_for_text(words: Sequence[WordEntry], text: str) -> Tuple[float, float, float, float] | None:
    tokens = text.split()
    return find_bbox_for_tokens(words, tokens)


__all__ = ["WordEntry", "extract_word_entries", "find_bbox_for_tokens", "find_bbox_for_text"]
