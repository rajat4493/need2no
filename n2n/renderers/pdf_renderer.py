from pathlib import Path
from typing import List

import fitz  # PyMuPDF

from n2n.models import DetectionResult


def apply_redactions(
    input_pdf: Path,
    detections: List[DetectionResult],
    output_pdf: Path,
) -> None:
    """
    Minimal v0.1 verification pass:
    - Highlight detected strings using page.search_for to ensure
      we have the right text before wiring precise rectangles.
    - Once spans contain coordinates we can switch back to true redactions.
    """
    doc = fitz.open(str(input_pdf))

    for det in detections:
        page = doc[det.span.page_index]

        # search_for returns rectangles for each textual occurrence
        matches = page.search_for(det.span.text)
        for rect in matches:
            highlight = page.add_highlight_annot(rect)
            highlight.update()

    doc.save(str(output_pdf))
    doc.close()
