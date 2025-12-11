from pathlib import Path
from typing import Optional

from n2n import DEFAULT_QUALITY_THRESHOLD
from n2n.detectors.bank_statement_uk import detect_pii_uk_bank_statement
from n2n.models import RedactionOutcome
from n2n.parsers.pdf_text_extractor import extract_text_with_quality
from n2n.renderers.pdf_renderer import apply_redactions


def redact_file(input_path: Path) -> RedactionOutcome:
    extraction = extract_text_with_quality(input_path)

    if extraction.quality_score < DEFAULT_QUALITY_THRESHOLD:
        return RedactionOutcome(
            input_path=input_path,
            output_path=None,
            redactions_applied=0,
            reason="quality_too_low",
        )

    detections = detect_pii_uk_bank_statement(extraction)

    # Strict rule: only redact if we have at least one detection with confidence == 1.0
    strict_detections = [d for d in detections if d.confidence >= 1.0]

    if not strict_detections:
        return RedactionOutcome(
            input_path=input_path,
            output_path=None,
            redactions_applied=0,
            reason="no_pii_found",
        )

    output_path = input_path.with_name(input_path.stem + "_redacted.pdf")

    apply_redactions(
        input_pdf=input_path,
        detections=strict_detections,
        output_pdf=output_path,
    )

    return RedactionOutcome(
        input_path=input_path,
        output_path=output_path,
        redactions_applied=len(strict_detections),
        reason=None,
    )
