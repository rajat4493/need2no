import logging
import re
from pathlib import Path
from typing import List, Tuple

from n2n import DEFAULT_QUALITY_THRESHOLD
from n2n.detectors.bank_statement_uk import detect_pii_uk_bank_statement
from n2n.extractors.pdf_ocr import extract_text_with_quality_ocr
from n2n.extractors.pdf_text import extract_text_with_quality
from n2n.models import DetectionResult, ExtractionResult, PiiCategory, RedactionOutcome
from n2n.renderers.pdf_highlight import highlight_pdf
from n2n.renderers.pdf_mupdf import apply_redactions
from n2n.utils.config_loader import load_global_config, load_profile_config

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGGER = logging.getLogger(__name__)
CURRENCY_RE = re.compile(
    r"""^[£]?\s*
        -?\d{1,3}
        (,\d{3})*
        (\.\d{2})?
        \s*$""",
    re.VERBOSE,
)


def _load_configs(config_dir: Path) -> Tuple[dict, dict]:
    defaults = load_global_config(config_dir)
    profile = load_profile_config(config_dir, defaults["country_pack"], defaults["profile"])
    return defaults, profile


def _get_threshold(defaults: dict) -> float:
    threshold = defaults.get("quality_threshold", DEFAULT_QUALITY_THRESHOLD)
    try:
        return float(threshold)
    except (TypeError, ValueError):
        return DEFAULT_QUALITY_THRESHOLD


def _get_output_suffix(defaults: dict) -> str:
    output = defaults.get("output", {})
    suffix = output.get("suffix")
    return str(suffix) if suffix else "_redacted"


def _extract_with_mode(
    input_path: Path,
    defaults: dict,
) -> Tuple[ExtractionResult | None, str | None]:
    mode = str(defaults.get("extractor_mode", "auto")).lower()
    threshold = _get_threshold(defaults)

    def _is_valid(result: ExtractionResult) -> bool:
        return result.quality_score >= threshold

    if mode == "text":
        text_result = extract_text_with_quality(input_path)
        if _is_valid(text_result):
            return text_result, None
        return None, "quality_too_low"

    if mode == "ocr":
        ocr_result = extract_text_with_quality_ocr(input_path)
        if _is_valid(ocr_result):
            return ocr_result, None
        return None, "ocr_quality_too_low"

    # auto mode
    text_result = extract_text_with_quality(input_path)
    if _is_valid(text_result):
        return text_result, None

    ocr_result = extract_text_with_quality_ocr(input_path)
    if _is_valid(ocr_result):
        return ocr_result, None

    return None, "ocr_quality_too_low"


def _prepare_detections(
    input_path: Path, config_dir: Path
) -> Tuple[dict, List[DetectionResult], str | None]:
    defaults, profile = _load_configs(config_dir)

    extraction, extraction_reason = _extract_with_mode(input_path, defaults)

    if extraction is None:
        return defaults, [], extraction_reason or "quality_too_low"

    detections = detect_pii_uk_bank_statement(extraction, profile)
    strict_detections = [d for d in detections if d.confidence >= 1.0 and d.severity == "hit"]
    strict_detections, dropped = _precision_filter(strict_detections)
    if dropped:
        LOGGER.info("Precision filter dropped %s detections", dropped)

    if not strict_detections:
        return defaults, [], "no_pii_found"

    return defaults, strict_detections, None


def run_pipeline(input_path: Path, config_dir: Path) -> RedactionOutcome:
    defaults, strict_detections, reason = _prepare_detections(input_path, config_dir)

    if reason:
        return RedactionOutcome(
            input_path=input_path,
            output_path=None,
            redactions_applied=0,
            reason=reason,
        )

    suffix = _get_output_suffix(defaults)
    output_path = input_path.with_name(f"{input_path.stem}{suffix}.pdf")

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


def run_highlight(input_path: Path, config_dir: Path) -> RedactionOutcome:
    defaults, strict_detections, reason = _prepare_detections(input_path, config_dir)

    if reason:
        return RedactionOutcome(
            input_path=input_path,
            output_path=None,
            redactions_applied=0,
            reason=reason,
        )

    output_path = highlight_pdf(input_path, strict_detections)

    return RedactionOutcome(
        input_path=input_path,
        output_path=output_path,
        redactions_applied=len(strict_detections),
        reason=None,
    )


def redact_file(input_path: Path) -> RedactionOutcome:
    """Backward-compatible helper that uses the project config directory."""

    return run_pipeline(input_path, PROJECT_ROOT)


def _precision_filter(detections: List[DetectionResult]) -> Tuple[List[DetectionResult], int]:
    filtered: List[DetectionResult] = []
    dropped = 0

    for det in detections:
        text = (det.span.text or "").strip()
        if det.category == PiiCategory.BANK_IDENTIFIERS and looks_like_currency(text):
            dropped += 1
            continue
        filtered.append(det)

    return filtered, dropped


def looks_like_currency(text: str) -> bool:
    return bool(CURRENCY_RE.match(text.strip()))
