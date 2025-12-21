from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from n2n.models import DetectionResult, PiiCategory, RedactionOutcome, TextSpan
from n2n.pipeline import _extract_with_mode, _load_configs, run_highlight, run_pipeline
from n2n.primitives.card_pan import detect_card_pan
from n2n.spans import build_text_spans
from n2n.renderers.pdf_highlight import highlight_pdf

LOGGER = logging.getLogger(__name__)
PACK_NAME = "global.pci_lite.v1"

CARD_CFG = {
    "id": "card_pan",
    "category": PiiCategory.CARD_NUMBERS,
}

REASON_DESCRIPTIONS = {
    "PAN_SUSPECT_OCR_LOW_CONF": "OCR-based PAN candidate failed Luhn with low confidence and needs manual review.",
    "PAN_REMAINS_AFTER_REDACTION": "One or more PAN values were still detectable after the redaction pass.",
    "EXTRACTION_QUALITY_LOW": "Text extraction quality fell below the configured threshold.",
    "EXTRACTION_EMPTY": "Extractor produced no spans or insufficient characters to inspect the document safely.",
}

MIN_CHAR_COUNT = 30


@dataclass
class DecisionReason:
    code: str
    description: str


@dataclass
class DecisionReport:
    pack: str
    decision: str
    reasons: List[DecisionReason]
    detections: List[DetectionResult]
    artifacts: Dict[str, Optional[str]]

    def to_dict(self) -> Dict[str, object]:
        return {
            "pack": self.pack,
            "decision": self.decision,
            "reasons": [asdict(reason) for reason in self.reasons],
            "detections": [_serialize_detection(det) for det in self.detections],
            "artifacts": dict(self.artifacts),
        }


def _serialize_detection(det: DetectionResult) -> Dict[str, object]:
    return {
        "field_id": det.field_id,
        "primitive": det.primitive,
        "category": det.category.value,
        "context": det.context,
        "span": {
            "page_index": det.span.page_index,
            "text": det.span.text,
            "bbox": det.span.bbox,
            "source": det.span.source,
            "ocr_confidence": det.span.ocr_confidence,
        },
        "raw_text": det.raw_text,
        "masked_text": det.masked_text,
        "source": det.source,
        "validators": det.validators or [],
        "severity": det.severity,
    }


def _reason(code: str) -> DecisionReason:
    return DecisionReason(code=code, description=REASON_DESCRIPTIONS[code])


def _write_report(report: DecisionReport, input_path: Path) -> None:
    report_path = input_path.with_name(f"{input_path.stem}_{PACK_NAME}_report.json")
    report.artifacts["report_json"] = str(report_path)
    report_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _finalize(
    decision: str,
    reasons: List[DecisionReason],
    detections: List[DetectionResult],
    artifacts: Dict[str, Optional[str]],
    input_path: Path,
) -> DecisionReport:
    report = DecisionReport(
        pack=PACK_NAME,
        decision=decision,
        reasons=reasons,
        detections=detections,
        artifacts=artifacts,
    )
    _write_report(report, input_path)
    return report


def _run_detection(extraction) -> List[DetectionResult]:
    spans = build_text_spans(extraction)
    return detect_card_pan(spans, CARD_CFG)


def _generate_highlight_artifact(input_path: Path, config_dir: Path) -> str:
    highlight_outcome = run_highlight(input_path, config_dir)
    if highlight_outcome.output_path:
        return str(highlight_outcome.output_path)

    LOGGER.info(
        "Highlight pipeline skipped for %s (reason=%s). Producing passthrough highlight artifact.",
        input_path,
        highlight_outcome.reason,
    )
    fallback_path = highlight_pdf(input_path, [])
    return str(fallback_path)


def _ensure_redacted_artifact(
    input_path: Path,
    config_dir: Path,
) -> Optional[RedactionOutcome]:
    outcome = run_pipeline(input_path, config_dir)
    if outcome.reason or not outcome.output_path:
        LOGGER.warning("Redaction pipeline failed for %s (reason=%s)", input_path, outcome.reason)
        return None
    return outcome


def _maybe_write_ocr_artifacts(
    input_path: Path,
    extraction,
    spans: List[TextSpan],
    artifacts: Dict[str, Optional[str]],
) -> None:
    if (extraction.source or "").lower() != "ocr":
        return

    text_path = input_path.with_name(f"{input_path.stem}_ocr_text.txt")
    text_path.write_text("\n\n".join(extraction.pages), encoding="utf-8")
    artifacts["ocr_text"] = str(text_path)

    span_records = []
    for span in spans:
        span_records.append(
            {
                "page_index": span.page_index,
                "text": span.text.strip(),
                "bbox": span.bbox,
                "source": span.source,
                "ocr_confidence": span.ocr_confidence,
            }
        )

    spans_path = input_path.with_name(f"{input_path.stem}_ocr_spans.json")
    spans_path.write_text(json.dumps(span_records, indent=2), encoding="utf-8")
    artifacts["ocr_spans"] = str(spans_path)


def _span_stats(spans: List[TextSpan]) -> tuple[int, int]:
    total_chars = sum(len((span.text or "").strip()) for span in spans)
    return len(spans), total_chars


def run_pci_lite_pack(input_path: Path, config_dir: Path) -> DecisionReport:
    defaults, _ = _load_configs(config_dir)
    artifacts: Dict[str, Optional[str]] = {
        "input_pdf": str(input_path),
        "highlight_pdf": None,
        "redacted_pdf": None,
        "report_json": None,
        "ocr_text": None,
        "ocr_spans": None,
    }

    # Produce highlight artifact up front.
    try:
        artifacts["highlight_pdf"] = _generate_highlight_artifact(input_path, config_dir)
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.exception("Highlight generation failed: %s", exc)
        artifacts["highlight_pdf"] = None

    extraction, extraction_reason = _extract_with_mode(input_path, defaults)
    if extraction is None or extraction_reason:
        return _finalize(
            decision="REJECTED",
            reasons=[_reason("EXTRACTION_QUALITY_LOW")],
            detections=[],
            artifacts=artifacts,
            input_path=input_path,
        )

    spans = build_text_spans(extraction)
    _maybe_write_ocr_artifacts(input_path, extraction, spans, artifacts)

    span_count, char_count = _span_stats(spans)
    if span_count == 0 or char_count < MIN_CHAR_COUNT:
        return _finalize(
            decision="REJECTED",
            reasons=[_reason("EXTRACTION_EMPTY")],
            detections=[],
            artifacts=artifacts,
            input_path=input_path,
        )

    detections = _run_detection(extraction)
    suspicions = [det for det in detections if det.severity == "suspicion"]
    if suspicions:
        return _finalize(
            decision="REVIEW",
            reasons=[_reason("PAN_SUSPECT_OCR_LOW_CONF")],
            detections=detections,
            artifacts=artifacts,
            input_path=input_path,
        )

    hits = [det for det in detections if det.severity == "hit"]

    if hits:
        redaction_outcome = _ensure_redacted_artifact(input_path, config_dir)
        if not redaction_outcome or not redaction_outcome.output_path:
            return _finalize(
                decision="REVIEW",
                reasons=[_reason("PAN_REMAINS_AFTER_REDACTION")],
                detections=detections,
                artifacts=artifacts,
                input_path=input_path,
            )

        artifacts["redacted_pdf"] = str(redaction_outcome.output_path)

        re_extraction, _ = _extract_with_mode(Path(redaction_outcome.output_path), defaults)
        if re_extraction is None:
            return _finalize(
                decision="REVIEW",
                reasons=[_reason("PAN_REMAINS_AFTER_REDACTION")],
                detections=detections,
                artifacts=artifacts,
                input_path=input_path,
            )

        post_detections = _run_detection(re_extraction)
        remaining_hits = [det for det in post_detections if det.severity == "hit"]
        if remaining_hits:
            return _finalize(
                decision="REVIEW",
                reasons=[_reason("PAN_REMAINS_AFTER_REDACTION")],
                detections=detections,
                artifacts=artifacts,
                input_path=input_path,
            )

    return _finalize(
        decision="CONFIRMED",
        reasons=[],
        detections=detections,
        artifacts=artifacts,
        input_path=input_path,
    )


__all__ = ["DecisionReport", "run_pci_lite_pack"]
