from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

from n2n.models import DetectionResult
from n2n.pipeline import _extract_with_mode, _load_configs, run_highlight
from n2n.primitives import get_primitive
from n2n.profiles import get_profile
from n2n.renderers.pdf_highlight import highlight_pdf
from n2n.renderers.pdf_mupdf import apply_redactions
from n2n.spans import build_text_spans

LOGGER = logging.getLogger(__name__)
PACK_ID = "uk.bank_statement.v1"

REASON_DESCRIPTIONS = {
    "EXTRACTION_QUALITY_LOW": "Text extraction quality fell below the configured threshold.",
    "EXTRACTION_EMPTY": "Extractor produced no spans or insufficient characters to inspect the document safely.",
    "NO_PII_FOUND": "No profile fields were detected with high enough confidence.",
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
    report_path = input_path.with_name(f"{input_path.stem}_{PACK_ID.replace('.', '_')}_report.json")
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
        pack=PACK_ID,
        decision=decision,
        reasons=reasons,
        detections=detections,
        artifacts=artifacts,
    )
    _write_report(report, input_path)
    return report


def _generate_highlight_artifact(input_path: Path, config_dir: Path) -> str:
    outcome = run_highlight(input_path, config_dir)
    if outcome.output_path:
        return str(outcome.output_path)
    LOGGER.info("Highlight pipeline skipped for %s, emitting empty artifact.", input_path)
    return str(highlight_pdf(input_path, []))


def _span_stats(spans) -> tuple[int, int]:
    total_chars = sum(len((span.text or "").strip()) for span in spans)
    return len(spans), total_chars


def _run_profile(spans, profile_id: str) -> List[DetectionResult]:
    profile = get_profile(profile_id)
    detections: List[DetectionResult] = []

    for field in profile.fields:
        primitive_fn = get_primitive(field.primitive)
        field_cfg = {
            "id": field.id,
            "category": field.category,
            **field.options,
        }
        detections.extend(primitive_fn(spans, field_cfg))

    return detections


def run_uk_bank_statement_pack(input_path: Path, config_dir: Path) -> DecisionReport:
    defaults, _ = _load_configs(config_dir)
    artifacts: Dict[str, Optional[str]] = {
        "input_pdf": str(input_path),
        "highlight_pdf": None,
        "redacted_pdf": None,
        "report_json": None,
    }

    artifacts["highlight_pdf"] = _generate_highlight_artifact(input_path, config_dir)

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
    span_count, char_count = _span_stats(spans)
    if span_count == 0 or char_count < MIN_CHAR_COUNT:
        return _finalize(
            decision="REJECTED",
            reasons=[_reason("EXTRACTION_EMPTY")],
            detections=[],
            artifacts=artifacts,
            input_path=input_path,
        )

    detections = _run_profile(spans, PACK_ID)
    hits = [det for det in detections if det.confidence >= 1.0 and det.severity == "hit"]

    if not hits:
        return _finalize(
            decision="REVIEW",
            reasons=[_reason("NO_PII_FOUND")],
            detections=detections,
            artifacts=artifacts,
            input_path=input_path,
        )

    output_pdf = input_path.with_name(f"{input_path.stem}_redacted.pdf")
    apply_redactions(input_path, hits, output_pdf)
    artifacts["redacted_pdf"] = str(output_pdf)

    return _finalize(
        decision="CONFIRMED",
        reasons=[],
        detections=detections,
        artifacts=artifacts,
        input_path=input_path,
    )


__all__ = ["DecisionReport", "run_uk_bank_statement_pack", "PACK_ID"]
