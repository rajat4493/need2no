from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

from n2n import ENGINE_VERSION
from n2n.extract import extract_spans
from n2n.models import DecisionReason, DecisionReport, DetectionResult
from n2n.primitives.card_pan import CardPanConfig, find_card_pans
from n2n.render import render_highlight, render_redact
from PIL import Image

PACK_ID = "global.pci_lite.v1"
MIN_CHAR_COUNT = 30
CARD_PAN_CFG = CardPanConfig(allow_lowercase_b_to_6=True)


@dataclass
class PackArtifacts:
    input_pdf: Path
    outdir: Path

    def highlight_path(self) -> Path:
        return self.outdir / f"{self.input_pdf.stem}_highlight.pdf"

    def redacted_path(self) -> Path:
        return self.outdir / f"{self.input_pdf.stem}_redacted.pdf"

    def report_path(self) -> Path:
        return self.outdir / f"{self.input_pdf.stem}_report.json"

    def ocr_text_path(self) -> Path:
        return self.outdir / f"{self.input_pdf.stem}_ocr_text.txt"

    def ocr_spans_path(self) -> Path:
        return self.outdir / f"{self.input_pdf.stem}_ocr_spans.json"


REASONS = {
    "EXTRACTION_EMPTY": "Extraction produced no spans or insufficient characters to inspect.",
    "PAN_SUSPECT_OCR_LOW_CONF": "OCR-derived PAN candidate failed Luhn with low confidence.",
    "PAN_REMAINS_AFTER_REDACTION": "PAN still detectable after redaction.",
}


def _build_report(
    decision: str,
    reasons: List[DecisionReason],
    detections: List[DetectionResult],
    artifacts: Dict[str, str | None],
    trace: Dict[str, object],
    action: str | None = None,
) -> DecisionReport:
    return DecisionReport(
        pack_id=PACK_ID,
        decision=decision,
        reasons=reasons,
        detections=detections,
        artifacts=artifacts,
        engine_version=ENGINE_VERSION,
        action=action,
        trace=trace,
    )


def run_pack(input_pdf: Path, outdir: Path, **_: object) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = PackArtifacts(input_pdf=input_pdf, outdir=outdir)

    spans, debug_artifacts, extract_stats = extract_spans(str(input_pdf), artifact_dir=outdir)
    artifact_map: Dict[str, str | None] = {
        "input_pdf": str(input_pdf),
        "highlight_pdf": None,
        "redacted_pdf": None,
        "report_json": None,
        "ocr_text": debug_artifacts.get("ocr_text"),
        "ocr_spans": debug_artifacts.get("ocr_spans"),
    }
    render_source = _ensure_pdf_source(input_pdf, outdir)
    trace: Dict[str, object] = {
        "extraction": {
            "span_count": extract_stats.get("span_count", 0),
            "text_span_count": extract_stats.get("text_span_count", 0),
            "ocr_span_count": extract_stats.get("ocr_span_count", 0),
            "used_ocr": extract_stats.get("used_ocr", False),
        }
    }
    card_trace: Dict[str, object] = {}

    if sum(len(span.text.strip()) for span in spans) < MIN_CHAR_COUNT:
        highlight = render_highlight(render_source, [], artifacts.highlight_path())
        artifact_map["highlight_pdf"] = highlight
        trace["card_pan"] = card_trace
        trace["post_redaction"] = {"hits_remaining": 0}
        report = _build_report(
            decision="REJECTED",
            reasons=[DecisionReason(code="EXTRACTION_EMPTY", description=REASONS["EXTRACTION_EMPTY"])],
            detections=[],
            artifacts=artifact_map,
            trace=trace,
        )
        _write_report(report, artifacts.report_path())
        return report.to_dict()

    detections = find_card_pans(spans, CARD_PAN_CFG, trace=card_trace)
    trace["card_pan"] = card_trace
    highlight_path = render_highlight(render_source, detections, artifacts.highlight_path())
    artifact_map["highlight_pdf"] = highlight_path

    suspicions = [det for det in detections if det.severity == "suspicion"]
    if suspicions:
        trace["post_redaction"] = {"hits_remaining": 0}
        report = _build_report(
            decision="REVIEW",
            reasons=[
                DecisionReason(
                    code="PAN_SUSPECT_OCR_LOW_CONF",
                    description=REASONS["PAN_SUSPECT_OCR_LOW_CONF"],
                )
            ],
            detections=detections,
            artifacts=artifact_map,
            trace=trace,
        )
        _write_report(report, artifacts.report_path())
        return report.to_dict()

    hits = [det for det in detections if det.severity == "hit"]
    if hits:
        redacted_path = render_redact(render_source, hits, artifacts.redacted_path())
        artifact_map["redacted_pdf"] = redacted_path

        re_spans, _, re_stats = extract_spans(redacted_path, artifact_dir=outdir)
        re_trace: Dict[str, object] = {}
        re_detections = find_card_pans(re_spans, CARD_PAN_CFG, trace=re_trace)
        trace["post_redaction"] = {
            "span_count": re_stats.get("span_count", 0),
            "hits_remaining": len([det for det in re_detections if det.severity == "hit"]),
            "suspicions_remaining": len([det for det in re_detections if det.severity == "suspicion"]),
        }
        if re_detections:
            report = _build_report(
                decision="REVIEW",
                reasons=[
                    DecisionReason(
                        code="PAN_REMAINS_AFTER_REDACTION",
                        description=REASONS["PAN_REMAINS_AFTER_REDACTION"],
                    )
                ],
                detections=detections,
                artifacts=artifact_map,
                trace=trace,
            )
            _write_report(report, artifacts.report_path())
            return report.to_dict()

        report = _build_report(
            decision="CONFIRMED",
            reasons=[],
            detections=detections,
            artifacts=artifact_map,
            trace=trace,
        )
        _write_report(report, artifacts.report_path())
        return report.to_dict()

    trace["post_redaction"] = {"hits_remaining": 0}
    report = _build_report(
        decision="CONFIRMED",
        reasons=[],
        detections=detections,
        artifacts=artifact_map,
        trace=trace,
    )
    _write_report(report, artifacts.report_path())
    return report.to_dict()


def _write_report(report: DecisionReport, path: Path) -> None:
    report.artifacts["report_json"] = str(path)
    path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")


def _ensure_pdf_source(input_path: Path, outdir: Path) -> Path:
    if input_path.suffix.lower() == ".pdf":
        return input_path
    converted = outdir / f"{input_path.stem}_source.pdf"
    if converted.exists() and converted.stat().st_mtime >= input_path.stat().st_mtime:
        return converted
    image = Image.open(str(input_path))
    if image.mode not in ("RGB", "CMYK"):
        image = image.convert("RGB")
    image.save(str(converted), "PDF", resolution=300.0)
    return converted


__all__ = ["run_pack", "PACK_ID"]
