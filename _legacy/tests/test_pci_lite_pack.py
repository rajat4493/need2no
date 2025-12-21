from __future__ import annotations

from pathlib import Path

import fitz
import pytest

from n2n.models import DetectionResult, ExtractionResult, PiiCategory, RedactionOutcome, TextSpan
from n2n.packs import global_pci_lite_v1 as pci_lite


def _extraction(path: Path, text: str | None = None, source: str = "text") -> ExtractionResult:
    payload = text or "Card data " * 5
    return ExtractionResult(file_path=path, quality_score=0.9, pages=[payload], source=source)


def _detection(severity: str, source: str = "text") -> DetectionResult:
    span = TextSpan(page_index=0, text="4242 4242 4242 4242", bbox=(0, 0, 1, 1), source=source)
    return DetectionResult(
        field_id="card_pan",
        category=PiiCategory.CARD_NUMBERS,
        primitive="card_pan",
        span=span,
        confidence=1.0,
        context="",
        raw_text="4242424242424242",
        masked_text="**** **** **** 4242",
        source=source,
        validators=["luhn"] if severity == "hit" else [],
        severity=severity,
    )


def test_suspicion_triggers_review(monkeypatch, tmp_path):
    input_pdf = tmp_path / "sample.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    defaults = {"quality_threshold": 0.5}
    monkeypatch.setattr(pci_lite, "_load_configs", lambda config_dir: (defaults, {}))
    monkeypatch.setattr(pci_lite, "_extract_with_mode", lambda path, defaults: (_extraction(path), None))
    monkeypatch.setattr(
        pci_lite,
        "_generate_highlight_artifact",
        lambda input_path, config_dir: str(input_path.with_name("sample_highlighted.pdf")),
    )

    suspicion = _detection("suspicion", source="ocr")
    monkeypatch.setattr(pci_lite, "_run_detection", lambda extraction: [suspicion])

    report = pci_lite.run_pci_lite_pack(input_pdf, tmp_path)

    assert report.decision == "REVIEW"
    assert report.reasons[0].code == "PAN_SUSPECT_OCR_LOW_CONF"
    assert report.artifacts["highlight_pdf"].endswith("sample_highlighted.pdf")
    assert report.artifacts["redacted_pdf"] is None
    assert Path(report.artifacts["report_json"]).exists()
    assert report.artifacts["ocr_text"] is None
    assert report.artifacts["ocr_spans"] is None


def test_remaining_pan_after_redaction_forces_review(monkeypatch, tmp_path):
    input_pdf = tmp_path / "card.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    defaults = {"quality_threshold": 0.5}
    monkeypatch.setattr(pci_lite, "_load_configs", lambda config_dir: (defaults, {}))
    monkeypatch.setattr(pci_lite, "_extract_with_mode", lambda path, defaults: (_extraction(path), None))
    monkeypatch.setattr(
        pci_lite,
        "_generate_highlight_artifact",
        lambda input_path, config_dir: str(input_path.with_name("card_highlighted.pdf")),
    )

    call_state = {"count": 0}

    def _fake_detect(extraction):
        if call_state["count"] == 0:
            call_state["count"] += 1
            return [_detection("hit")]
        return [_detection("hit")]

    monkeypatch.setattr(pci_lite, "_run_detection", _fake_detect)

    redacted_path = input_pdf.with_name("card_redacted.pdf")
    redacted_path.write_bytes(b"%PDF-1.7")
    redaction_outcome = RedactionOutcome(
        input_path=input_pdf,
        output_path=redacted_path,
        redactions_applied=1,
        reason=None,
    )
    monkeypatch.setattr(pci_lite, "_ensure_redacted_artifact", lambda input_path, config_dir: redaction_outcome)

    report = pci_lite.run_pci_lite_pack(input_pdf, tmp_path)

    assert report.decision == "REVIEW"
    assert report.reasons[0].code == "PAN_REMAINS_AFTER_REDACTION"
    assert report.artifacts["redacted_pdf"] == str(redacted_path)


def test_clean_run_is_confirmed(monkeypatch, tmp_path):
    input_pdf = tmp_path / "ok.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    defaults = {"quality_threshold": 0.5}
    monkeypatch.setattr(pci_lite, "_load_configs", lambda config_dir: (defaults, {}))
    monkeypatch.setattr(pci_lite, "_extract_with_mode", lambda path, defaults: (_extraction(path), None))
    monkeypatch.setattr(
        pci_lite,
        "_generate_highlight_artifact",
        lambda input_path, config_dir: str(input_path.with_name("ok_highlighted.pdf")),
    )

    call_state = {"count": 0}

    def _fake_detect(extraction):
        if call_state["count"] == 0:
            call_state["count"] += 1
            return [_detection("hit")]
        return []

    monkeypatch.setattr(pci_lite, "_run_detection", _fake_detect)

    redacted_path = input_pdf.with_name("ok_redacted.pdf")
    redacted_path.write_bytes(b"%PDF-1.7")
    redaction_outcome = RedactionOutcome(
        input_path=input_pdf,
        output_path=redacted_path,
        redactions_applied=1,
        reason=None,
    )
    monkeypatch.setattr(pci_lite, "_ensure_redacted_artifact", lambda input_path, config_dir: redaction_outcome)

    report = pci_lite.run_pci_lite_pack(input_pdf, tmp_path)

    assert report.decision == "CONFIRMED"
    assert report.reasons == []
    assert report.artifacts["redacted_pdf"] == str(redacted_path)
    assert report.artifacts["highlight_pdf"].endswith("ok_highlighted.pdf")
    assert report.artifacts["ocr_text"] is None
    assert report.artifacts["ocr_spans"] is None


def test_empty_extraction_rejected(monkeypatch, tmp_path):
    input_pdf = tmp_path / "empty.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    defaults = {"quality_threshold": 0.5}
    monkeypatch.setattr(pci_lite, "_load_configs", lambda config_dir: (defaults, {}))
    monkeypatch.setattr(pci_lite, "_generate_highlight_artifact", lambda input_path, config_dir: "empty_highlighted.pdf")
    monkeypatch.setattr(pci_lite, "_extract_with_mode", lambda path, defaults: (_extraction(path, text="  "), None))

    report = pci_lite.run_pci_lite_pack(input_pdf, tmp_path)
    assert report.decision == "REJECTED"
    assert report.reasons[0].code == "EXTRACTION_EMPTY"


def test_ocr_artifacts_written(monkeypatch, tmp_path):
    input_pdf = tmp_path / "ocr.pdf"
    input_pdf.write_bytes(b"%PDF-1.7")

    defaults = {"quality_threshold": 0.5}
    monkeypatch.setattr(pci_lite, "_load_configs", lambda config_dir: (defaults, {}))
    monkeypatch.setattr(pci_lite, "_generate_highlight_artifact", lambda input_path, config_dir: "ocr_highlighted.pdf")
    extraction = _extraction(input_pdf, text="OCR digits 4111 1111 1111 1111", source="ocr")
    monkeypatch.setattr(pci_lite, "_extract_with_mode", lambda path, defaults: (extraction, None))
    monkeypatch.setattr(pci_lite, "_run_detection", lambda extraction: [])

    report = pci_lite.run_pci_lite_pack(input_pdf, tmp_path)

    assert report.decision == "CONFIRMED"
    assert report.artifacts["ocr_text"]
    assert report.artifacts["ocr_spans"]
    assert Path(report.artifacts["ocr_text"]).exists()
    assert Path(report.artifacts["ocr_spans"]).exists()


def _create_pdf(path: Path, text: str) -> None:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(str(path))
    doc.close()


def test_full_pack_generates_artifacts(tmp_path):
    pdf_path = tmp_path / "integration.pdf"
    _create_pdf(pdf_path, "Card 4111 1111 1111 1111 for testing.")

    config_dir = Path(__file__).resolve().parent.parent
    report = pci_lite.run_pci_lite_pack(pdf_path, config_dir)

    assert report.decision == "CONFIRMED"
    assert report.artifacts["highlight_pdf"]
    assert report.artifacts["redacted_pdf"]
    assert report.artifacts["report_json"]

    assert Path(report.artifacts["highlight_pdf"]).exists()
    assert Path(report.artifacts["redacted_pdf"]).exists()
    assert Path(report.artifacts["report_json"]).exists()
