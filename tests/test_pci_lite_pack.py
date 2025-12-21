from pathlib import Path

import fitz

from n2n.models import DetectionResult, TextSpan
from n2n.packs import global_pci_lite_v1 as pack


def _span(text="4242 4242 4242 4242", source="text", conf=1.0):
    return TextSpan(text=text, bbox=(0, 0, 1, 1), page=0, source=source, ocr_conf=conf)


def _det(severity="hit") -> DetectionResult:
    return DetectionResult(
        field_id="card_pan",
        text="**** **** **** 4242",
        raw="4242424242424242",
        bbox=(0, 0, 1, 1),
        page=0,
        source="text",
        validators=["luhn"] if severity == "hit" else ["regex"],
        severity=severity,
    )


def _patch_render(monkeypatch, tmp_path):
    monkeypatch.setattr(pack, "render_highlight", lambda *args, **kwargs: str(tmp_path / "highlight.pdf"))
    monkeypatch.setattr(pack, "render_redact", lambda *args, **kwargs: str(tmp_path / "redacted.pdf"))


def _extract_result(spans, used_ocr=False):
    stats = {
        "used_ocr": used_ocr,
        "span_count": len(spans),
        "text_span_count": sum(1 for s in spans if (s.source or "") == "text"),
        "ocr_span_count": sum(1 for s in spans if (s.source or "").lower() == "ocr"),
    }
    return (spans, {}, stats)


def test_suspicion_triggers_review(tmp_path, monkeypatch):
    pdf = tmp_path / "suspicion.pdf"
    pdf.write_text("dummy")

    monkeypatch.setattr(pack, "MIN_CHAR_COUNT", 0)
    monkeypatch.setattr(
        pack,
        "extract_spans",
        lambda *args, **kwargs: _extract_result([_span(source="ocr", conf=0.5)], used_ocr=True),
    )
    monkeypatch.setattr(pack, "find_card_pans", lambda spans, cfg=None, trace=None: [_det("suspicion")])
    _patch_render(monkeypatch, tmp_path)

    report = pack.run_pack(pdf, tmp_path)
    assert report["decision"] == "REVIEW"
    assert report["reasons"][0]["code"] == "PAN_SUSPECT_OCR_LOW_CONF"
    assert report["artifacts"]["highlight_pdf"].endswith("highlight.pdf")
    assert report["artifacts"]["redacted_pdf"] is None
    assert report["trace"]["extraction"]["used_ocr"] is True
    assert "card_pan" in report["trace"]


def test_hits_confirmed_when_redaction_clean(tmp_path, monkeypatch):
    pdf = tmp_path / "confirm.pdf"
    pdf.write_text("dummy")

    monkeypatch.setattr(pack, "MIN_CHAR_COUNT", 0)
    call_state = {"count": 0}

    def fake_extract(path, artifact_dir=None):
        call_state["count"] += 1
        return _extract_result([_span()], used_ocr=False)

    monkeypatch.setattr(pack, "extract_spans", fake_extract)

    def fake_find(spans, cfg=None, trace=None):
        if call_state["count"] == 1:
            return [_det("hit")]
        return []

    monkeypatch.setattr(pack, "find_card_pans", fake_find)
    _patch_render(monkeypatch, tmp_path)

    report = pack.run_pack(pdf, tmp_path)
    assert report["decision"] == "CONFIRMED"
    assert report["artifacts"]["redacted_pdf"].endswith("redacted.pdf")
    assert report["trace"]["post_redaction"]["hits_remaining"] == 0


def test_remaining_hits_after_redaction_review(tmp_path, monkeypatch):
    pdf = tmp_path / "review.pdf"
    pdf.write_text("dummy")

    monkeypatch.setattr(pack, "MIN_CHAR_COUNT", 0)
    call_state = {"count": 0}

    def fake_extract(path, artifact_dir=None):
        call_state["count"] += 1
        return _extract_result([_span()], used_ocr=False)

    monkeypatch.setattr(pack, "extract_spans", fake_extract)
    monkeypatch.setattr(pack, "find_card_pans", lambda spans, cfg=None, trace=None: [_det("hit")])
    _patch_render(monkeypatch, tmp_path)

    report = pack.run_pack(pdf, tmp_path)
    assert report["decision"] == "REVIEW"
    assert report["reasons"][0]["code"] == "PAN_REMAINS_AFTER_REDACTION"
    assert report["trace"]["post_redaction"]["hits_remaining"] >= 1


def _create_pdf(path: Path, text: str):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 720), text, fontsize=14)
    doc.save(path)
    doc.close()


def test_full_pack_generates_artifacts(tmp_path):
    pdf_path = tmp_path / "integration.pdf"
    _create_pdf(pdf_path, "Card 4111 1111 1111 1111 for testing.")

    report = pack.run_pack(pdf_path, tmp_path)

    artifacts = report["artifacts"]
    assert Path(artifacts["highlight_pdf"]).exists()
    assert Path(artifacts["redacted_pdf"]).exists()
    assert Path(artifacts["report_json"]).exists()
    assert report["decision"] == "CONFIRMED"
    assert "trace" in report
    assert report["trace"]["extraction"]["span_count"] > 0
