from pathlib import Path
from typing import List

import numpy as np

from n2n.models import TextSpan
from n2n.packs import global_card_photo_v1 as pack
from n2n.packs.photo_common import PageContext
from n2n.vision.preprocess import PreprocessOutput


class DummyBox:
    def __init__(self, label: str, bbox: tuple[float, float, float, float], conf: float = 0.9):
        self.label = label
        self._bbox = bbox
        self.conf = conf

    def as_tuple(self):
        return self._bbox


class DummyPage:
    def __init__(self):
        self.index = 0
        self.width = 400
        self.height = 250
        self.scale = 1.0
        self.render_dpi = 300
        self.source = "image"


def _make_context(boxes: List[DummyBox]) -> PageContext:
    page = DummyPage()
    image = np.zeros((250, 400, 3), dtype=np.uint8)
    return PageContext(page=page, image=image, boxes=boxes)


def _fake_preprocess(*_args, **_kwargs) -> PreprocessOutput:
    img = np.zeros((200, 320, 3), dtype=np.uint8)
    trace = {"blur_score": 80.0, "quality": {"occlusion_suspected": False}}
    return PreprocessOutput(
        image=img,
        roi_bbox=(0, 0, img.shape[1], img.shape[0]),
        forward_matrix=None,
        inverse_matrix=None,
        used_warp=False,
        trace=trace,
    )


def _mock_roi(monkeypatch, mapping):
    def _runner(preprocess, ctx, roi_box, label, config, backend_mode, prefer_digits=False):
        entry = mapping[label]
        stats = entry.get("stats", {"avg_conf": 0.95})
        text = entry.get("text", "")
        spans = []
        if text:
            spans.append(
                TextSpan(
                    text=text,
                    bbox=roi_box,
                    page=ctx.page.index,
                    source="roi_ocr",
                    ocr_conf=stats.get("avg_conf", 0.95),
                )
            )
        return pack.RoiOcrResult(
            page=ctx.page.index,
            label=label,
            roi_norm=roi_box,
            roi_page=roi_box,
            text=text,
            stats=stats,
            spans=spans,
            engine="test",
            attempts=[{"engine": "test", "success": True, "text_preview": text, "avg_conf": stats.get("avg_conf", 0.0), "elapsed_ms": 1.0}],
        )

    monkeypatch.setattr(pack, "_run_roi_ocr", _runner)


def _fake_render(path: Path) -> str:
    path.write_text("pdf")
    return str(path)


def test_card_pack_confirms_when_pan_detected(tmp_path, monkeypatch):
    input_path = tmp_path / "card.jpg"
    input_path.write_text("img")

    ctx = _make_context([DummyBox("card", (0, 0, 400, 250))])

    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "PAN ROI": {"text": "4111 1111 1111 1111", "stats": {"avg_conf": 0.95}},
            "EXPIRY ROI": {"text": "12/34", "stats": {"avg_conf": 0.9}},
        },
    )

    highlight_file = tmp_path / "highlight.pdf"
    redacted_file = tmp_path / "redacted.pdf"
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(highlight_file))
    monkeypatch.setattr(pack, "render_redact_from_boxes", lambda *args, **kwargs: _fake_render(redacted_file))
    monkeypatch.setattr(pack, "_verify_redaction", lambda *args, **kwargs: {"checked": 1, "hits_remaining": 0})

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "CONFIRMED"
    assert Path(report["artifacts"]["highlight_pdf"]).exists()
    assert Path(report["artifacts"]["redacted_pdf"]).exists()
    assert report["trace"]["post_redaction"]["hits_remaining"] == 0


def test_card_pack_review_when_luhn_fails(tmp_path, monkeypatch):
    input_path = tmp_path / "card2.jpg"
    input_path.write_text("img")
    ctx = _make_context([])
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))

    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "PAN ROI": {"text": "4111 1111 1111 1112", "stats": {"avg_conf": 0.95}},
            "EXPIRY ROI": {"text": "", "stats": {"avg_conf": 0.0}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "hl.pdf"))
    monkeypatch.setattr(pack, "_verify_redaction", lambda *args, **kwargs: {"checked": 0, "hits_remaining": 0})

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "REVIEW"
    reason_codes = {reason["code"] for reason in report["reasons"]}
    assert "PAN_SUSPECT" in reason_codes
    assert "QUALITY_LOW" not in reason_codes
    assert report["artifacts"]["redacted_pdf"] is None


def test_card_pack_review_if_redaction_remains(tmp_path, monkeypatch):
    input_path = tmp_path / "card3.jpg"
    input_path.write_text("img")
    ctx = _make_context([])
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "PAN ROI": {"text": "5555 5555 5555 4444", "stats": {"avg_conf": 0.95}},
            "EXPIRY ROI": {"text": "01/30", "stats": {"avg_conf": 0.9}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "h2.pdf"))
    monkeypatch.setattr(pack, "render_redact_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "r2.pdf"))
    monkeypatch.setattr(
        pack,
        "_verify_redaction",
        lambda *args, **kwargs: {"checked": 1, "hits_remaining": 1},
    )

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "REVIEW"
    assert report["reasons"][0]["code"] == "PAN_REMAINS"
    assert report["artifacts"]["redacted_pdf"] is None


def test_visual_pan_triggers_review_when_no_ocr(tmp_path, monkeypatch):
    input_path = tmp_path / "card4.jpg"
    input_path.write_text("img")
    ctx = _make_context([])
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "PAN ROI": {"text": "", "stats": {"avg_conf": 0.0}},
            "EXPIRY ROI": {"text": "", "stats": {"avg_conf": 0.0}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "vh.pdf"))
    monkeypatch.setattr(pack, "_verify_redaction", lambda *args, **kwargs: {"checked": 0, "hits_remaining": 0})
    visual_trace = {"visual_pan": {"digit_like_count": 12}}
    monkeypatch.setattr(
        pack,
        "detect_visual_pan_suspicion",
        lambda *args, **kwargs: ((10.0, 20.0, 110.0, 60.0), visual_trace),
    )

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "REVIEW"
    codes = {reason["code"] for reason in report["reasons"]}
    assert "PAN_SUSPECT_VISUAL" in codes
    assert report["suggested_redactions"]
    assert report["artifacts"]["redacted_pdf"] is None


def test_visual_pan_force_band_redact(tmp_path, monkeypatch):
    input_path = tmp_path / "force.jpg"
    input_path.write_text("img")
    ctx = _make_context([])
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "PAN ROI": {"text": "", "stats": {"avg_conf": 0.0}},
            "EXPIRY ROI": {"text": "", "stats": {"avg_conf": 0.0}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "vh2.pdf"))
    monkeypatch.setattr(
        pack,
        "detect_visual_pan_suspicion",
        lambda *args, **kwargs: ((15.0, 25.0, 105.0, 65.0), {"visual_pan": {"digit_like_count": 12}}),
    )
    redacted = tmp_path / "forced.pdf"
    monkeypatch.setattr(pack, "render_redact_from_boxes", lambda *args, **kwargs: _fake_render(redacted))
    monkeypatch.setattr(pack, "_verify_redaction", lambda *args, **kwargs: {"checked": 0, "hits_remaining": 0})

    report = pack.run_pack(input_path, tmp_path, force_band_redact=True)
    assert report["decision"] == "REVIEW"
    assert report["artifacts"]["redacted_pdf"].endswith("forced.pdf")
    assert report["action"] == "FORCED_REDACT_REVIEW"
