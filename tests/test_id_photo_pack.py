from pathlib import Path

import numpy as np

from n2n.models import TextSpan
from n2n.packs import global_id_photo_v1 as pack
from n2n.packs.photo_common import PageContext
from n2n.vision.preprocess import PreprocessOutput


class DummyBox:
    def __init__(self, label, bbox, conf=0.9):
        self.label = label
        self._bbox = bbox
        self.conf = conf

    def as_tuple(self):
        return self._bbox


class DummyPage:
    def __init__(self):
        self.index = 0
        self.width = 500
        self.height = 320
        self.scale = 1.0
        self.render_dpi = 300
        self.source = "image"


def _context(boxes):
    page = DummyPage()
    image = np.zeros((320, 500, 3), dtype=np.uint8)
    return PageContext(page=page, image=image, boxes=boxes)


def _fake_preprocess(*_args, **_kwargs):
    img = np.zeros((260, 420, 3), dtype=np.uint8)
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
    def _runner(preprocess, ctx, roi_box, label, config, backend_mode):
        entry = mapping[label]
        stats = entry.get("stats", {"avg_conf": 0.9})
        text = entry.get("text", "")
        spans = []
        if text:
            spans.append(
                TextSpan(
                    text=text,
                    bbox=roi_box,
                    page=ctx.page.index,
                    source="roi_ocr",
                    ocr_conf=stats.get("avg_conf", 0.9),
                )
            )
        return pack.RoiOcrRecord(
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


def test_id_pack_confirms_on_mrz(tmp_path, monkeypatch):
    input_path = tmp_path / "id.jpg"
    input_path.write_text("img")
    boxes = [DummyBox("id_card", (0, 0, 500, 320))]
    ctx = _context(boxes)
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    monkeypatch.setattr(pack, "preprocess_document_region", _fake_preprocess)
    mrz_text = "P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<<\n1234567890GBR7411250M2001012<<<<<<<<<4"
    _mock_roi(
        monkeypatch,
        {
            "MRZ": {"text": mrz_text, "stats": {"avg_conf": 0.9}},
            "ID NUMBER": {"text": "ID123456", "stats": {"avg_conf": 0.8}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "hl.pdf"))
    monkeypatch.setattr(pack, "render_redact_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "rd.pdf"))
    monkeypatch.setattr(
        pack,
        "_verify_redaction",
        lambda *args, **kwargs: {"checked": 1, "mrz_hits_remaining": 0},
    )

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "CONFIRMED"
    assert Path(report["artifacts"]["redacted_pdf"]).exists()
    assert report["trace"]["post_redaction"]["mrz_hits_remaining"] == 0


def test_id_pack_review_when_only_id_suspect(tmp_path, monkeypatch):
    input_path = tmp_path / "id2.jpg"
    input_path.write_text("img")
    ctx = _context([])
    monkeypatch.setattr(pack, "load_page_contexts", lambda *args, **kwargs: ([ctx], {"page_count": 1}, {"model_used": False}))
    bad_trace = {"blur_score": 5.0, "quality": {"occlusion_suspected": True}}

    def low_preprocess(*_args, **_kwargs):
        out = _fake_preprocess()
        out.trace = bad_trace
        return out

    monkeypatch.setattr(pack, "preprocess_document_region", low_preprocess)
    _mock_roi(
        monkeypatch,
        {
            "MRZ": {"text": "", "stats": {"avg_conf": 0.0}},
            "ID NUMBER": {"text": "ABC12345", "stats": {"avg_conf": 0.6}},
        },
    )
    monkeypatch.setattr(pack, "render_highlight_from_boxes", lambda *args, **kwargs: _fake_render(tmp_path / "hl2.pdf"))

    report = pack.run_pack(input_path, tmp_path)
    assert report["decision"] == "REVIEW"
    reason_codes = {reason["code"] for reason in report["reasons"]}
    assert "ID_SUSPECT" in reason_codes
    assert "OCCLUSION" in reason_codes
