from types import SimpleNamespace

import numpy as np

from n2n.ocr import registry


class _StubBackend:
    def __init__(self, name: str, available: bool = True):
        self.name = name
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def ocr_roi(self, image, roi_bbox, config):
        raise RuntimeError("not implemented")


def _factory(name: str, available: bool):
    def _create():
        return _StubBackend(name, available)

    return _create


def test_resolve_backend_mode_env(monkeypatch):
    monkeypatch.delenv("N2N_OCR_BACKEND", raising=False)
    assert registry.resolve_backend_mode(None) == "auto"
    monkeypatch.setenv("N2N_OCR_BACKEND", "tesseract")
    assert registry.resolve_backend_mode(None) == "tesseract"
    assert registry.resolve_backend_mode("easy") == "easy"
    assert registry.resolve_backend_mode("UNKNOWN") == "auto"


def test_get_backends_for_mode_prefers_available(monkeypatch):
    monkeypatch.setattr(
        registry,
        "_FACTORIES",
        {
            "apple": _factory("apple", False),
            "paddle": _factory("paddle", True),
            "easy": _factory("easy", False),
            "tesseract": _factory("tesseract", True),
        },
    )
    monkeypatch.setattr(registry.sys, "platform", "darwin")
    backends = registry.get_backends_for_mode("combo")
    assert [backend.name for backend in backends] == ["paddle", "tesseract"]


def test_run_ocr_backends_falls_back_to_tesseract(monkeypatch):
    class AlwaysAvailable(_StubBackend):
        def __init__(self):
            super().__init__("tesseract", True)

        def ocr_roi(self, image, roi_bbox, config):
            from n2n.ocr.backends.base import OCRResult

            return OCRResult(text="", avg_conf=0.0, words=[], engine="tesseract", elapsed_ms=0.1)

    monkeypatch.setattr(registry, "_FACTORIES", {"tesseract": lambda: AlwaysAvailable()})
    results, attempts = registry.run_ocr_backends(np.zeros((10, 10, 3), dtype=np.uint8), (0, 0, 5, 5), registry.OCRConfig(), "auto")
    assert results
    assert attempts[0]["engine"] == "tesseract"
