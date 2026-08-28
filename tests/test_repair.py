"""Coverage for the pikepdf-based repair pass (n2n/repair.py) and its
orchestration in n2n/pipeline.py.

MuPDF's own repair-on-open is already strong enough that constructing a
"naturally" corrupted PDF fitz can't open but pikepdf can turns out to be
hard to hit reliably (most truncation/corruption we could throw at it
recovers via fitz alone). The orchestration logic — try repair when
preflight first fails, use the repaired bytes for the rest of the
pipeline if it helps, record that transparently, clean up the temp file
either way — is tested directly instead, by controlling what
attempt_repair returns rather than depending on finding exactly the
right kind of broken input.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pymupdf as fitz

from n2n import pipeline
from n2n.repair import attempt_repair

PACK_ID = "uk.bank_statement.share_with_ai"


def test_attempt_repair_returns_none_for_unrecoverable_garbage(tmp_path):
    path = tmp_path / "garbage.pdf"
    path.write_bytes(b"not a pdf at all, just garbage bytes")
    assert attempt_repair(path) is None


def test_attempt_repair_returns_bytes_for_a_valid_pdf(tmp_path):
    path = tmp_path / "valid.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(path)
    doc.close()
    repaired = attempt_repair(path)
    assert repaired is not None
    assert repaired.startswith(b"%PDF")


def _make_valid_statement_bytes() -> bytes:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((60, 60), "Sort code: 12-34-56", fontsize=10)
    page.insert_text((60, 90), "Account number: 12345678", fontsize=10)
    data = doc.tobytes()
    doc.close()
    return data


def test_pipeline_uses_repaired_bytes_when_repair_rescues_an_unsupported_input(tmp_path):
    """Preflight's first attempt is forced to see garbage (so it reports
    UNSUPPORTED), but attempt_repair is mocked to return a genuinely valid
    document — proving the pipeline actually switches to and processes the
    repaired bytes, rather than just reporting repair happened."""
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"garbage, not a real pdf")

    with patch("n2n.pipeline.attempt_repair", return_value=_make_valid_statement_bytes()):
        out = tmp_path / "out.pdf"
        manifest = tmp_path / "out.n2n.json"
        report = pipeline.run(input_path, PACK_ID, out, manifest)

    assert report.status == "PASS_AUTO"
    assert out.exists()
    assert "pikepdf_repair_applied" in report.manifest["extraction_methods"]
    structural = {f.field_id for f in report.findings if f.tier == "structural"}
    assert structural == {"sort_code", "account_number"}


def test_pipeline_falls_back_to_unsupported_when_repair_cannot_help(tmp_path):
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"garbage, not a real pdf")

    with patch("n2n.pipeline.attempt_repair", return_value=None):
        out = tmp_path / "out.pdf"
        manifest = tmp_path / "out.n2n.json"
        report = pipeline.run(input_path, PACK_ID, out, manifest)

    assert report.status == "UNSUPPORTED"
    assert not out.exists()


def test_pipeline_ignores_repair_bytes_that_still_dont_classify_as_supported(tmp_path):
    """attempt_repair succeeding doesn't automatically mean the result is
    usable — e.g. pikepdf might "repair" a file into something that opens
    but still has no extractable text. The pipeline must re-run full
    preflight classification on the repaired bytes, not just trust that
    repair succeeding means the document is now fine."""
    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"garbage, not a real pdf")

    empty_doc = fitz.open()
    empty_doc.new_page()  # a page with no text and no images
    still_empty_bytes = empty_doc.tobytes()
    empty_doc.close()

    with patch("n2n.pipeline.attempt_repair", return_value=still_empty_bytes):
        out = tmp_path / "out.pdf"
        manifest = tmp_path / "out.n2n.json"
        report = pipeline.run(input_path, PACK_ID, out, manifest)

    assert report.status == "UNSUPPORTED"
    assert not out.exists()


def test_no_leftover_temp_file_after_a_repaired_run(tmp_path):
    import glob
    import tempfile as tempfile_module

    before = set(glob.glob(str(Path(tempfile_module.gettempdir()) / "*.pdf")))

    input_path = tmp_path / "input.pdf"
    input_path.write_bytes(b"garbage, not a real pdf")
    with patch("n2n.pipeline.attempt_repair", return_value=_make_valid_statement_bytes()):
        pipeline.run(input_path, PACK_ID, tmp_path / "out.pdf", tmp_path / "out.json")

    after = set(glob.glob(str(Path(tempfile_module.gettempdir()) / "*.pdf")))
    assert after - before == set(), f"leftover temp files: {after - before}"
