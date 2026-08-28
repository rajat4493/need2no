from __future__ import annotations

from fastapi.testclient import TestClient

from n2n.webapp.server import app

client = TestClient(app)


def _upload(path, pack_id="uk.bank_statement.share_with_ai"):
    with open(path, "rb") as f:
        return client.post(
            f"/v1/redact?pack_id={pack_id}",
            files={"file": (path.name, f, "application/pdf")},
        )


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "N2N" in r.text


def test_packs_endpoint_lists_share_with_ai():
    r = client.get("/v1/packs")
    assert r.status_code == 200
    pack_ids = {p["pack_id"] for p in r.json()}
    assert "uk.bank_statement.share_with_ai" in pack_ids


def test_clean_statement_certifies_and_downloads_are_gated(clean_statement_pdf):
    r = _upload(clean_statement_pdf)
    assert r.status_code == 200
    report = r.json()
    assert report["status"] == "PASS_AUTO"
    assert report["download_token"]

    token = report["download_token"]
    out = client.get(f"/v1/download/{token}/output.pdf")
    assert out.status_code == 200
    assert out.headers["content-type"] == "application/pdf"
    assert len(out.content) > 0

    manifest = client.get(f"/v1/download/{token}/manifest.json")
    assert manifest.status_code == 200
    assert manifest.json()["release_status"] == "PASS_AUTO"


def test_findings_never_include_raw_sensitive_text(clean_statement_pdf):
    r = _upload(clean_statement_pdf)
    report = r.json()
    for finding in report["findings"]:
        assert set(finding.keys()) == {"field", "page", "bbox", "tier", "validators", "action"}
        assert "text" not in finding


def test_needs_review_statement_has_no_download_token(statement_with_name_pdf):
    r = _upload(statement_with_name_pdf)
    report = r.json()
    assert report["status"] == "NEEDS_REVIEW"
    assert report["download_token"] is None
    assert report["reasons"]


def test_unknown_download_token_is_404():
    r = client.get("/v1/download/not-a-real-token/output.pdf")
    assert r.status_code == 404
    r2 = client.get("/v1/download/not-a-real-token/manifest.json")
    assert r2.status_code == 404


def test_a_refusal_token_cannot_be_forged_to_download_a_previous_certification(
    clean_statement_pdf, statement_with_name_pdf
):
    """A refused document must never become downloadable by any means —
    confirms there's no code path (e.g. reusing another session's token
    format) that could expose a file for a request that didn't earn one."""
    pass_report = _upload(clean_statement_pdf).json()
    refuse_report = _upload(statement_with_name_pdf).json()
    assert refuse_report["download_token"] is None

    good_token = pass_report["download_token"]
    forged = good_token[:-1] + ("A" if good_token[-1] != "A" else "B")
    r = client.get(f"/v1/download/{forged}/output.pdf")
    assert r.status_code == 404


def test_unsupported_document_returns_no_download(unsupported_scanned_pdf):
    r = _upload(unsupported_scanned_pdf)
    report = r.json()
    assert report["status"] == "UNSUPPORTED"
    assert report["download_token"] is None


def test_unknown_pack_id_is_rejected(clean_statement_pdf):
    r = _upload(clean_statement_pdf, pack_id="not.a.real.pack")
    assert r.status_code == 400


def test_empty_upload_is_rejected(tmp_path):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    r = _upload(empty)
    assert r.status_code == 400
