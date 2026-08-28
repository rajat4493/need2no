from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import n2n.webapp.server as server_module
from n2n.auth import ApiKeyStore
from n2n.webapp.ratelimit import RateLimiter

client = TestClient(server_module.app)


@pytest.fixture(autouse=True)
def isolated_auth(tmp_path, monkeypatch):
    """Every test gets its own API key store and a fresh, generously-
    limited rate limiter — never the real ~/.n2n key store, and never
    sharing rate-limit state across tests."""
    test_store = ApiKeyStore(path=tmp_path / "api_keys.json")
    monkeypatch.setattr(server_module, "api_key_store", test_store)
    monkeypatch.setattr(server_module, "limiter", RateLimiter(limit=1000, window_seconds=60))
    plaintext, _ = test_store.create("test-key")
    return plaintext


@pytest.fixture
def auth_headers(isolated_auth):
    return {"Authorization": f"Bearer {isolated_auth}"}


def _upload(path, headers, pack_id="uk.bank_statement.share_with_ai"):
    with open(path, "rb") as f:
        return client.post(
            f"/v1/redact?pack_id={pack_id}",
            files={"file": (path.name, f, "application/pdf")},
            headers=headers,
        )


# ---------------------------------------------------------------------------
# Auth: missing/invalid/revoked keys, rate limiting
# ---------------------------------------------------------------------------


def test_v1_endpoints_reject_missing_authorization(clean_statement_pdf):
    r = client.get("/v1/packs")
    assert r.status_code == 401

    r2 = _upload(clean_statement_pdf, headers={})
    assert r2.status_code == 401

    r3 = client.get("/v1/download/whatever/output.pdf")
    assert r3.status_code == 401


def test_v1_endpoints_reject_malformed_authorization_header(auth_headers):
    r = client.get("/v1/packs", headers={"Authorization": "not-bearer-format"})
    assert r.status_code == 401


def test_v1_endpoints_reject_invalid_key():
    r = client.get("/v1/packs", headers={"Authorization": "Bearer n2n_live_totallyfake"})
    assert r.status_code == 401


def test_revoked_key_is_rejected(isolated_auth, tmp_path):
    test_store = ApiKeyStore(path=tmp_path / "api_keys.json")
    records = test_store.list()
    assert len(records) == 1
    test_store.revoke(records[0].id)

    r = client.get("/v1/packs", headers={"Authorization": f"Bearer {isolated_auth}"})
    assert r.status_code == 401


def test_valid_key_is_accepted(auth_headers):
    r = client.get("/v1/packs", headers=auth_headers)
    assert r.status_code == 200


def test_rate_limit_returns_429_with_retry_after(monkeypatch, auth_headers):
    monkeypatch.setattr(server_module, "limiter", RateLimiter(limit=2, window_seconds=60))
    assert client.get("/v1/packs", headers=auth_headers).status_code == 200
    assert client.get("/v1/packs", headers=auth_headers).status_code == 200
    third = client.get("/v1/packs", headers=auth_headers)
    assert third.status_code == 429
    assert "Retry-After" in third.headers


def test_downloads_require_auth_even_with_a_valid_certified_token(
    clean_statement_pdf, auth_headers
):
    report = _upload(clean_statement_pdf, headers=auth_headers).json()
    token = report["download_token"]
    assert token

    r = client.get(f"/v1/download/{token}/output.pdf")  # no auth header
    assert r.status_code == 401

    r2 = client.get(f"/v1/download/{token}/output.pdf", headers=auth_headers)
    assert r2.status_code == 200


# ---------------------------------------------------------------------------
# Original functional coverage, now with auth attached
# ---------------------------------------------------------------------------


def test_index_serves_html():
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "N2N" in r.text


def test_packs_endpoint_lists_share_with_ai(auth_headers):
    r = client.get("/v1/packs", headers=auth_headers)
    assert r.status_code == 200
    pack_ids = {p["pack_id"] for p in r.json()}
    assert "uk.bank_statement.share_with_ai" in pack_ids


def test_clean_statement_certifies_and_downloads_are_gated(clean_statement_pdf, auth_headers):
    r = _upload(clean_statement_pdf, auth_headers)
    assert r.status_code == 200
    report = r.json()
    assert report["status"] == "PASS_AUTO"
    assert report["download_token"]

    token = report["download_token"]
    out = client.get(f"/v1/download/{token}/output.pdf", headers=auth_headers)
    assert out.status_code == 200
    assert out.headers["content-type"] == "application/pdf"
    assert len(out.content) > 0

    manifest = client.get(f"/v1/download/{token}/manifest.json", headers=auth_headers)
    assert manifest.status_code == 200
    assert manifest.json()["release_status"] == "PASS_AUTO"


def test_findings_never_include_raw_sensitive_text(clean_statement_pdf, auth_headers):
    r = _upload(clean_statement_pdf, auth_headers)
    report = r.json()
    for finding in report["findings"]:
        assert set(finding.keys()) == {"field", "page", "bbox", "tier", "validators", "action"}
        assert "text" not in finding


def test_needs_review_statement_has_no_download_token(statement_with_name_pdf, auth_headers):
    r = _upload(statement_with_name_pdf, auth_headers)
    report = r.json()
    assert report["status"] == "NEEDS_REVIEW"
    assert report["download_token"] is None
    assert report["reasons"]


def test_unknown_download_token_is_404(auth_headers):
    r = client.get("/v1/download/not-a-real-token/output.pdf", headers=auth_headers)
    assert r.status_code == 404
    r2 = client.get("/v1/download/not-a-real-token/manifest.json", headers=auth_headers)
    assert r2.status_code == 404


def test_a_refusal_token_cannot_be_forged_to_download_a_previous_certification(
    clean_statement_pdf, statement_with_name_pdf, auth_headers
):
    """A refused document must never become downloadable by any means —
    confirms there's no code path (e.g. reusing another session's token
    format) that could expose a file for a request that didn't earn one."""
    pass_report = _upload(clean_statement_pdf, auth_headers).json()
    refuse_report = _upload(statement_with_name_pdf, auth_headers).json()
    assert refuse_report["download_token"] is None

    good_token = pass_report["download_token"]
    forged = good_token[:-1] + ("A" if good_token[-1] != "A" else "B")
    r = client.get(f"/v1/download/{forged}/output.pdf", headers=auth_headers)
    assert r.status_code == 404


def test_unsupported_document_returns_no_download(unsupported_scanned_pdf, auth_headers):
    r = _upload(unsupported_scanned_pdf, auth_headers)
    report = r.json()
    assert report["status"] == "UNSUPPORTED"
    assert report["download_token"] is None


def test_unknown_pack_id_is_rejected(clean_statement_pdf, auth_headers):
    r = _upload(clean_statement_pdf, auth_headers, pack_id="not.a.real.pack")
    assert r.status_code == 400


def test_empty_upload_is_rejected(tmp_path, auth_headers):
    empty = tmp_path / "empty.pdf"
    empty.write_bytes(b"")
    r = _upload(empty, auth_headers)
    assert r.status_code == 400
