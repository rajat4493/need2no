from pathlib import Path

from n2n import pipeline
from n2n.manifest import verify_manifest_signature
from n2n.keys import load_or_create_keypair
from cryptography.hazmat.primitives import serialization


def test_clean_statement_reaches_pass_auto_and_writes_certified_output(clean_statement_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    report = pipeline.run(clean_statement_pdf, "uk.bank_statement.share_with_ai", out, manifest_path)

    assert report.status == "PASS_AUTO"
    assert out.exists()
    assert manifest_path.exists()
    assert report.verification is not None
    assert report.verification.residual_matches_found is False

    structural_fields = {f.field_id for f in report.findings if f.tier == "structural"}
    assert structural_fields == {"sort_code", "account_number", "iban", "card_number"}
    assert all(f.action == "removed" for f in report.findings if f.tier == "structural")


def test_statement_with_name_header_forces_needs_review_and_writes_nothing(
    statement_with_name_pdf, tmp_path
):
    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    report = pipeline.run(statement_with_name_pdf, "uk.bank_statement.share_with_ai", out, manifest_path)

    assert report.status == "NEEDS_REVIEW"
    assert not out.exists()
    assert not manifest_path.exists()
    assert any("name_header" in r for r in report.reasons)


def test_scanned_document_is_unsupported_and_writes_nothing(unsupported_scanned_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    report = pipeline.run(unsupported_scanned_pdf, "uk.bank_statement.share_with_ai", out, manifest_path)

    assert report.status == "UNSUPPORTED"
    assert not out.exists()
    assert not manifest_path.exists()


def test_dry_run_never_writes_output(clean_statement_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    report = pipeline.run(
        clean_statement_pdf,
        "uk.bank_statement.share_with_ai",
        out,
        manifest_path,
        dry_run=True,
    )
    assert report.status != "PASS_AUTO"
    assert not out.exists()


def test_redacted_output_removes_sensitive_text_content(clean_statement_pdf, tmp_path):
    import fitz

    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    pipeline.run(clean_statement_pdf, "uk.bank_statement.share_with_ai", out, manifest_path)

    doc = fitz.open(out)
    full_text = "\n".join(page.get_text("text") for page in doc)
    doc.close()

    assert "12-34-56" not in full_text
    assert "12345678" not in full_text
    assert "GB29NWBK60161331926819" not in full_text
    assert "4111 1111 1111 1111" not in full_text.replace("\n", " ")


def test_manifest_is_signed_and_verifiable(clean_statement_pdf, tmp_path):
    out = tmp_path / "out.pdf"
    manifest_path = tmp_path / "out.n2n.json"
    report = pipeline.run(clean_statement_pdf, "uk.bank_statement.share_with_ai", out, manifest_path)

    _, public_key = load_or_create_keypair()
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert verify_manifest_signature(report.manifest, public_pem) is True

    tampered = dict(report.manifest)
    tampered["release_status"] = "NEEDS_REVIEW"
    assert verify_manifest_signature(tampered, public_pem) is False


def test_deterministic_replay_same_status_and_output_hash(clean_statement_pdf, tmp_path):
    out1 = tmp_path / "out1.pdf"
    manifest1 = tmp_path / "out1.n2n.json"
    out2 = tmp_path / "out2.pdf"
    manifest2 = tmp_path / "out2.n2n.json"

    report1 = pipeline.run(clean_statement_pdf, "uk.bank_statement.share_with_ai", out1, manifest1)
    report2 = pipeline.run(clean_statement_pdf, "uk.bank_statement.share_with_ai", out2, manifest2)

    assert report1.status == report2.status == "PASS_AUTO"
    assert report1.manifest["output_hash"] == report2.manifest["output_hash"]
    assert report1.manifest["deterministic_replay_id"] == report2.manifest["deterministic_replay_id"]
    assert out1.read_bytes() == out2.read_bytes()
