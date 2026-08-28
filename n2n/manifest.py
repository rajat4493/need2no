from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from n2n import ENGINE_VERSION
from n2n.keys import public_key_fingerprint
from n2n.models import DocumentInfo, Finding, VerificationResult
from n2n.policy import Pack


def _canonical_json(obj: dict) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_manifest(
    *,
    input_bytes: bytes,
    output_bytes: Optional[bytes],
    pack: Pack,
    document_info: Optional[DocumentInfo],
    findings: list[Finding],
    verification: Optional[VerificationResult],
    status: str,
    reasons: list[str],
    private_key: Ed25519PrivateKey,
) -> dict:
    input_hash = hashlib.sha256(input_bytes).hexdigest()
    output_hash = hashlib.sha256(output_bytes).hexdigest() if output_bytes is not None else None
    replay_id = hashlib.sha256(
        f"{input_hash}:{pack.pack_id}:{pack.version}:{ENGINE_VERSION}".encode()
    ).hexdigest()

    public_key = private_key.public_key()

    manifest = {
        "input_hash": f"sha256:{input_hash}",
        "output_hash": f"sha256:{output_hash}" if output_hash else None,
        "pack_id": pack.pack_id,
        "pack_version": pack.version,
        "engine_version": ENGINE_VERSION,
        "page_count": document_info.page_count if document_info else None,
        "document_classification": document_info.classification if document_info else None,
        "extraction_methods": list(document_info.extraction_methods) if document_info else [],
        "findings": [
            {
                "field": f.field_id,
                "page": f.page,
                "geometry": list(f.bbox),
                "tier": f.tier,
                "action": f.action,
                "validators": list(f.validators),
            }
            for f in findings
        ],
        "redaction_technique": pack.redaction_technique,
        "unsupported_regions": [],
        "removed_object_categories": (
            ["metadata", "form_fields", "annotations", "embedded_files", "incremental_history"]
            if status == "PASS_AUTO"
            else []
        ),
        "verification": (
            {
                "method": verification.method,
                "residual_matches_found": verification.residual_matches_found,
                "pages_verified": verification.pages_verified,
            }
            if verification
            else None
        ),
        "release_status": status,
        "reasons": list(reasons),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "deterministic_replay_id": replay_id,
        "signer_public_key_fingerprint": public_key_fingerprint(public_key),
    }

    signature = private_key.sign(_canonical_json(manifest))
    manifest["signature"] = signature.hex()
    manifest["signature_algorithm"] = "ed25519"
    return manifest


def verify_manifest_signature(manifest: dict, public_key_pem: bytes) -> bool:
    manifest = dict(manifest)
    signature_hex = manifest.pop("signature", None)
    manifest.pop("signature_algorithm", None)
    if not signature_hex:
        return False
    public_key = serialization.load_pem_public_key(public_key_pem)
    try:
        public_key.verify(bytes.fromhex(signature_hex), _canonical_json(manifest))
        return True
    except Exception:  # noqa: BLE001
        return False


def write_manifest_bytes(manifest: dict) -> bytes:
    return json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
