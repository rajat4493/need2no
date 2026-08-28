"""Orchestrates the eight pipeline stages (spec 5.3) and is the ONLY module
permitted to mint a release token — see n2n/output_gate.py.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from n2n import ENGINE_VERSION
from n2n.detectors import STRUCTURAL_DETECTORS, detect_name_header_candidates
from n2n.extract import extract_native, group_spans_into_lines
from n2n.keys import load_or_create_keypair
from n2n.manifest import build_manifest, write_manifest_bytes
from n2n.models import DecisionReport, DocumentInfo, Finding, VerificationResult
from n2n.output_gate import mint_release_token, write_certified_output
from n2n.packs.registry import get_pack
from n2n.policy import Pack, resolve
from n2n.preflight import classify
from n2n.status import ReleaseStatus
from n2n.transform import redact_document
from n2n.verify import verify_output


def run(
    input_path: Path,
    pack_id: str,
    output_path: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    dry_run: bool = False,
) -> DecisionReport:
    input_path = Path(input_path)
    pack = get_pack(pack_id)
    input_bytes = input_path.read_bytes()

    # Stage 1: preflight
    preflight = classify(input_path)
    if not preflight.supported:
        return _finish(
            status=ReleaseStatus.UNSUPPORTED,
            pack=pack,
            reasons=[preflight.reason],
            findings=[],
            document_info=None,
            verification=None,
            input_bytes=input_bytes,
            output_bytes=None,
        )

    try:
        # Stage 2: extraction
        extraction = extract_native(input_path)

        # Stage 3: detection
        lines = group_spans_into_lines(extraction.spans)
        findings: list[Finding] = []
        for detector in STRUCTURAL_DETECTORS:
            findings.extend(detector(lines))
        page_zero_height = extraction.page_heights[0] if extraction.page_heights else None
        findings.extend(detect_name_header_candidates(lines, page_height=page_zero_height))

        # Stage 4: policy resolution
        resolution = resolve(findings, pack)

        if resolution.conflicts:
            return _finish(
                status=ReleaseStatus.NEEDS_REVIEW,
                pack=pack,
                reasons=[f"Policy conflict on field(s): {', '.join(resolution.conflicts)}"],
                findings=findings,
                document_info=extraction.document_info,
                verification=None,
                input_bytes=input_bytes,
                output_bytes=None,
            )

        if resolution.needs_review:
            review_fields = sorted({f.field_id for f in resolution.needs_review})
            return _finish(
                status=ReleaseStatus.NEEDS_REVIEW,
                pack=pack,
                reasons=[
                    f"Field(s) require human review before release: {', '.join(review_fields)}"
                ],
                findings=findings,
                document_info=extraction.document_info,
                verification=None,
                input_bytes=input_bytes,
                output_bytes=None,
            )

        if extraction.document_info.has_incremental_history:
            # We flatten history during transform, but a source document
            # carrying prior incremental updates is a signal worth surfacing
            # rather than silently absorbing.
            pass  # handled by transform; not a refusal condition on its own

        if dry_run:
            return _finish(
                status=ReleaseStatus.NEEDS_REVIEW,
                pack=pack,
                reasons=["Dry run: no output produced. Findings above show what would happen."],
                findings=findings,
                document_info=extraction.document_info,
                verification=None,
                input_bytes=input_bytes,
                output_bytes=None,
            )

        # Stage 5: transform (irreversible removal)
        redacted_bytes = redact_document(input_path, resolution.to_remove)

        # Stage 6: independent verification (separate code path)
        verification = verify_output(redacted_bytes, resolution.to_remove)

        if verification.residual_matches_found:
            return _finish(
                status=ReleaseStatus.FAILED_VERIFY,
                pack=pack,
                reasons=[
                    "Independent re-verification found residual matches for: "
                    + ", ".join(verification.residual_fields)
                ],
                findings=findings,
                document_info=extraction.document_info,
                verification=verification,
                input_bytes=input_bytes,
                output_bytes=None,
            )

        # Stage 7 + 8: manifest + release decision
        resolved_findings = resolution.to_remove + resolution.to_preserve
        report = _finish(
            status=ReleaseStatus.PASS_AUTO,
            pack=pack,
            reasons=["All mandatory controls for this pack passed."],
            findings=resolved_findings,
            document_info=extraction.document_info,
            verification=verification,
            input_bytes=input_bytes,
            output_bytes=redacted_bytes,
        )

        if output_path is not None and manifest_path is not None:
            manifest_bytes = write_manifest_bytes(report.manifest)
            token = mint_release_token(redacted_bytes)
            write_certified_output(
                token=token,
                output_payload=redacted_bytes,
                output_path=output_path,
                manifest_payload=manifest_bytes,
                manifest_path=manifest_path,
            )
            report.output_path = str(output_path)
            report.manifest_path = str(manifest_path)

        return report

    except Exception as exc:  # noqa: BLE001 - any unhandled failure is PROCESSING_ERROR, never a partial release
        return _finish(
            status=ReleaseStatus.PROCESSING_ERROR,
            pack=pack,
            reasons=[f"Pipeline did not complete: {exc}"],
            findings=[],
            document_info=None,
            verification=None,
            input_bytes=input_bytes,
            output_bytes=None,
        )


def _finish(
    *,
    status: ReleaseStatus,
    pack: Pack,
    reasons: list[str],
    findings: list[Finding],
    document_info: Optional[DocumentInfo],
    verification: Optional[VerificationResult],
    input_bytes: bytes,
    output_bytes: Optional[bytes],
) -> DecisionReport:
    private_key, _ = load_or_create_keypair()
    manifest = build_manifest(
        input_bytes=input_bytes,
        output_bytes=output_bytes,
        pack=pack,
        document_info=document_info,
        findings=findings,
        verification=verification,
        status=status.value,
        reasons=reasons,
        private_key=private_key,
    )
    return DecisionReport(
        status=status.value,
        pack_id=pack.pack_id,
        pack_version=pack.version,
        engine_version=ENGINE_VERSION,
        reasons=reasons,
        findings=findings,
        document_info=document_info,
        verification=verification,
        manifest=manifest,
    )
