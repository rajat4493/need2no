from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


Bbox = tuple[float, float, float, float]  # x0, y0, x1, y1


@dataclass(frozen=True)
class TextSpan:
    """A run of text extracted from a page, with its location."""

    text: str
    bbox: Bbox
    page: int
    # PDF content-stream block index this span came from (0 when unknown,
    # e.g. spans built directly in tests). Used to detect two independent
    # text layers occupying the same coordinates — see
    # extract.group_spans_into_lines.
    block: int = 0


@dataclass(frozen=True)
class Finding:
    """A candidate piece of sensitive content detected in the document."""

    field_id: str
    text: str
    page: int
    bbox: Bbox
    tier: str  # "structural" (checksum/label-validated) or "review" (free-text candidate)
    validators: tuple[str, ...] = ()
    action: str = "flagged"  # "removed" | "flagged" | "preserved"


@dataclass(frozen=True)
class DocumentInfo:
    """What preflight + extraction learned about the input document."""

    classification: str
    page_count: int
    extraction_methods: tuple[str, ...]
    has_form_fields: bool
    has_annotations: bool
    has_incremental_history: bool
    has_embedded_files: bool


@dataclass
class VerificationResult:
    method: str
    residual_matches_found: bool
    residual_fields: list[str]
    pages_verified: int


@dataclass
class DecisionReport:
    """The full record of one pipeline run."""

    status: str
    pack_id: str
    pack_version: str
    engine_version: str
    reasons: list[str]
    findings: list[Finding]
    document_info: Optional[DocumentInfo]
    verification: Optional[VerificationResult]
    manifest: Optional[dict]
    output_path: Optional[str] = None
    manifest_path: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "pack_id": self.pack_id,
            "pack_version": self.pack_version,
            "engine_version": self.engine_version,
            "reasons": list(self.reasons),
            "findings": [
                {
                    "field": f.field_id,
                    "page": f.page,
                    "bbox": list(f.bbox),
                    "tier": f.tier,
                    "validators": list(f.validators),
                    "action": f.action,
                }
                for f in self.findings
            ],
            "document_info": (
                {
                    "classification": self.document_info.classification,
                    "page_count": self.document_info.page_count,
                    "extraction_methods": list(self.document_info.extraction_methods),
                    "has_form_fields": self.document_info.has_form_fields,
                    "has_annotations": self.document_info.has_annotations,
                    "has_incremental_history": self.document_info.has_incremental_history,
                    "has_embedded_files": self.document_info.has_embedded_files,
                }
                if self.document_info
                else None
            ),
            "verification": (
                {
                    "method": self.verification.method,
                    "residual_matches_found": self.verification.residual_matches_found,
                    "residual_fields": list(self.verification.residual_fields),
                    "pages_verified": self.verification.pages_verified,
                }
                if self.verification
                else None
            ),
            "manifest": self.manifest,
            "output_path": self.output_path,
            "manifest_path": self.manifest_path,
        }
