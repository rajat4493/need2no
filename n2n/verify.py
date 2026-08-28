"""Independent re-verification of the OUTPUT file.

Deliberately uses pdfplumber, not fitz/PyMuPDF — a different parsing
library on a different code path than n2n/extract.py and n2n/transform.py,
so a bug shared between "what we redacted" and "what we checked" can't
hide a leak (spec 5.3 step 6). This module must never import from
n2n.extract or n2n.transform.
"""

from __future__ import annotations

import re

import pdfplumber

from n2n.models import Finding, VerificationResult


def verify_output(output_bytes: bytes, removed_findings: list[Finding]) -> VerificationResult:
    residual_fields: list[str] = []

    needles = [_normalize(f.text) for f in removed_findings if f.text]

    import io

    with pdfplumber.open(io.BytesIO(output_bytes)) as pdf:
        pages_verified = len(pdf.pages)
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        normalized_text = _normalize(full_text)

        for finding, needle in zip(removed_findings, needles):
            if needle and needle in normalized_text:
                residual_fields.append(finding.field_id)

    return VerificationResult(
        method="independent_reextraction_pdfplumber",
        residual_matches_found=len(residual_fields) > 0,
        residual_fields=sorted(set(residual_fields)),
        pages_verified=pages_verified,
    )


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()
