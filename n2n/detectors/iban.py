from __future__ import annotations

import re

from n2n.detectors.common import compact_line, spans_for_match, union_bbox
from n2n.detectors.validators import iban_mod97_valid
from n2n.models import Finding, TextSpan

# Unanchored scan pattern — validators.GB_IBAN_RE is anchored (^...$) for
# standalone validation, so it can't be used with finditer over a line that
# also contains a label like "IBAN:".
SCAN_RE = re.compile(r"GB\d{2}[A-Z]{4}\d{14}")


def detect_ibans(lines: list[list[TextSpan]]) -> list[Finding]:
    """GB IBANs are self-validating via mod-97 — the checksum proof is
    the structural signal, no label required."""
    findings: list[Finding] = []
    for line in lines:
        compact, offsets = compact_line(line)
        for match in SCAN_RE.finditer(compact.upper()):
            candidate = compact[match.start() : match.end()]
            if not iban_mod97_valid(candidate):
                continue
            matched_spans = spans_for_match(offsets, match.start(), match.end())
            if not matched_spans:
                continue
            findings.append(
                Finding(
                    field_id="iban",
                    text=candidate.upper(),
                    page=line[0].page,
                    bbox=union_bbox(matched_spans),
                    tier="structural",
                    validators=("mod97_checksum",),
                )
            )
    return findings
