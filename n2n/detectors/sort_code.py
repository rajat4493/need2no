from __future__ import annotations

import re

from n2n.detectors.common import line_text, spans_for_joined_match, union_bbox
from n2n.detectors.validators import SORT_CODE_LABELS, normalize_sort_code
from n2n.models import Finding, TextSpan

VALUE_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\b|\b\d{6}\b")


def detect_sort_codes(lines: list[list[TextSpan]]) -> list[Finding]:
    """A sort code is only auto-tier when a labelled value pair is found on
    the same line — a bare 6-digit number is not distinctive enough on its
    own (spec 5.4)."""
    findings: list[Finding] = []
    for line in lines:
        text = line_text(line)
        lowered = text.lower()
        if not any(label in lowered for label in SORT_CODE_LABELS):
            continue
        match = VALUE_RE.search(text)
        if not match:
            continue
        normalized = normalize_sort_code(match.group(0))
        if normalized is None:
            continue
        value_spans = spans_for_joined_match(line, match.start(), match.end())
        if not value_spans:
            continue
        findings.append(
            Finding(
                field_id="sort_code",
                text=normalized,
                page=line[0].page,
                bbox=union_bbox(value_spans),
                tier="structural",
                validators=("labelled", "format"),
            )
        )
    return findings
