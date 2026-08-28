from __future__ import annotations

import re

from n2n.detectors.common import compact_line, line_text, spans_for_match, union_bbox
from n2n.detectors.validators import SORT_CODE_LABELS, normalize_sort_code
from n2n.models import Finding, TextSpan

# Scans the COMPACT (no-separator) line, not the space-joined one: some
# generators/OCR layers split a sort code's digits across multiple
# word-tokens with real gaps between them ("12-" "34-" "56"), which a
# space-joined match would miss entirely and silently release. Digit
# boundaries (?<!\d)/(?!\d) do the job word-boundaries would in normal
# text, since there are no spaces left to anchor on here.
VALUE_RE = re.compile(r"(?<!\d)\d{2}-\d{2}-\d{2}(?!\d)|(?<!\d)\d{6}(?!\d)")


def detect_sort_codes(lines: list[list[TextSpan]]) -> list[Finding]:
    """A sort code is only auto-tier when a labelled value pair is found on
    the same line — a bare 6-digit number is not distinctive enough on its
    own (spec 5.4)."""
    findings: list[Finding] = []
    for line in lines:
        lowered = line_text(line).lower()
        if not any(label in lowered for label in SORT_CODE_LABELS):
            continue
        compact, offsets = compact_line(line)
        match = VALUE_RE.search(compact)
        if not match:
            continue
        normalized = normalize_sort_code(match.group(0))
        if normalized is None:
            continue
        value_spans = spans_for_match(offsets, match.start(), match.end())
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
