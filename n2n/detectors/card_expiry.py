from __future__ import annotations

import re

from n2n.detectors.common import compact_line, line_text, spans_for_match, union_bbox
from n2n.detectors.validators import CARD_EXPIRY_LABELS, DASH_CLASS, normalize_card_expiry
from n2n.models import Finding, TextSpan

# MM/YY or MM/YYYY, scanned on the compact line like sort_code/account_number
# so a split-token expiry ("03" "/" "27" as separate spans) isn't missed.
# Digit-boundary lookarounds do the job word-boundaries would in normal
# text, same technique as sort_code. Accepts "/" (conventional) or a
# dash-like separator, same reasoning as sort_code's dash-variant fix.
VALUE_RE = re.compile(rf"(?<!\d)\d{{2}}(?:/|{DASH_CLASS})\d{{2,4}}(?!\d)")


def detect_card_expiry(lines: list[list[TextSpan]]) -> list[Finding]:
    """A card expiry date is only auto-tier when a labelled value is found
    on the same line — MM/YY alone is too easily confused with other
    fractions/dates on a document to flag without a label."""
    findings: list[Finding] = []
    for line in lines:
        lowered = line_text(line).lower()
        if not any(label in lowered for label in CARD_EXPIRY_LABELS):
            continue
        compact, offsets = compact_line(line)
        match = VALUE_RE.search(compact)
        if not match:
            continue
        normalized = normalize_card_expiry(match.group(0))
        if normalized is None:
            continue
        value_spans = spans_for_match(offsets, match.start(), match.end())
        if not value_spans:
            continue
        findings.append(
            Finding(
                field_id="card_expiry",
                text=normalized,
                page=line[0].page,
                bbox=union_bbox(value_spans),
                tier="structural",
                validators=("labelled", "format"),
            )
        )
    return findings
