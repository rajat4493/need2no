from __future__ import annotations

import re

from n2n.detectors.common import line_text, spans_for_joined_match, union_bbox
from n2n.detectors.validators import ACCOUNT_LABELS, is_valid_account_number
from n2n.models import Finding, TextSpan

VALUE_RE = re.compile(r"\b\d{8}\b")


def detect_account_numbers(lines: list[list[TextSpan]]) -> list[Finding]:
    """An 8-digit number is only auto-tier when it appears next to an
    account-number label. A bare 8-digit number elsewhere (e.g. a
    transaction reference) is never flagged — spec 5.4."""
    findings: list[Finding] = []
    for line in lines:
        text = line_text(line)
        lowered = text.lower()
        if not any(label in lowered for label in ACCOUNT_LABELS):
            continue
        for match in VALUE_RE.finditer(text):
            value = match.group(0)
            if not is_valid_account_number(value):
                continue
            value_spans = spans_for_joined_match(line, match.start(), match.end())
            if not value_spans:
                continue
            findings.append(
                Finding(
                    field_id="account_number",
                    text=value,
                    page=line[0].page,
                    bbox=union_bbox(value_spans),
                    tier="structural",
                    validators=("labelled", "format"),
                )
            )
    return findings
