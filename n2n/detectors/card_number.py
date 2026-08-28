from __future__ import annotations

import re

from n2n.detectors.common import line_text, spans_for_joined_match, union_bbox
from n2n.detectors.validators import luhn_valid
from n2n.models import Finding, TextSpan

# Clearly-formatted card numbers only: 4-digit groups separated by a space
# or dash, 4 groups (16 digits) — matches spec 5.4's "clearly formatted only".
FORMATTED_CARD_RE = re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b")


def detect_card_numbers(lines: list[list[TextSpan]]) -> list[Finding]:
    findings: list[Finding] = []
    for line in lines:
        text = line_text(line)
        for match in FORMATTED_CARD_RE.finditer(text):
            value = match.group(0)
            if not luhn_valid(value):
                continue
            value_spans = spans_for_joined_match(line, match.start(), match.end())
            if not value_spans:
                continue
            findings.append(
                Finding(
                    field_id="card_number",
                    text=value,
                    page=line[0].page,
                    bbox=union_bbox(value_spans),
                    tier="structural",
                    validators=("luhn_checksum", "format"),
                )
            )
    return findings
