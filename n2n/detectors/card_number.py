from __future__ import annotations

import re

from n2n.detectors.common import line_text, spans_for_joined_match, union_bbox
from n2n.detectors.validators import SEPARATOR_CLASS, luhn_valid
from n2n.models import Finding, TextSpan

# Clearly-formatted card numbers only (spec 5.4: "clearly formatted only",
# to avoid false positives on other 13-19 digit numbers): standard 4-4-4-4
# grouping (Visa/Mastercard/Discover/etc., 16 digits) and Amex's 4-6-5
# grouping (15 digits). Uses the space-joined line reconstruction, not the
# compact one — the separator between groups is structurally required by
# this pattern (that's what "clearly formatted" means), so collapsing it
# out would break detection rather than help it; word-tokenization at a
# real gap in the source already reconstructs faithfully via a single
# joining space. Dash-variant separators are still accepted, the same way
# sort_code accepts a font's dash substitute for a plain hyphen.
FORMATTED_CARD_RE = re.compile(
    rf"\d{{4}}{SEPARATOR_CLASS}\d{{4}}{SEPARATOR_CLASS}\d{{4}}{SEPARATOR_CLASS}\d{{4}}"
    rf"|\d{{4}}{SEPARATOR_CLASS}\d{{6}}{SEPARATOR_CLASS}\d{{5}}"
)


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
