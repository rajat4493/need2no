from __future__ import annotations

import re

from n2n.detectors.common import line_text, union_bbox
from n2n.models import Finding, TextSpan

# Two-to-four Title-Case words, typically in the header region (top ~20% of
# page 1). Deliberately loose: position and typography alone cannot prove
# this is the account holder's name (spec 5.6) — it could be a branch name,
# a product name, or an OCR artifact. It ALWAYS lands in review tier; only a
# layout-specific, individually-validated override may promote it, and no
# such override exists in Phase 1.
NAME_CANDIDATE_RE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3})\b")

HEADER_Y_FRACTION = 0.20

# Common UK bank-statement vocabulary that is routinely Title-Cased in real
# layouts ("Sort Code", "Account Number", "Statement Date", ...). A line
# built entirely from this vocabulary is a label, not a name candidate.
# This is a precision fix, not a promotion to auto-tier: anything that
# clears this stoplist still lands at review tier only (spec 5.6).
STATEMENT_VOCAB = {
    "sort", "code", "account", "number", "statement", "date", "balance",
    "opening", "closing", "available", "transaction", "reference", "payment",
    "deposit", "withdrawal", "bank", "plc", "ltd", "branch", "address",
    "total", "description", "amount", "currency", "iban", "swift", "bic",
    "card", "sortcode", "period", "summary", "page", "of", "and", "the",
}


def _is_statement_vocab_only(phrase: str) -> bool:
    words = re.findall(r"[A-Za-z]+", phrase)
    return bool(words) and all(w.lower() in STATEMENT_VOCAB for w in words)


def detect_name_header_candidates(
    lines: list[list[TextSpan]], page_height: float | None = None
) -> list[Finding]:
    findings: list[Finding] = []
    for line in lines:
        if line[0].page != 0:
            continue
        if page_height is not None:
            y0 = min(s.bbox[1] for s in line)
            if y0 > page_height * HEADER_Y_FRACTION:
                continue
        text = line_text(line)
        if ":" in text or any(ch.isdigit() for ch in text):
            # A "Label: value" pair or a line carrying digits (dates, sort
            # codes, amounts) is not a free-text name candidate.
            continue
        match = NAME_CANDIDATE_RE.search(text)
        if not match:
            continue
        if _is_statement_vocab_only(match.group(0)):
            continue
        findings.append(
            Finding(
                field_id="name_header",
                text=match.group(0),
                page=line[0].page,
                bbox=union_bbox(line),
                tier="review",  # never auto-tier — see module docstring
                validators=("header_position_heuristic",),
                action="flagged",
            )
        )
    return findings
