from __future__ import annotations

from n2n.models import TextSpan


def compact_line(line: list[TextSpan]) -> tuple[str, list[tuple[int, int, TextSpan]]]:
    """Concatenate a line's word spans with no separator, and record which
    character range of the compact string each span occupies. Lets
    regexes match tokens that a bank statement layout has split across
    multiple whitespace-delimited words (e.g. an IBAN printed in 4-char
    groups)."""
    compact = ""
    offsets: list[tuple[int, int, TextSpan]] = []
    for span in line:
        start = len(compact)
        compact += span.text
        offsets.append((start, len(compact), span))
    return compact, offsets


def union_bbox(spans: list[TextSpan]) -> tuple[float, float, float, float]:
    x0 = min(s.bbox[0] for s in spans)
    y0 = min(s.bbox[1] for s in spans)
    x1 = max(s.bbox[2] for s in spans)
    y1 = max(s.bbox[3] for s in spans)
    return (x0, y0, x1, y1)


def spans_for_match(
    offsets: list[tuple[int, int, TextSpan]], match_start: int, match_end: int
) -> list[TextSpan]:
    return [span for start, end, span in offsets if start < match_end and end > match_start]


def line_text(line: list[TextSpan]) -> str:
    return " ".join(s.text for s in line)


def spans_for_joined_match(line: list[TextSpan], start: int, end: int) -> list[TextSpan]:
    """Map a match range within `line_text(line)` (space-joined) back to
    the spans it overlaps."""
    spans = []
    cursor = 0
    for span in line:
        span_start = cursor
        span_end = cursor + len(span.text)
        if span_start < end and span_end > start:
            spans.append(span)
        cursor = span_end + 1  # +1 for the joining space
    return spans
