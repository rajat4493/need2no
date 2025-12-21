from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from n2n.models import DetectionResult, TextSpan
from n2n.primitives.registry import register_primitive

CONFUSABLE_MAP = {
    "O": "0",
    "o": "0",
    "I": "1",
    "i": "1",
    "l": "1",
    "S": "5",
    "s": "5",
    "B": "8",
    "b": "8",
    "Z": "2",
    "z": "2",
}
MASKED_MARKERS = {"*", "•"}
_ALLOWED_PATTERN_CHARS = set("0123456789") | set(CONFUSABLE_MAP.keys())
_ALLOWED_PATTERN_CLASS = "0-9" + "".join(sorted({re.escape(ch) for ch in CONFUSABLE_MAP.keys()}))
PAN_RE = re.compile(rf"(?<![{_ALLOWED_PATTERN_CLASS}])(?:[{_ALLOWED_PATTERN_CLASS}][ \-]*){{13,19}}(?![{_ALLOWED_PATTERN_CLASS}])")


@dataclass
class CardPanConfig:
    ocr_conf_suspicion_threshold: float = 0.75
    allow_confusable_normalization: bool = True
    line_y_tol_px: float = 8.0
    max_x_gap_px: float = 40.0
    max_x_gap_ratio: float = 1.0
    digitish_ratio: float = 0.5
    stitch_window_min: int = 2
    stitch_window_max: int = 6
    allow_symbol_confusables: bool = True
    allow_lowercase_b_to_6: bool = False
    min_token_conf_threshold: float = 0.6


def _luhn(digits: str) -> bool:
    total = 0
    parity = len(digits) % 2
    for idx, ch in enumerate(digits):
        value = int(ch)
        if idx % 2 == parity:
            value *= 2
            if value > 9:
                value -= 9
        total += value
    return total % 10 == 0


def _mask(digits: str) -> str:
    masked = "*" * max(0, len(digits) - 4) + digits[-4:]
    return " ".join(masked[i : i + 4] for i in range(0, len(masked), 4))


def _normalize_candidate(candidate: str, allow_confusable: bool) -> str:
    if any(marker in candidate for marker in MASKED_MARKERS):
        return ""
    cleaned = candidate
    if allow_confusable:
        cleaned = "".join(CONFUSABLE_MAP.get(ch, ch) for ch in cleaned)
    digits = re.sub(r"[^0-9]", "", cleaned)
    return digits


def find_card_pans(
    spans: List[TextSpan],
    cfg: CardPanConfig | None = None,
    trace: Dict[str, object] | None = None,
) -> List[DetectionResult]:
    cfg = cfg or CardPanConfig()
    threshold = max(0.0, min(1.0, cfg.ocr_conf_suspicion_threshold))
    detections: List[DetectionResult] = []
    card_trace = {
        "single_span": {"candidates": 0, "hits": 0, "suspicions": 0},
        "stitched": {
            "windows_evaluated": 0,
            "hits": 0,
            "suspicions": 0,
            "best_window": None,
        },
    }

    for span in spans:
        raw_text = span.text or ""
        chars = []
        length = len(raw_text)
        for idx, ch in enumerate(raw_text):
            if ch.isdigit() or ch in " -":
                chars.append(ch)
                continue
            if ch in CONFUSABLE_MAP:
                prev_is_digit = idx > 0 and raw_text[idx - 1].isdigit()
                next_is_digit = idx + 1 < length and raw_text[idx + 1].isdigit()
                if prev_is_digit or next_is_digit:
                    chars.append(ch)
                    continue
            chars.append(" ")
        sanitized_text = "".join(chars)

        for match in PAN_RE.finditer(sanitized_text):
            start, end = match.span()
            while start < end and sanitized_text[start] not in _ALLOWED_PATTERN_CHARS:
                start += 1
            while end > start and sanitized_text[end - 1] not in _ALLOWED_PATTERN_CHARS:
                end -= 1
            if end <= start:
                continue
            raw_candidate = raw_text[start:end]
            normalized = _normalize_candidate(raw_candidate, cfg.allow_confusable_normalization)
            if not 13 <= len(normalized) <= 19:
                continue
            if not normalized.isdigit():
                continue

            card_trace["single_span"]["candidates"] += 1
            passes_luhn = _luhn(normalized)
            severity = "hit"
            validators = ["luhn"] if passes_luhn else []

            if not passes_luhn:
                if span.source == "ocr" and (span.ocr_conf or 0.0) < threshold:
                    severity = "suspicion"
                    validators = ["regex"]
                    card_trace["single_span"]["suspicions"] += 1
                else:
                    continue
            else:
                card_trace["single_span"]["hits"] += 1

            detections.append(
                DetectionResult(
                    field_id="card_pan",
                    text=_mask(normalized),
                    raw=normalized,
                    bbox=span.bbox,
                    page=span.page,
                    source=span.source,
                    validators=validators,
                    severity=severity,
                )
            )

    detections.extend(_stitch_ocr_spans(spans, cfg, threshold, card_trace["stitched"]))
    if trace is not None:
        trace.update(card_trace)
    return detections


@register_primitive("card_pan")
def card_pan_primitive(spans: List[TextSpan]) -> List[DetectionResult]:
    return find_card_pans(spans, CardPanConfig())


def _stitch_ocr_spans(
    spans: List[TextSpan],
    cfg: CardPanConfig,
    threshold: float,
    trace_entry: Dict[str, object],
) -> List[DetectionResult]:
    ocr_spans = [span for span in spans if (span.source or "").lower() == "ocr"]
    if not ocr_spans:
        return []

    window_min = max(2, cfg.stitch_window_min)
    window_max = max(window_min, cfg.stitch_window_max)
    grouped: Dict[int, List[TextSpan]] = {}
    for span in ocr_spans:
        grouped.setdefault(span.page, []).append(span)

    candidates: Dict[Tuple[int, str], Dict[str, object]] = {}
    trace_entry.setdefault("windows_evaluated", 0)
    trace_entry.setdefault("hits", 0)
    trace_entry.setdefault("suspicions", 0)
    trace_entry.setdefault("best_window", None)

    for page, page_spans in grouped.items():
        lines = _group_lines(page_spans, cfg.line_y_tol_px)
        for line in lines:
            ordered = sorted(line, key=lambda s: s.bbox[0])
            length = len(ordered)
            if length < window_min:
                continue

            max_window = min(window_max, length)
            for size in range(window_min, max_window + 1):
                for start in range(0, length - size + 1):
                    window = ordered[start : start + size]
                    if not all(_is_digitish(span.text or "", cfg) for span in window):
                        continue
                    if not _gaps_within_limits(window, cfg):
                        continue
                    candidate_raw = " ".join(span.text or "" for span in window)
                    digits_primary = _normalize_stitched_candidate(candidate_raw, cfg, allow_b_to_6=False)
                    if not 13 <= len(digits_primary) <= 19:
                        continue
                    trace_entry["windows_evaluated"] += 1
                    avg_conf = _average_conf(window)
                    min_conf = min(span.ocr_conf or 0.0 for span in window)
                    passes_primary = _luhn(digits_primary)
                    digits_used = digits_primary
                    b6_used = False
                    if (
                        not passes_primary
                        and cfg.allow_lowercase_b_to_6
                        and any("b" in ((span.text or "").lower()) for span in window)
                    ):
                        digits_b6 = _normalize_stitched_candidate(candidate_raw, cfg, allow_b_to_6=True)
                        if 13 <= len(digits_b6) <= 19 and _luhn(digits_b6):
                            passes_primary = True
                            digits_used = digits_b6
                            b6_used = True
                    low_conf = avg_conf < threshold or min_conf < cfg.min_token_conf_threshold
                    validators = ["regex", "stitch"]
                    severity = None
                    reason = "luhn_fail_high_conf"
                    if passes_primary:
                        severity = "hit"
                        validators.append("luhn")
                        reason = "luhn_pass_b6" if b6_used else "luhn_pass"
                        if b6_used:
                            validators.append("confusable:b->6")
                    elif low_conf:
                        severity = "suspicion"
                        validators.append("near_pan")
                        reason = "luhn_fail_low_conf"

                    info = _build_best_window_info(
                        candidate_raw, digits_used, avg_conf, min_conf, passes_primary, reason
                    )
                    _update_best_window(trace_entry, info)

                    if not severity:
                        continue

                    bbox = _window_bbox(window)
                    detection = DetectionResult(
                        field_id="card_pan",
                        text=_mask(digits_used),
                        raw=candidate_raw.strip(),
                        bbox=bbox,
                        page=page,
                        source="ocr",
                        validators=validators,
                        severity=severity,
                    )
                    key = (page, digits_used)
                    meta = {
                        "detection": detection,
                        "avg_conf": avg_conf,
                        "window_size": len(window),
                        "x0": bbox[0],
                        "severity": severity,
                        "raw": candidate_raw.strip(),
                        "normalized": digits_used,
                        "min_conf": min_conf,
                    }
                    if severity == "hit":
                        trace_entry["hits"] += 1
                    else:
                        trace_entry["suspicions"] += 1
                    existing = candidates.get(key)
                    if not existing or _is_better_candidate(meta, existing):
                        candidates[key] = meta

    return [entry["detection"] for entry in candidates.values()]


def _group_lines(spans: List[TextSpan], tolerance: float) -> List[List[TextSpan]]:
    sorted_spans = sorted(spans, key=lambda s: (_y_center(s.bbox), s.bbox[0]))
    lines: List[List[TextSpan]] = []
    current: List[TextSpan] = []
    current_center: float | None = None

    for span in sorted_spans:
        center = _y_center(span.bbox)
        if not current:
            current = [span]
            current_center = center
            continue
        if abs(center - (current_center or center)) <= tolerance:
            current.append(span)
            # Update running center
            current_center = (current_center * (len(current) - 1) + center) / len(current)
        else:
            lines.append(current)
            current = [span]
            current_center = center

    if current:
        lines.append(current)
    return lines


def _is_digitish(text: str, cfg: CardPanConfig) -> bool:
    if not text:
        return False
    allowed_letters = set(CONFUSABLE_MAP.keys())
    if cfg.allow_lowercase_b_to_6:
        allowed_letters.add("b")
    relevant_chars = [ch for ch in text if ch.isalnum() or ch == "%"]
    if not relevant_chars:
        return False
    digitish = 0
    for ch in relevant_chars:
        if ch.isdigit():
            digitish += 1
        elif ch in allowed_letters:
            digitish += 1
        elif ch == "%" and cfg.allow_symbol_confusables:
            digitish += 1
    ratio = digitish / len(relevant_chars)
    return ratio >= cfg.digitish_ratio


def _gaps_within_limits(window: Sequence[TextSpan], cfg: CardPanConfig) -> bool:
    for left, right in zip(window, window[1:]):
        left_x2 = left.bbox[2]
        right_x1 = right.bbox[0]
        gap = right_x1 - left_x2
        if gap <= 0:
            continue
        left_h = max(0.0, left.bbox[3] - left.bbox[1])
        right_h = max(0.0, right.bbox[3] - right.bbox[1])
        avg_h = (left_h + right_h) / 2 or 1.0
        if gap > cfg.max_x_gap_px and gap > cfg.max_x_gap_ratio * avg_h:
            return False
    return True


def _normalize_stitched_candidate(candidate_raw: str, cfg: CardPanConfig, allow_b_to_6: bool) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z%]", "", candidate_raw or "")
    mapped_chars: List[str] = []
    for ch in cleaned:
        if ch.isdigit():
            mapped_chars.append(ch)
            continue
        mapped = _map_stitch_char(ch, cfg, allow_b_to_6)
        if mapped.isdigit():
            mapped_chars.append(mapped)
    return "".join(mapped_chars)


def _map_stitch_char(ch: str, cfg: CardPanConfig, allow_b_to_6: bool) -> str:
    if ch in {"O", "o"}:
        return "0"
    if ch in {"I", "i", "l"}:
        return "1"
    if ch in {"S", "s"}:
        return "5"
    if ch == "B":
        return "8"
    if ch == "b":
        if allow_b_to_6 and cfg.allow_lowercase_b_to_6:
            return "6"
        return "8"
    if ch in {"Z", "z"}:
        return "2"
    if ch == "%" and cfg.allow_symbol_confusables:
        return "4"
    return ch


def _average_conf(window: Sequence[TextSpan]) -> float:
    confidences = [span.ocr_conf or 0.0 for span in window]
    return sum(confidences) / len(confidences)


def _window_bbox(window: Sequence[TextSpan]) -> Tuple[float, float, float, float]:
    x0 = min(span.bbox[0] for span in window)
    y0 = min(span.bbox[1] for span in window)
    x1 = max(span.bbox[2] for span in window)
    y1 = max(span.bbox[3] for span in window)
    return (x0, y0, x1, y1)


def _y_center(bbox: Tuple[float, float, float, float]) -> float:
    return (bbox[1] + bbox[3]) / 2.0


def _is_better_candidate(new: Dict[str, object], old: Dict[str, object]) -> bool:
    severity_order = {"hit": 2, "suspicion": 1}
    if severity_order[new["severity"]] != severity_order[old["severity"]]:
        return severity_order[new["severity"]] > severity_order[old["severity"]]
    if new["avg_conf"] != old["avg_conf"]:
        return new["avg_conf"] > old["avg_conf"]
    if new["window_size"] != old["window_size"]:
        return new["window_size"] > old["window_size"]
    return new["x0"] < old["x0"]


def _build_best_window_info(
    raw: str,
    normalized: str,
    avg_conf: float,
    min_conf: float,
    luhn_result: bool,
    reason: str,
) -> Dict[str, object]:
    severity = "hit" if luhn_result else ("suspicion" if "low_conf" in reason else "rejected")
    return {
        "raw": raw.strip(),
        "normalized": normalized,
        "length": len(normalized),
        "luhn_result": luhn_result,
        "avg_conf": avg_conf,
        "min_conf": min_conf,
        "reject_reason": None if luhn_result else reason,
        "severity": severity,
    }


def _update_best_window(trace_entry: Dict[str, object], info: Dict[str, object]) -> None:
    best = trace_entry.get("best_window")
    if not best or info["avg_conf"] > best["avg_conf"]:
        trace_entry["best_window"] = info


def find_pan_candidates_from_roi_text(
    text: str,
    conf_stats: Dict[str, float] | None,
    bbox_union: Tuple[float, float, float, float],
    page: int = 0,
) -> List[DetectionResult]:
    if not text:
        return []
    cleaned = text.strip()
    candidates: List[str] = []
    match = PAN_RE.search(cleaned)
    if match:
        candidates.append(cleaned[match.start() : match.end()])
    else:
        digits = re.sub(r"[^0-9]", "", cleaned)
        if 13 <= len(digits) <= 19:
            candidates.append(digits)
    detections: List[DetectionResult] = []
    for candidate in candidates:
        normalized = _normalize_candidate(candidate, allow_confusable=True)
        if not 13 <= len(normalized) <= 19:
            continue
        passes_luhn = _luhn(normalized)
        validators = ["regex", "roi"]
        severity = "hit" if passes_luhn else "suspicion"
        if passes_luhn:
            validators.append("luhn")
        conf_note = conf_stats.get("avg_conf", 0.0) if conf_stats else 0.0
        if conf_note < 0.5 and passes_luhn:
            severity = "suspicion"
            validators.append("low_conf")
        detections.append(
            DetectionResult(
                field_id="card_pan",
                text=_mask(normalized),
                raw=normalized,
                bbox=bbox_union,
                page=page,
                source="roi_ocr",
                validators=validators,
                severity=severity,
            )
        )
    return detections


__all__ = ["CardPanConfig", "find_card_pans", "find_pan_candidates_from_roi_text"]
