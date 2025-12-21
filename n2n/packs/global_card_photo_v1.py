from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2

from n2n.models import DecisionReason, DetectionResult, TextSpan
from n2n.packs.photo_common import (
    ArtifactPaths,
    build_report,
    load_page_contexts,
    render_pdf_to_image,
    spans_to_payload,
    write_report,
)
from n2n.packs.policy import Decision, PolicyConfig
from n2n.ocr.backends.base import OCRConfig, OCRResult
from n2n.ocr.registry import resolve_backend_mode, run_ocr_backends
from n2n.primitives.card_expiry import build_detection as build_expiry_detection
from n2n.primitives.card_expiry import parse_expiry_from_text
from n2n.primitives.card_pan import find_pan_candidates_from_roi_text
from n2n.render.pdf_render import RenderBox, render_highlight_from_boxes, render_redact_from_boxes
from n2n.vision.pan_visual_heuristic import detect_visual_pan_suspicion
from n2n.vision.preprocess import (
    PreprocessOutput,
    map_normalized_box_to_page,
    map_page_box_to_normalized,
    preprocess_document_region,
)

PACK_ID = "global.card_photo.v1"
MODEL_PATH = Path("n2n_assets/models/card_id_yolo.pt")
REASONS = {
    "PAN_CONFIRMED": "PAN verified via Luhn or strong structure.",
    "PAN_SUSPECT": "PAN-like structure detected but not validated.",
    "PAN_SUSPECT_VISUAL": "Visual PAN-like pattern detected without OCR confirmation.",
    "EXPIRY_ONLY": "Expiry detected without PAN confirmation.",
    "QUALITY_LOW": "Image quality too low for decision.",
    "OCCLUSION": "Potential occlusion or fingers detected.",
    "PAN_REMAINS": "PAN still detected after redaction.",
}
PAN_OCR_CONFIG = OCRConfig(psm=7, lang="eng", whitelist_digits=True)
EXPIRY_OCR_CONFIG = OCRConfig(psm=7, lang="eng", whitelist_digits=True, extra_whitelist="/")


@dataclass
class RoiOcrResult:
    page: int
    label: str
    roi_norm: Tuple[float, float, float, float]
    roi_page: Tuple[float, float, float, float]
    text: str
    stats: Dict[str, float]
    spans: List[TextSpan]
    engine: str
    attempts: List[Dict[str, object]]


@dataclass
class PackState:
    detections: List[DetectionResult] = field(default_factory=list)
    highlight_boxes: List[RenderBox] = field(default_factory=list)
    redact_boxes: List[RenderBox] = field(default_factory=list)
    ocr_spans: List[TextSpan] = field(default_factory=list)
    ocr_records: List[RoiOcrResult] = field(default_factory=list)
    suggested_boxes: List[RenderBox] = field(default_factory=list)
    suggested_payload: List[Dict[str, object]] = field(default_factory=list)


def run_pack(
    input_path: Path,
    outdir: Path,
    *,
    force_band_redact: bool = False,
    policy: PolicyConfig | None = None,
    ocr_backend: str | None = None,
) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactPaths(input_path=input_path, outdir=outdir)
    contexts, input_trace, vision_trace = load_page_contexts(input_path, outdir, MODEL_PATH)
    backend_mode = resolve_backend_mode(ocr_backend)
    policy = policy or PolicyConfig()

    trace: Dict[str, object] = {
        "input": input_trace,
        "vision": vision_trace,
        "preprocess": [],
        "ocr": [],
        "validators": {},
        "ocr_backend_mode": backend_mode,
    }
    state = PackState()
    used_engines: set[str] = set()
    card_candidate_present = False

    for ctx in contexts:
        page_idx = ctx.page.index
        candidate_bbox, candidate_source = _resolve_card_candidate(ctx.boxes, ctx.image)
        if candidate_bbox:
            card_candidate_present = True
        preprocess = preprocess_document_region(ctx.image, candidate_bbox)
        prep_trace = {"page": page_idx, "card_candidate_source": candidate_source}
        prep_trace.update(preprocess.trace)
        trace["preprocess"].append(prep_trace)

        pan_roi = _resolve_pan_roi(preprocess, ctx.boxes)
        expiry_roi = _resolve_expiry_roi(preprocess, ctx.boxes)
        pan_ocr = _run_roi_ocr(preprocess, ctx, pan_roi, "PAN ROI", PAN_OCR_CONFIG, backend_mode, prefer_digits=True)
        expiry_ocr = _run_roi_ocr(preprocess, ctx, expiry_roi, "EXPIRY ROI", EXPIRY_OCR_CONFIG, backend_mode)
        used_engines.update([pan_ocr.engine, expiry_ocr.engine])

        trace["ocr"].append(
            {
                "page": page_idx,
                "pan": _trace_entry(pan_ocr),
                "expiry": _trace_entry(expiry_ocr),
                "attempts": {
                    "pan": pan_ocr.attempts,
                    "expiry": expiry_ocr.attempts,
                },
            }
        )
        state.ocr_spans.extend(pan_ocr.spans)
        state.ocr_spans.extend(expiry_ocr.spans)
        state.ocr_records.extend([pan_ocr, expiry_ocr])

        roi_image = _extract_roi(ctx.image, candidate_bbox)
        visual = detect_visual_pan_suspicion(roi_image)
        if visual:
            bbox, visual_trace = visual
            offset_x = candidate_bbox[0] if candidate_bbox else 0
            offset_y = candidate_bbox[1] if candidate_bbox else 0
            adjusted_bbox = (
                bbox[0] + offset_x,
                bbox[1] + offset_y,
                bbox[2] + offset_x,
                bbox[3] + offset_y,
            )
            visual_trace.setdefault("visual_pan", {})["roi_offset"] = [offset_x, offset_y]
            trace.setdefault("visual_pan", []).append(visual_trace)
            det = DetectionResult(
                field_id="card_pan",
                text="PAN_SUSPECT_VISUAL",
                raw="PAN_SUSPECT_VISUAL",
                bbox=adjusted_bbox,
                page=page_idx,
                source="visual",
                validators=["PAN_SUSPECT_VISUAL"],
                severity="suspicion",
            )
            state.detections.append(det)
            visual_box = RenderBox(
                page=page_idx,
                bbox=adjusted_bbox,
                label="PAN VISUAL",
                color=(0.9, 0.3, 0.1),
                page_scale=ctx.page.scale,
            )
            state.highlight_boxes.append(visual_box)
            state.suggested_boxes.append(visual_box)
            state.suggested_payload.append(
                {"page": page_idx, "bbox": [round(v, 2) for v in adjusted_bbox], "label": "PAN VISUAL"}
            )

        pan_dets = find_pan_candidates_from_roi_text(
            pan_ocr.text,
            pan_ocr.stats,
            pan_ocr.roi_page,
            page=page_idx,
        )
        for det in pan_dets:
            state.detections.append(det)
            state.highlight_boxes.append(
                RenderBox(
                    page=det.page,
                    bbox=det.bbox,
                    label=f"{det.field_id}:{det.severity}",
                    color=(0.0, 0.8, 0.0) if det.severity == "hit" else (0.8, 0.5, 0.0),
                    page_scale=ctx.page.scale,
                )
            )
            if det.severity == "hit":
                state.redact_boxes.append(
                    RenderBox(
                        page=det.page,
                        bbox=det.bbox,
                        label="PAN",
                        color=(0.0, 0.0, 0.0),
                        page_scale=ctx.page.scale,
                    )
                )

        expiry_detection = parse_expiry_from_text(expiry_ocr.text)
        if expiry_detection:
            det = build_expiry_detection("card_expiry", expiry_detection, expiry_ocr.roi_page, page_idx)
            state.detections.append(det)
            state.highlight_boxes.append(
                RenderBox(
                    page=det.page,
                    bbox=det.bbox,
                    label="EXPIRY",
                    color=(0.2, 0.5, 0.9),
                    page_scale=ctx.page.scale,
                )
            )

        # Always highlight ROI regions for transparency
        state.highlight_boxes.append(
            RenderBox(
                page=page_idx,
                bbox=pan_ocr.roi_page,
                label="PAN ROI",
                color=(0.9, 0.2, 0.2),
                page_scale=ctx.page.scale,
            )
        )
        state.highlight_boxes.append(
            RenderBox(
                page=page_idx,
                bbox=expiry_ocr.roi_page,
                label="EXPIRY ROI",
                color=(0.2, 0.6, 0.8),
                page_scale=ctx.page.scale,
            )
        )

    decision, reasons = _decide(state, trace, card_candidate_present, policy)
    allow_suggestions = _allow_suggestions(state, decision)

    artifact_map = {
        "input": str(input_path),
        "highlight_pdf": None,
        "redacted_pdf": None,
        "report_json": None,
        "ocr_text": str(artifacts.ocr_text_path()),
        "ocr_spans": str(artifacts.ocr_spans_path()),
    }
    _write_ocr_artifacts(artifacts, state.ocr_records, state.ocr_spans)

    highlight_path = render_highlight_from_boxes(
        input_path,
        state.highlight_boxes,
        artifacts.highlight_path(),
    )
    artifact_map["highlight_pdf"] = highlight_path

    post_redaction: Dict[str, object] = {"checked": 0, "hits_remaining": 0}
    forced_action = None
    if decision == "CONFIRMED" and state.redact_boxes:
        redacted_path = render_redact_from_boxes(
            input_path,
            state.redact_boxes,
            artifacts.redacted_path(),
        )
        artifact_map["redacted_pdf"] = redacted_path
        dpi = contexts[0].page.render_dpi if contexts else 350
        post_redaction = _verify_redaction(Path(redacted_path), state.redact_boxes, dpi)
        if post_redaction["hits_remaining"] > 0:
            decision = "REVIEW"
            reasons = [DecisionReason(code="PAN_REMAINS", description=REASONS["PAN_REMAINS"])]
            artifact_map["redacted_pdf"] = None
    elif decision == "REVIEW" and force_band_redact and allow_suggestions and state.suggested_boxes:
        redacted_path = render_redact_from_boxes(
            input_path,
            state.suggested_boxes,
            artifacts.redacted_path(),
        )
        artifact_map["redacted_pdf"] = redacted_path
        forced_action = "FORCED_REDACT_REVIEW"
        post_redaction = {"forced": True, "boxes": len(state.suggested_boxes)}
    trace["post_redaction"] = post_redaction

    trace["ocr_backends_used"] = sorted(engine for engine in used_engines if engine)

    report = build_report(
        pack_id=PACK_ID,
        decision=decision,
        reasons=reasons,
        detections=state.detections,
        artifacts=artifact_map,
        trace=trace,
        suggested_redactions=state.suggested_payload if (allow_suggestions and decision == "REVIEW") else [],
        action=forced_action,
    )
    write_report(report, artifacts.report_path())
    return report.to_dict()


def _select_card_box(boxes: Sequence[object]) -> object | None:
    card_boxes = [box for box in boxes if box.label in {"card", "id_card"}]
    if not card_boxes:
        return None
    return max(card_boxes, key=lambda b: b.conf)


def _resolve_card_candidate(boxes: Sequence[object], image) -> Tuple[Tuple[int, int, int, int] | None, str]:
    card_box = _select_card_box(boxes)
    if card_box:
        return tuple(map(int, card_box.as_tuple())), "detector"
    guessed = _guess_card_bbox(image)
    if guessed:
        return guessed, "contour"
    height, width = image.shape[:2]
    margin_w = int(width * 0.05)
    margin_h = int(height * 0.07)
    return (margin_w, margin_h, width - margin_w, height - margin_h), "fallback"


def _guess_card_bbox(image) -> Tuple[int, int, int, int] | None:
    if image is None or image.size == 0:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    h, w = gray.shape[:2]
    area_threshold = 0.2 * h * w
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < area_threshold:
        return None
    x, y, cw, ch = cv2.boundingRect(largest)
    pad_w = int(cw * 0.03)
    pad_h = int(ch * 0.03)
    return (
        max(0, x - pad_w),
        max(0, y - pad_h),
        min(w, x + cw + pad_w),
        min(h, y + ch + pad_h),
    )


def _extract_roi(image, bbox: Tuple[int, int, int, int] | None):
    if bbox is None:
        return image
    x1, y1, x2, y2 = map(int, bbox)
    x1 = max(0, min(x1, image.shape[1] - 1))
    y1 = max(0, min(y1, image.shape[0] - 1))
    x2 = max(x1 + 1, min(x2, image.shape[1]))
    y2 = max(y1 + 1, min(y2, image.shape[0]))
    return image[y1:y2, x1:x2]


def _resolve_pan_roi(preprocess: PreprocessOutput, boxes: Sequence[object]) -> Tuple[float, float, float, float]:
    candidates = [box for box in boxes if box.label == "pan"]
    if candidates:
        best = max(candidates, key=lambda b: b.conf)
        return map_page_box_to_normalized(best.as_tuple(), preprocess)
    height, width = preprocess.image.shape[:2]
    band = int(height * 0.3)
    y1 = max(0, height // 2 - band // 2)
    y2 = min(height, y1 + band)
    return (int(width * 0.08), y1, int(width * 0.92), y2)


def _resolve_expiry_roi(preprocess: PreprocessOutput, boxes: Sequence[object]) -> Tuple[float, float, float, float]:
    candidates = [box for box in boxes if box.label == "expiry"]
    if candidates:
        best = max(candidates, key=lambda b: b.conf)
        return map_page_box_to_normalized(best.as_tuple(), preprocess)
    height, width = preprocess.image.shape[:2]
    return (int(width * 0.55), int(height * 0.6), int(width * 0.95), int(height * 0.92))


def _run_roi_ocr(
    preprocess: PreprocessOutput,
    ctx,
    roi_box: Tuple[float, float, float, float],
    label: str,
    config: OCRConfig,
    backend_mode: str,
    prefer_digits: bool = False,
) -> RoiOcrResult:
    results, attempts = run_ocr_backends(preprocess.image, roi_box, config, backend_mode)
    chosen = _select_best_result(results, prefer_digits=prefer_digits)
    stats = {"avg_conf": round(chosen.avg_conf, 4)}
    bbox_page = map_normalized_box_to_page(roi_box, preprocess)
    converted_spans = _convert_words_to_page(chosen.words, preprocess, ctx.page.index, bbox_page, chosen.text, stats["avg_conf"])
    return RoiOcrResult(
        page=ctx.page.index,
        label=label,
        roi_norm=roi_box,
        roi_page=bbox_page,
        text=chosen.text,
        stats=stats,
        spans=converted_spans,
        engine=chosen.engine,
        attempts=attempts,
    )


def _convert_words_to_page(
    words,
    preprocess: PreprocessOutput,
    page_idx: int,
    fallback_bbox: Tuple[float, float, float, float],
    fallback_text: str,
    fallback_conf: float,
) -> List[TextSpan]:
    converted: List[TextSpan] = []
    for word in words:
        bbox_page = map_normalized_box_to_page(word.bbox, preprocess)
        converted.append(
            TextSpan(
                text=word.text,
                bbox=bbox_page,
                page=page_idx,
                source="roi_ocr",
                ocr_conf=word.confidence,
            )
        )
    if not converted and fallback_text:
        converted.append(
            TextSpan(
                text=fallback_text,
                bbox=fallback_bbox,
                page=page_idx,
                source="roi_ocr",
                ocr_conf=fallback_conf,
            )
        )
    return converted


def _trace_entry(roi: RoiOcrResult) -> Dict[str, object]:
    return {
        "page": roi.page,
        "bbox": [round(v, 2) for v in roi.roi_page],
        "stats": roi.stats,
        "text_preview": _mask_digits(roi.text),
        "engine": roi.engine,
    }

def _mask_digits(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        digits = match.group(0)
        if len(digits) <= 4:
            return "#" * len(digits)
        return "#" * (len(digits) - 4) + digits[-4:]

    return re.sub(r"\d{2,}", repl, text or "")


def _select_best_result(results, prefer_digits: bool) -> OCRResult:
    if not results:
        return OCRResult(text="", avg_conf=0.0, words=[], engine="none", elapsed_ms=0.0)

    def key(res: OCRResult):
        digits = re.sub(r"[^0-9]", "", res.text or "")
        digit_count = len(digits)
        return (
            digit_count if prefer_digits else 0,
            res.avg_conf,
            len(res.text or ""),
        )

    best = results[0]
    best_key = key(best)
    for candidate in results[1:]:
        cand_key = key(candidate)
        if cand_key > best_key:
            best = candidate
            best_key = cand_key
    return best


def _decide(
    state: PackState,
    trace: Dict[str, object],
    card_candidate_present: bool,
    policy: PolicyConfig,
) -> Tuple[str, List[DecisionReason]]:
    pan_hits = [
        det for det in state.detections if det.field_id == "card_pan" and det.severity == "hit" and det.source != "visual"
    ]
    pan_suspects = [
        det
        for det in state.detections
        if det.field_id == "card_pan" and det.severity == "suspicion" and det.source != "visual"
    ]
    visual_pan = [det for det in state.detections if det.field_id == "card_pan" and det.source == "visual"]
    expiry_hits = [det for det in state.detections if det.field_id == "card_expiry"]
    occlusion_flags = [
        entry for entry in trace["preprocess"] if entry.get("quality", {}).get("occlusion_suspected")
    ]
    blur_low = any(entry.get("blur_score", 0.0) < policy.blur_max for entry in trace["preprocess"])
    trace["validators"].update(
        {
            "pan_hits": len(pan_hits),
            "pan_suspicions": len(pan_suspects),
            "expiry_hits": len(expiry_hits),
            "occlusion_flags": len(occlusion_flags),
            "card_candidate_present": card_candidate_present,
        }
    )

    if pan_hits and card_candidate_present:
        return Decision.CONFIRMED.value, [
            DecisionReason(code="PAN_CONFIRMED", description=REASONS["PAN_CONFIRMED"])
        ]

    reasons: List[DecisionReason] = []
    ocr_pan_present = any(det.field_id == "card_pan" and det.source == "roi_ocr" for det in state.detections)
    if visual_pan and not ocr_pan_present:
        reasons.append(DecisionReason(code="PAN_SUSPECT_VISUAL", description=REASONS["PAN_SUSPECT_VISUAL"]))
    if pan_suspects:
        reasons.append(DecisionReason(code="PAN_SUSPECT", description=REASONS["PAN_SUSPECT"]))
    if expiry_hits:
        reasons.append(DecisionReason(code="EXPIRY_ONLY", description=REASONS["EXPIRY_ONLY"]))
    if occlusion_flags:
        reasons.append(DecisionReason(code="OCCLUSION", description=REASONS["OCCLUSION"]))
    if blur_low:
        reasons.append(DecisionReason(code="QUALITY_LOW", description=REASONS["QUALITY_LOW"]))

    if not card_candidate_present and blur_low and not reasons:
        return Decision.REJECTED.value, [
            DecisionReason(code="QUALITY_LOW", description=REASONS["QUALITY_LOW"])
        ]
    if reasons:
        return Decision.REVIEW.value, reasons
    return Decision.REVIEW.value, [
        DecisionReason(code="QUALITY_LOW", description=REASONS["QUALITY_LOW"])
    ]


def _allow_suggestions(state: PackState, decision: str) -> bool:
    if decision != Decision.REVIEW.value:
        return False
    visual_detected = any(det.field_id == "card_pan" and det.source == "visual" for det in state.detections)
    ocr_pan_present = any(det.field_id == "card_pan" and det.source == "roi_ocr" for det in state.detections)
    return visual_detected and not ocr_pan_present


def _verify_redaction(
    redacted_pdf: Path,
    boxes: Sequence[RenderBox],
    dpi: int,
) -> Dict[str, int]:
    hits_remaining = 0
    checked = 0
    page_cache: Dict[int, object] = {}
    for box in boxes:
        if box.page not in page_cache:
            image = render_pdf_to_image(redacted_pdf, box.page, dpi=dpi)
            page_cache[box.page] = image
        image = page_cache[box.page]
        text, stats, _ = ocr_roi(image, box.bbox, mode="pan_digits")
        detections = find_pan_candidates_from_roi_text(text, stats, box.bbox, page=box.page)
        if any(det.severity == "hit" for det in detections):
            hits_remaining += 1
        checked += 1
    return {"checked": checked, "hits_remaining": hits_remaining}


def _write_ocr_artifacts(
    artifacts: ArtifactPaths,
    roi_records: Sequence[RoiOcrResult],
    spans: Sequence[TextSpan],
) -> None:
    text_lines = [record.text for record in roi_records if record.text]
    artifacts.ocr_text_path().write_text("\n".join(text_lines), encoding="utf-8")
    payload = spans_to_payload(spans)
    artifacts.ocr_spans_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["run_pack", "PACK_ID"]
