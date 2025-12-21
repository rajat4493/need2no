from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from n2n.models import DecisionReason, DetectionResult, TextSpan
from n2n.packs.photo_common import (
    ArtifactPaths,
    build_report,
    load_page_contexts,
    render_pdf_to_image,
    spans_to_payload,
    write_report,
)
from n2n.ocr.backends.base import OCRConfig, OCRResult
from n2n.ocr.registry import resolve_backend_mode, run_ocr_backends
from n2n.primitives.id_mrz import build_detection as build_mrz_detection
from n2n.primitives.id_mrz import detect_mrz
from n2n.primitives.id_number import build_detection as build_id_detection
from n2n.primitives.id_number import detect_id_number
from n2n.render.pdf_render import RenderBox, render_highlight_from_boxes, render_redact_from_boxes
from n2n.vision.preprocess import (
    PreprocessOutput,
    map_normalized_box_to_page,
    map_page_box_to_normalized,
    preprocess_document_region,
)

PACK_ID = "global.id_photo.v1"
MODEL_PATH = Path("n2n_assets/models/card_id_yolo.pt")
REDACT_FACE = True
REASONS = {
    "MRZ_CONFIRMED": "MRZ structure detected.",
    "ID_SUSPECT": "ID number detected but not verified.",
    "QUALITY_LOW": "Insufficient quality to verify ID.",
    "OCCLUSION": "Occlusion suspected.",
    "MRZ_REMAINS": "MRZ still readable after redaction.",
}
MRZ_OCR_CONFIG = OCRConfig(psm=6, lang="eng", whitelist_digits=False, extra_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789<")
ID_OCR_CONFIG = OCRConfig(psm=7, lang="eng", whitelist_digits=False, extra_whitelist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass
class RoiOcrRecord:
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
    ocr_records: List[RoiOcrRecord] = field(default_factory=list)


def run_pack(input_path: Path, outdir: Path, *, ocr_backend: str | None = None, **_: object) -> Dict[str, object]:
    outdir.mkdir(parents=True, exist_ok=True)
    artifacts = ArtifactPaths(input_path=input_path, outdir=outdir)
    contexts, input_trace, vision_trace = load_page_contexts(input_path, outdir, MODEL_PATH)
    backend_mode = resolve_backend_mode(ocr_backend)
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

    for ctx in contexts:
        page_idx = ctx.page.index
        id_box = _select_id_box(ctx.boxes)
        preprocess = preprocess_document_region(ctx.image, id_box.as_tuple() if id_box else None)
        prep_trace = {"page": page_idx}
        prep_trace.update(preprocess.trace)
        trace["preprocess"].append(prep_trace)

        mrz_roi = _resolve_mrz_roi(preprocess, ctx.boxes)
        id_roi = _resolve_id_roi(preprocess, ctx.boxes)
        mrz_ocr = _run_roi_ocr(preprocess, ctx, mrz_roi, "MRZ", MRZ_OCR_CONFIG, backend_mode)
        id_ocr = _run_roi_ocr(preprocess, ctx, id_roi, "ID NUMBER", ID_OCR_CONFIG, backend_mode)
        used_engines.update([mrz_ocr.engine, id_ocr.engine])
        trace["ocr"].append(
            {
                "page": page_idx,
                "mrz": _trace_entry(mrz_ocr),
                "id_number": _trace_entry(id_ocr),
                "attempts": {
                    "mrz": mrz_ocr.attempts,
                    "id": id_ocr.attempts,
                },
            }
        )
        state.ocr_spans.extend(mrz_ocr.spans)
        state.ocr_spans.extend(id_ocr.spans)
        state.ocr_records.extend([mrz_ocr, id_ocr])

        mrz_detection = detect_mrz(mrz_ocr.text)
        if mrz_detection:
            det = build_mrz_detection("mrz", mrz_detection, mrz_ocr.roi_page, page_idx)
            state.detections.append(det)
            state.highlight_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=det.bbox,
                    label="MRZ HIT",
                    color=(0.0, 0.7, 0.2),
                    page_scale=ctx.page.scale,
                )
            )
            state.redact_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=det.bbox,
                    label="MRZ",
                    color=(0.0, 0.0, 0.0),
                    page_scale=ctx.page.scale,
                )
            )

        id_candidate = detect_id_number(id_ocr.text)
        if id_candidate:
            det = build_id_detection("id_number", id_candidate, id_ocr.roi_page, page_idx)
            state.detections.append(det)
            state.highlight_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=det.bbox,
                    label="ID NUMBER",
                    color=(0.9, 0.5, 0.1),
                    page_scale=ctx.page.scale,
                )
            )
            state.redact_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=det.bbox,
                    label="ID",
                    color=(0.0, 0.0, 0.0),
                    page_scale=ctx.page.scale,
                )
            )

        dob_boxes = [box for box in ctx.boxes if box.label == "dob"]
        if dob_boxes:
            best = max(dob_boxes, key=lambda b: b.conf)
            dob_bbox = map_page_box_to_normalized(best.as_tuple(), preprocess)
            dob_page = map_normalized_box_to_page(dob_bbox, preprocess)
            state.highlight_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=dob_page,
                    label="DOB",
                    color=(0.9, 0.2, 0.2),
                    page_scale=ctx.page.scale,
                )
            )
            state.redact_boxes.append(
                RenderBox(
                    page=page_idx,
                    bbox=dob_page,
                    label="DOB",
                    color=(0.0, 0.0, 0.0),
                    page_scale=ctx.page.scale,
                )
            )

        if REDACT_FACE:
            face_boxes = [box for box in ctx.boxes if box.label == "face"]
            if face_boxes:
                best = max(face_boxes, key=lambda b: b.conf)
                face_bbox = map_page_box_to_normalized(best.as_tuple(), preprocess)
                face_page = map_normalized_box_to_page(face_bbox, preprocess)
                state.highlight_boxes.append(
                    RenderBox(
                        page=page_idx,
                        bbox=face_page,
                        label="FACE",
                        color=(0.3, 0.3, 0.9),
                        page_scale=ctx.page.scale,
                    )
                )
                state.redact_boxes.append(
                    RenderBox(
                        page=page_idx,
                        bbox=face_page,
                        label="FACE",
                        color=(0.0, 0.0, 0.0),
                        page_scale=ctx.page.scale,
                    )
                )

        state.highlight_boxes.append(
            RenderBox(
                page=page_idx,
                bbox=mrz_ocr.roi_page,
                label="MRZ ROI",
                color=(0.9, 0.2, 0.2),
                page_scale=ctx.page.scale,
            )
        )
        state.highlight_boxes.append(
            RenderBox(
                page=page_idx,
                bbox=id_ocr.roi_page,
                label="ID ROI",
                color=(0.2, 0.6, 0.8),
                page_scale=ctx.page.scale,
            )
        )

    decision, reasons = _decide(state, trace)
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

    post_redaction = {"checked": 0, "mrz_hits_remaining": 0}
    if decision == "CONFIRMED" and state.redact_boxes:
        redacted_path = render_redact_from_boxes(
            input_path,
            state.redact_boxes,
            artifacts.redacted_path(),
        )
        artifact_map["redacted_pdf"] = redacted_path
        dpi = contexts[0].page.render_dpi if contexts else 350
        post_redaction = _verify_redaction(Path(redacted_path), state.redact_boxes, dpi)
        if post_redaction["mrz_hits_remaining"] > 0:
            decision = "REVIEW"
            reasons = [DecisionReason(code="MRZ_REMAINS", description=REASONS["MRZ_REMAINS"])]
            artifact_map["redacted_pdf"] = None
    trace["post_redaction"] = post_redaction

    trace["ocr_backends_used"] = sorted(engine for engine in used_engines if engine)

    report = build_report(
        pack_id=PACK_ID,
        decision=decision,
        reasons=reasons,
        detections=state.detections,
        artifacts=artifact_map,
        trace=trace,
    )
    write_report(report, artifacts.report_path())
    return report.to_dict()


def _select_id_box(boxes: Sequence[object]) -> object | None:
    id_boxes = [box for box in boxes if box.label in {"id_card", "card"}]
    if not id_boxes:
        return None
    return max(id_boxes, key=lambda b: b.conf)


def _resolve_mrz_roi(preprocess: PreprocessOutput, boxes: Sequence[object]) -> Tuple[float, float, float, float]:
    mrz_boxes = [box for box in boxes if box.label == "mrz"]
    if mrz_boxes:
        best = max(mrz_boxes, key=lambda b: b.conf)
        return map_page_box_to_normalized(best.as_tuple(), preprocess)
    height, width = preprocess.image.shape[:2]
    return (int(width * 0.05), int(height * 0.75), int(width * 0.95), height)


def _resolve_id_roi(preprocess: PreprocessOutput, boxes: Sequence[object]) -> Tuple[float, float, float, float]:
    number_boxes = [box for box in boxes if box.label in {"id_number", "id"}]
    if number_boxes:
        best = max(number_boxes, key=lambda b: b.conf)
        return map_page_box_to_normalized(best.as_tuple(), preprocess)
    height, width = preprocess.image.shape[:2]
    return (int(width * 0.2), int(height * 0.2), int(width * 0.8), int(height * 0.55))


def _run_roi_ocr(
    preprocess: PreprocessOutput,
    ctx,
    roi_box: Tuple[float, float, float, float],
    label: str,
    config: OCRConfig,
    backend_mode: str,
) -> RoiOcrRecord:
    results, attempts = run_ocr_backends(preprocess.image, roi_box, config, backend_mode)
    chosen = _select_best_result(results)
    bbox_page = map_normalized_box_to_page(roi_box, preprocess)
    stats = {"avg_conf": round(chosen.avg_conf, 4)}
    spans = _convert_words_to_page(
        chosen.words,
        preprocess,
        ctx.page.index,
        bbox_page,
        chosen.text,
        stats["avg_conf"],
    )
    return RoiOcrRecord(
        page=ctx.page.index,
        label=label,
        roi_norm=roi_box,
        roi_page=bbox_page,
        text=chosen.text,
        stats=stats,
        spans=spans,
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


def _trace_entry(roi: RoiOcrRecord) -> Dict[str, object]:
    return {
        "page": roi.page,
        "bbox": [round(v, 2) for v in roi.roi_page],
        "stats": roi.stats,
        "text_preview": roi.text[:40],
        "engine": roi.engine,
    }


def _select_best_result(results: Sequence[OCRResult]) -> OCRResult:
    if not results:
        return OCRResult(text="", avg_conf=0.0, words=[], engine="none", elapsed_ms=0.0)
    best = results[0]
    best_key = (best.avg_conf, len(best.text or ""))
    for candidate in results[1:]:
        key = (candidate.avg_conf, len(candidate.text or ""))
        if key > best_key:
            best = candidate
            best_key = key
    return best


def _decide(state: PackState, trace: Dict[str, object]) -> Tuple[str, List[DecisionReason]]:
    mrz_hits = [det for det in state.detections if det.field_id == "mrz"]
    id_suspects = [det for det in state.detections if det.field_id == "id_number"]
    occlusion = any(entry.get("quality", {}).get("occlusion_suspected") for entry in trace["preprocess"])
    blur_low = any(entry.get("blur_score", 0.0) < 18.0 for entry in trace["preprocess"])
    trace["validators"].update(
        {
            "mrz_hits": len(mrz_hits),
            "id_suspicions": len(id_suspects),
            "occlusion_flags": sum(1 for entry in trace["preprocess"] if entry.get("quality", {}).get("occlusion_suspected")),
        }
    )
    if mrz_hits:
        return "CONFIRMED", [DecisionReason(code="MRZ_CONFIRMED", description=REASONS["MRZ_CONFIRMED"])]

    reasons: List[DecisionReason] = []
    if id_suspects:
        reasons.append(DecisionReason(code="ID_SUSPECT", description=REASONS["ID_SUSPECT"]))
    if occlusion or blur_low:
        reasons.append(DecisionReason(code="QUALITY_LOW", description=REASONS["QUALITY_LOW"]))
        if occlusion:
            reasons.append(DecisionReason(code="OCCLUSION", description=REASONS["OCCLUSION"]))

    if reasons:
        return "REVIEW", reasons
    return "REJECTED", [DecisionReason(code="QUALITY_LOW", description=REASONS["QUALITY_LOW"])]


def _verify_redaction(
    redacted_pdf: Path,
    boxes: Sequence[RenderBox],
    dpi: int,
) -> Dict[str, int]:
    page_cache: Dict[int, object] = {}
    mrz_hits_remaining = 0
    checked = 0
    for box in boxes:
        if box.label not in {"MRZ", "ID"}:
            continue
        if box.page not in page_cache:
            image = render_pdf_to_image(redacted_pdf, box.page, dpi=dpi)
            page_cache[box.page] = image
        image = page_cache[box.page]
        mode = "mrz" if box.label == "MRZ" else "id_alnum"
        text, stats, _ = ocr_roi(image, box.bbox, mode=mode)
        if box.label == "MRZ":
            mrz_hit = detect_mrz(text)
            if mrz_hit:
                mrz_hits_remaining += 1
        else:  # ID
            id_candidate = detect_id_number(text)
            if id_candidate:
                mrz_hits_remaining += 1
        checked += 1
    return {"checked": checked, "mrz_hits_remaining": mrz_hits_remaining}


def _write_ocr_artifacts(
    artifacts: ArtifactPaths,
    roi_records: Sequence[RoiOcrRecord],
    spans: Sequence[TextSpan],
) -> None:
    text_lines = [record.text for record in roi_records if record.text]
    artifacts.ocr_text_path().write_text("\n".join(text_lines), encoding="utf-8")
    payload = spans_to_payload(spans)
    artifacts.ocr_spans_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")


__all__ = ["run_pack", "PACK_ID"]
