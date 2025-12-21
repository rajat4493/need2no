from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np


@dataclass
class PreprocessOutput:
    image: np.ndarray
    roi_bbox: Tuple[int, int, int, int]
    forward_matrix: np.ndarray | None
    inverse_matrix: np.ndarray | None
    used_warp: bool
    trace: Dict[str, object]


def preprocess_document_region(
    image: np.ndarray,
    bbox: Tuple[float, float, float, float] | None = None,
    padding_ratio: float = 0.08,
) -> PreprocessOutput:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (0, 0, width, height)
    if bbox is not None:
        x1, y1, x2, y2 = _clamp_box(bbox, width, height)
        pad_w = int((x2 - x1) * padding_ratio)
        pad_h = int((y2 - y1) * padding_ratio)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)
    roi = image[y1:y2, x1:x2].copy()
    forward_matrix = None
    inverse_matrix = None
    used_warp = False

    warped, forward_matrix, inverse_matrix, used_warp = _warp_roi(roi)
    processed, trace = _normalize_image(warped)
    quality = _quality_metrics(roi if not used_warp else warped)
    trace.update(
        {
            "roi_bbox": (x1, y1, x2, y2),
            "used_warp": used_warp,
            "quality": quality,
        }
    )

    return PreprocessOutput(
        image=processed,
        roi_bbox=(x1, y1, x2, y2),
        forward_matrix=forward_matrix,
        inverse_matrix=inverse_matrix,
        used_warp=used_warp,
        trace=trace,
    )


def map_page_box_to_normalized(box: Tuple[float, float, float, float], prep: PreprocessOutput) -> Tuple[float, float, float, float]:
    points = _bbox_to_points(box)
    offset = np.array([prep.roi_bbox[0], prep.roi_bbox[1]], dtype=np.float32)
    local = points - offset
    if prep.forward_matrix is not None:
        mapped = cv2.perspectiveTransform(local.reshape(-1, 1, 2), prep.forward_matrix).reshape(-1, 2)
    else:
        mapped = local
    return _points_to_bbox(mapped)


def map_normalized_box_to_page(box: Tuple[float, float, float, float], prep: PreprocessOutput) -> Tuple[float, float, float, float]:
    points = _bbox_to_points(box)
    if prep.inverse_matrix is not None:
        mapped = cv2.perspectiveTransform(points.reshape(-1, 1, 2), prep.inverse_matrix).reshape(-1, 2)
    else:
        mapped = points
    mapped += np.array([prep.roi_bbox[0], prep.roi_bbox[1]], dtype=np.float32)
    return _points_to_bbox(mapped)


def _warp_roi(roi: np.ndarray):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 40, 120)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    forward_matrix = None
    inverse_matrix = None
    used_warp = False
    if not contours:
        return roi, forward_matrix, inverse_matrix, used_warp

    for contour in contours[:5]:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        pts = approx.reshape(4, 2).astype(np.float32)
        ordered = _order_points(pts)
        (tl, tr, br, bl) = ordered
        width_top = np.linalg.norm(tr - tl)
        width_bottom = np.linalg.norm(br - bl)
        height_left = np.linalg.norm(bl - tl)
        height_right = np.linalg.norm(br - tr)
        max_width = int(max(width_top, width_bottom))
        max_height = int(max(height_left, height_right))
        if max_width < 10 or max_height < 10:
            continue
        dst = np.array(
            [[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]],
            dtype=np.float32,
        )
        forward_matrix = cv2.getPerspectiveTransform(ordered, dst)
        inverse_matrix = cv2.getPerspectiveTransform(dst, ordered)
        warped = cv2.warpPerspective(roi, forward_matrix, (max_width, max_height))
        used_warp = True
        return warped, forward_matrix, inverse_matrix, used_warp

    return roi, forward_matrix, inverse_matrix, used_warp


def _normalize_image(warped: np.ndarray) -> Tuple[np.ndarray, Dict[str, object]]:
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) if warped.ndim == 3 else warped
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, h=3, templateWindowSize=7, searchWindowSize=21)

    method = "adaptive_gaussian"
    try:
        thresh = cv2.adaptiveThreshold(
            denoised,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
    except Exception:
        method = "otsu"
        _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    final = cv2.bitwise_and(denoised, thresh)

    blur_score = float(cv2.Laplacian(denoised, cv2.CV_64F).var())
    skew_angle = _estimate_skew(denoised)

    trace = {
        "blur_score": blur_score,
        "skew_angle": skew_angle,
        "used_threshold": True,
        "threshold_method": method,
    }
    return final, trace


def _estimate_skew(image: np.ndarray) -> float:
    edges = cv2.Canny(image, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180.0, threshold=60, minLineLength=image.shape[1] * 0.5, maxLineGap=20)
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:20]:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -45 <= angle <= 45:
            angles.append(angle)
    if not angles:
        return 0.0
    return float(np.median(angles))


def _quality_metrics(image: np.ndarray) -> Dict[str, object]:
    if image.ndim == 2:
        color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        color = image
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    lower1 = np.array([0, 30, 60], dtype=np.uint8)
    upper1 = np.array([25, 160, 255], dtype=np.uint8)
    lower2 = np.array([160, 30, 60], dtype=np.uint8)
    upper2 = np.array([180, 160, 255], dtype=np.uint8)
    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    skin_mask = cv2.bitwise_or(mask1, mask2)
    skin_ratio = float(cv2.countNonZero(skin_mask)) / float(skin_mask.size or 1)

    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    dark_ratio = float(np.sum(gray < 25)) / float(gray.size or 1)
    bright_ratio = float(np.sum(gray > 230)) / float(gray.size or 1)
    occlusion_ratio = max(skin_ratio, dark_ratio)

    return {
        "skin_ratio": round(skin_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
        "bright_ratio": round(bright_ratio, 4),
        "occlusion_ratio": round(occlusion_ratio, 4),
        "occlusion_suspected": occlusion_ratio > 0.12,
    }


def _order_points(pts: np.ndarray) -> np.ndarray:
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def _bbox_to_points(box: Tuple[float, float, float, float]) -> np.ndarray:
    x1, y1, x2, y2 = box
    return np.array(
        [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        dtype=np.float32,
    )


def _points_to_bbox(points: np.ndarray) -> Tuple[float, float, float, float]:
    xs = points[:, 0]
    ys = points[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _clamp_box(box: Tuple[float, float, float, float], width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), width - 1))
    y1 = max(0, min(int(y1), height - 1))
    x2 = max(x1 + 1, min(int(x2), width))
    y2 = max(y1 + 1, min(int(y2), height))
    return x1, y1, x2, y2


__all__ = [
    "PreprocessOutput",
    "preprocess_document_region",
    "map_page_box_to_normalized",
    "map_normalized_box_to_page",
]
