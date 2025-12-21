from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


BBox = Tuple[float, float, float, float]


def detect_visual_pan_suspicion(image: np.ndarray) -> Optional[Tuple[BBox, Dict[str, object]]]:
    if image is None or image.size == 0:
        return None

    working = image.copy()
    if working.ndim == 2:
        gray = working
    else:
        gray = cv2.cvtColor(working, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges_card = cv2.Canny(blur, 30, 100)
    edges_card = cv2.dilate(edges_card, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges_card, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    h, w = gray.shape[:2]
    image_area = h * w
    largest = max(contours, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(largest)
    card_area = max(cv2.contourArea(largest), float(cw * ch))
    if card_area < 0.4 * image_area:
        return None

    if ch == 0:
        return None
    aspect_ratio = cw / float(ch)
    if not 1.4 <= aspect_ratio <= 1.9:
        return None

    card_crop = working[y : y + ch, x : x + cw]
    pan_y_start = int(ch * 0.30)
    pan_y_end = int(ch * 0.60)
    pan_x_start = int(cw * 0.05)
    pan_x_end = int(cw * 0.95)
    pan_band = card_crop[pan_y_start:pan_y_end, pan_x_start:pan_x_end]
    if pan_band.size == 0:
        return None

    pan_gray = cv2.cvtColor(pan_band, cv2.COLOR_BGR2GRAY) if pan_band.ndim == 3 else pan_band
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(pan_gray)
    edges = cv2.Canny(enhanced, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    digit_like: List[Tuple[int, int, int, int]] = []
    min_h = ch * 0.03
    max_h = ch * 0.10
    min_w = cw * 0.01
    max_w = cw * 0.06
    area_threshold = card_area * 0.0002

    for contour in contours:
        px, py, pw, ph = cv2.boundingRect(contour)
        if pw < min_w or pw > max_w:
            continue
        if ph < min_h or ph > max_h:
            continue
        aspect = pw / float(ph) if ph else 0.0
        if not 0.2 <= aspect <= 1.0:
            continue
        if cv2.contourArea(contour) < area_threshold:
            continue
        digit_like.append((px, py, pw, ph))

    if not (10 <= len(digit_like) <= 32):
        return None

    centers_y = [py + ph / 2.0 for (_, py, _, ph) in digit_like]
    heights = [ph for (_, _, _, ph) in digit_like]
    if not heights:
        return None
    median_y = float(np.median(centers_y))
    avg_height = float(np.mean(heights))
    max_dev = max(abs(cy - median_y) for cy in centers_y)
    if avg_height == 0 or max_dev > 0.25 * avg_height:
        return None

    digit_like.sort(key=lambda item: item[0])
    centers_x = [px + pw / 2.0 for (px, _, pw, _) in digit_like]
    spacings = [centers_x[i + 1] - centers_x[i] for i in range(len(centers_x) - 1)]
    spacings = [s for s in spacings if s > 0]
    if not spacings:
        return None
    median_spacing = float(np.median(spacings))
    std_spacing = float(np.std(spacings))
    if median_spacing <= 0:
        return None
    if (std_spacing / median_spacing) >= 0.6:
        return None

    x_min = min(px for (px, _, _, _) in digit_like)
    y_min = min(py for (_, py, _, _) in digit_like)
    x_max = max(px + pw for (px, _, pw, _) in digit_like)
    y_max = max(py + ph for (_, py, _, ph) in digit_like)

    bbox = (
        float(x + pan_x_start + x_min),
        float(y + pan_y_start + y_min),
        float(x + pan_x_start + x_max),
        float(y + pan_y_start + y_max),
    )

    trace = {
        "visual_pan": {
            "card_aspect_ratio": round(aspect_ratio, 3),
            "pan_band_rel": [round(pan_y_start / ch, 3), round(pan_y_end / ch, 3)],
            "digit_like_count": len(digit_like),
            "spacing_cv": round((std_spacing / median_spacing), 3),
            "ocr_used": False,
        }
    }
    return bbox, trace


__all__ = ["detect_visual_pan_suspicion"]
