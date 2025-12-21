from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable, List

import numpy as np

from n2n.vision.models import Box

try:  # pragma: no cover - ultralytics is heavy but imported lazily
    from ultralytics import YOLO  # type: ignore
except Exception:  # pragma: no cover
    YOLO = None  # type: ignore


_MODEL_CACHE: dict[str, object] = {}


def load_yolo_model(weights_path: str | Path) -> tuple[object | None, dict[str, object]]:
    path = Path(weights_path)
    info: dict[str, object] = {"model_used": False, "reason": ""}
    if not path.exists():
        info["reason"] = "weights_missing"
        return None, info
    if YOLO is None:
        info["reason"] = "ultralytics_unavailable"
        return None, info
    key = str(path.resolve())
    model = _MODEL_CACHE.get(key)
    if model is not None:
        info["reason"] = "cached"
        info["model_used"] = True
        return model, info
    model = YOLO(str(path))
    _MODEL_CACHE[key] = model
    info["reason"] = "loaded"
    info["model_used"] = True
    return model, info


def detect_objects(image: np.ndarray, model: object | None, conf_threshold: float = 0.25) -> List[Box]:
    if model is None or image is None:
        return []
    results = model(image, verbose=False, conf=conf_threshold)
    boxes: List[Box] = []
    if not results:
        return boxes
    names = getattr(results[0], "names", {}) or getattr(model, "names", {}) or {}
    for page_idx, result in enumerate(results):
        r_boxes = getattr(result, "boxes", None)
        if r_boxes is None:
            continue
        xyxy = r_boxes.xyxy.cpu().numpy() if hasattr(r_boxes.xyxy, "cpu") else r_boxes.xyxy
        confs = r_boxes.conf.cpu().numpy() if hasattr(r_boxes.conf, "cpu") else r_boxes.conf
        cls_indices = r_boxes.cls.cpu().numpy() if hasattr(r_boxes.cls, "cpu") else r_boxes.cls
        for idx, coords in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, coords)
            cls_idx = int(cls_indices[idx]) if cls_indices is not None else 0
            label = names.get(cls_idx, str(cls_idx))
            conf = float(confs[idx]) if confs is not None else 0.0
            boxes.append(Box(label=label, conf=conf, page=page_idx, x1=x1, y1=y1, x2=x2, y2=y2))
    return boxes


__all__ = ["load_yolo_model", "detect_objects"]
