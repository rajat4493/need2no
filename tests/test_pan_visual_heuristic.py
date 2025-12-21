import cv2
import numpy as np

from n2n.vision.pan_visual_heuristic import detect_visual_pan_suspicion


def test_visual_pan_detects_synthetic_digits():
    img = np.full((300, 500, 3), 255, dtype=np.uint8)
    cv2.rectangle(img, (40, 40), (460, 260), (200, 200, 200), thickness=-1)
    cv2.rectangle(img, (40, 40), (460, 260), (60, 60, 60), thickness=3)

    card_height = 220
    y_start = int(card_height * 0.30) + 40
    digit_height = int(card_height * 0.06)
    y_top = y_start
    for idx in range(14):
        x_left = 65 + idx * 25
        cv2.rectangle(
            img,
            (x_left, y_top),
            (x_left + 12, y_top + digit_height),
            (20, 20, 20),
            thickness=-1,
        )

    result = detect_visual_pan_suspicion(img)
    assert result is not None
    bbox, trace = result
    assert bbox[2] > bbox[0]
    assert "visual_pan" in trace


def test_visual_pan_not_triggered_on_blank_image():
    img = np.full((200, 200, 3), 255, dtype=np.uint8)
    assert detect_visual_pan_suspicion(img) is None
