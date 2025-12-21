from n2n.primitives.card_expiry import parse_expiry_from_text
from n2n.primitives.card_pan import find_pan_candidates_from_roi_text
from n2n.primitives.id_mrz import detect_mrz
from n2n.primitives.id_number import detect_id_number


def test_pan_roi_parser_detects_luhn_hit():
    text = "Card 4111 1111 1111 1111"
    detections = find_pan_candidates_from_roi_text(text, {"avg_conf": 0.9}, (0, 0, 100, 40), page=0)
    assert len(detections) == 1
    det = detections[0]
    assert det.severity == "hit"
    assert "luhn" in det.validators


def test_expiry_parser_parses_mm_yy():
    detection = parse_expiry_from_text("valid thru 12/34")
    assert detection is not None
    assert detection.text == "12/34"
    assert detection.severity == "hit"


def test_mrz_detection_identifies_two_lines():
    mrz_text = "P<GBRSMITH<<JOHN<<<<<<<<<<<<<<<<<<\n1234567890GBR7411250M2001012<<<<<<<<<4"
    detection = detect_mrz(mrz_text)
    assert detection is not None
    assert "MRZ" in detection.text or "<" in detection.text


def test_id_number_detector_requires_min_length():
    assert detect_id_number("AB12345") == "AB12345"
    assert detect_id_number("123") is None
