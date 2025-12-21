from n2n.models import TextSpan
from n2n.primitives.card_pan import CardPanConfig, find_card_pans


def _span(text: str, source: str = "text", ocr_conf: float | None = None, page: int = 0):
    return TextSpan(
        text=text,
        bbox=(10.0, 20.0, 200.0, 40.0),
        page=page,
        source=source,
        ocr_conf=ocr_conf,
    )


def test_detects_valid_pan():
    spans = [_span("Please charge my card 4242 4242 4242 4242 for this order.")]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    d = dets[0]
    assert d.field_id == "card_pan"
    assert d.severity == "hit"
    assert d.source == "text"
    assert d.text.endswith("4242")
    assert "*" in d.text
    assert "luhn" in [v.lower() for v in d.validators]
    assert d.bbox == spans[0].bbox


def test_invalid_luhn_text_source_is_ignored():
    spans = [_span("Card: 4242 4242 4242 4243")]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert dets == []


def test_masked_last_four_not_detected():
    spans = [_span("Card used: **** **** **** 4242")]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert dets == []


def test_low_confidence_ocr_emits_suspicion():
    spans = [_span("4242 4242 4242 4243", source="ocr", ocr_conf=0.60)]
    cfg = CardPanConfig(ocr_conf_suspicion_threshold=0.75)

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    d = dets[0]
    assert d.field_id == "card_pan"
    assert d.source == "ocr"
    assert d.severity == "suspicion"
    assert "luhn" not in [v.lower() for v in d.validators]
    assert d.bbox == spans[0].bbox


def test_confusable_letters_are_normalized():
    spans = [_span("Card 4O12 8888 8888 1881")]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    assert dets[0].raw == "4012888888881881"


def test_ocr_stitching_detects_pan():
    spans = [
        TextSpan(text="4000", bbox=(0.0, 0.0, 40.0, 20.0), page=0, source="ocr", ocr_conf=0.95),
        TextSpan(text="123%", bbox=(45.0, 0.0, 85.0, 20.0), page=0, source="ocr", ocr_conf=0.60),
        TextSpan(text="Sb78", bbox=(90.0, 0.0, 130.0, 20.0), page=0, source="ocr", ocr_conf=0.65),
        TextSpan(text="9017", bbox=(135.0, 0.0, 175.0, 20.0), page=0, source="ocr", ocr_conf=0.90),
    ]
    cfg = CardPanConfig(allow_lowercase_b_to_6=True)

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    det = dets[0]
    assert det.severity == "hit"
    assert "stitch" in det.validators
    assert "confusable:b->6" in det.validators
    assert det.source == "ocr"
    assert det.raw.replace(" ", "").startswith("4000")
    assert det.bbox == (0.0, 0.0, 175.0, 20.0)


def test_ocr_stitching_low_conf_triggers_suspicion():
    spans = [
        TextSpan(text="6000", bbox=(0.0, 0.0, 40.0, 20.0), page=0, source="ocr", ocr_conf=0.55),
        TextSpan(text="1234", bbox=(45.0, 0.0, 85.0, 20.0), page=0, source="ocr", ocr_conf=0.50),
        TextSpan(text="5678", bbox=(90.0, 0.0, 130.0, 20.0), page=0, source="ocr", ocr_conf=0.45),
        TextSpan(text="9010", bbox=(135.0, 0.0, 175.0, 20.0), page=0, source="ocr", ocr_conf=0.40),
    ]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    det = dets[0]
    assert det.severity == "suspicion"
    assert "near_pan" in det.validators
    assert det.source == "ocr"
