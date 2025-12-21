from api.primitives.card_pan import CardPanConfig, find_card_pans
from n2n.models import TextSpan


def _span(text: str, source: str = "text", ocr_conf: float | None = None, page: int = 0):
    return TextSpan(
        text=text,
        bbox=(10.0, 20.0, 200.0, 40.0),
        page_index=page,
        source=source,
        ocr_confidence=ocr_conf,
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
    spans = [_span("Card 4242 4242 4242 42O2")]
    cfg = CardPanConfig()

    dets = find_card_pans(spans, cfg)

    assert len(dets) == 1
    assert dets[0].raw.endswith("4202")
