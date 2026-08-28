from n2n.detectors import STRUCTURAL_DETECTORS
from n2n.detectors.account_number import detect_account_numbers
from n2n.detectors.name_header import detect_name_header_candidates
from n2n.detectors.sort_code import detect_sort_codes
from n2n.models import TextSpan


def _line(*texts_and_x, page=0, y=100.0):
    spans = []
    x = 0.0
    for text in texts_and_x:
        spans.append(TextSpan(text=text, bbox=(x, y, x + len(text) * 6, y + 12), page=page))
        x += len(text) * 6 + 4
    return spans


def test_bare_8_digit_number_without_label_is_never_flagged():
    line = _line("Reference", "87654321", "Payment", "-25.00")
    findings = detect_account_numbers([line])
    assert findings == []


def test_account_number_with_label_is_flagged():
    line = _line("Account", "number:", "12345678")
    findings = detect_account_numbers([line])
    assert len(findings) == 1
    assert findings[0].tier == "structural"
    assert findings[0].text == "12345678"


def test_sort_code_requires_label():
    unlabelled = _line("12-34-56")
    assert detect_sort_codes([unlabelled]) == []

    labelled = _line("Sort", "code:", "12-34-56")
    findings = detect_sort_codes([labelled])
    assert len(findings) == 1
    assert findings[0].text == "12-34-56"


def test_name_header_candidate_is_always_review_tier():
    line = _line("Jane", "Smith", page=0, y=10.0)
    findings = detect_name_header_candidates([line], page_height=800)
    assert len(findings) == 1
    assert findings[0].tier == "review"


def test_name_header_candidate_outside_page_zero_ignored():
    line = _line("Jane", "Smith", page=1, y=10.0)
    assert detect_name_header_candidates([line]) == []


def test_all_structural_detectors_return_lists():
    line = _line("Sort", "code:", "12-34-56")
    for detector in STRUCTURAL_DETECTORS:
        result = detector([line])
        assert isinstance(result, list)
