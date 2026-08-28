from n2n.detectors.validators import (
    iban_mod97_valid,
    luhn_valid,
    normalize_sort_code,
)


def test_valid_gb_iban_passes_mod97():
    assert iban_mod97_valid("GB29NWBK60161331926819") is True


def test_iban_with_corrupted_checksum_fails():
    assert iban_mod97_valid("GB30NWBK60161331926819") is False


def test_non_gb_iban_rejected():
    assert iban_mod97_valid("DE89370400440532013000") is False


def test_luhn_valid_test_card_passes():
    assert luhn_valid("4111 1111 1111 1111") is True


def test_luhn_rejects_bad_checksum():
    assert luhn_valid("4111 1111 1111 1112") is False


def test_sort_code_normalizes_and_rejects_garbage():
    assert normalize_sort_code("12-34-56") == "12-34-56"
    assert normalize_sort_code("123456") == "12-34-56"
    assert normalize_sort_code("1234567") is None
