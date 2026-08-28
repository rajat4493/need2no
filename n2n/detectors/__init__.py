from n2n.detectors.account_number import detect_account_numbers
from n2n.detectors.card_expiry import detect_card_expiry
from n2n.detectors.card_number import detect_card_numbers
from n2n.detectors.iban import detect_ibans
from n2n.detectors.name_header import detect_name_header_candidates
from n2n.detectors.sort_code import detect_sort_codes

# Structural detectors take (lines) and return only checksum/label-validated
# findings — these are the only ones eligible for automatic tier.
STRUCTURAL_DETECTORS = (
    detect_sort_codes,
    detect_account_numbers,
    detect_ibans,
    detect_card_numbers,
    detect_card_expiry,
)

__all__ = ["STRUCTURAL_DETECTORS", "detect_name_header_candidates"]
