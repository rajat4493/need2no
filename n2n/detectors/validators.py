"""Checksum/structural validators. A finding only reaches automatic tier
if it passes one of these — free-text pattern matches alone are not enough.
"""

from __future__ import annotations

import re

SORT_CODE_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")
SORT_CODE_LOOSE_RE = re.compile(r"^\d{6}$")
ACCOUNT_NUMBER_RE = re.compile(r"^\d{8}$")
GB_IBAN_RE = re.compile(r"^GB\d{2}[A-Z]{4}\d{14}$")
CARD_NUMBER_DIGITS_RE = re.compile(r"^\d{13,19}$")

ACCOUNT_LABELS = ("account number", "account no", "a/c no", "acc no", "account no.")
SORT_CODE_LABELS = ("sort code", "sortcode")


def normalize_sort_code(raw: str) -> str | None:
    digits = re.sub(r"[\s-]", "", raw)
    if re.fullmatch(r"\d{6}", digits):
        return f"{digits[0:2]}-{digits[2:4]}-{digits[4:6]}"
    return None


def is_valid_account_number(raw: str) -> bool:
    return bool(ACCOUNT_NUMBER_RE.fullmatch(raw))


def iban_mod97_valid(iban: str) -> bool:
    """ISO 7064 mod-97-10 checksum, per IBAN spec."""
    iban = iban.replace(" ", "").upper()
    if not GB_IBAN_RE.fullmatch(iban):
        return False
    rearranged = iban[4:] + iban[:4]
    numeric = "".join(str(int(ch, 36)) for ch in rearranged)
    return int(numeric) % 97 == 1


def luhn_valid(digits: str) -> bool:
    digits = re.sub(r"[\s-]", "", digits)
    if not CARD_NUMBER_DIGITS_RE.fullmatch(digits):
        return False
    total = 0
    reversed_digits = digits[::-1]
    for i, ch in enumerate(reversed_digits):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0
