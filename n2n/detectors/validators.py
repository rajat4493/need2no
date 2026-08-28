"""Checksum/structural validators. A finding only reaches automatic tier
if it passes one of these — free-text pattern matches alone are not enough.
"""

from __future__ import annotations

import re

# Dash-like characters a real document's font can substitute for a plain
# hyphen (en/em dash, minus sign, non-breaking hyphen, middle dot as a
# font-fallback glyph) — a sort code printed with any of these must still
# be recognized, not silently missed. Kept last in any character class
# built from it below so it's never misread as a range operator.
_DASH_CHARS = "‐‑‒–—−·-"
DASH_CLASS = "[" + _DASH_CHARS + "]"
# Separator class for multi-group numbers (card numbers, expiry dates):
# a real space or any dash-like character. Kept as its own class since a
# card/expiry separator is conventionally a space, not a dash, but must
# still tolerate dash-like substitutes the same way a sort code does.
SEPARATOR_CLASS = "[ " + _DASH_CHARS + "]"

SORT_CODE_RE = re.compile(r"^\d{2}-\d{2}-\d{2}$")
SORT_CODE_LOOSE_RE = re.compile(r"^\d{6}$")
ACCOUNT_NUMBER_RE = re.compile(r"^\d{8}$")
GB_IBAN_RE = re.compile(r"^GB\d{2}[A-Z]{4}\d{14}$")
CARD_NUMBER_DIGITS_RE = re.compile(r"^\d{13,19}$")

ACCOUNT_LABELS = ("account number", "account no", "a/c no", "acc no", "account no.")
SORT_CODE_LABELS = ("sort code", "sortcode")
CARD_EXPIRY_LABELS = (
    "expiry date",
    "expiration date",
    "expiry",
    "expires",
    "exp date",
    "exp.date",
    "valid thru",
    "valid through",
    "good thru",
)


def normalize_sort_code(raw: str) -> str | None:
    digits = re.sub(r"[\s" + _DASH_CHARS + r"]", "", raw)
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


def normalize_card_expiry(raw: str) -> str | None:
    """MM/YY or MM/YYYY, month-validated. Accepts any dash/space variant as
    the separator. Deliberately does NOT reject a date that's already in
    the past — an expired card's expiry date printed on an old statement
    is still real cardholder data, whether or not the card still works."""
    # "/" placed before _DASH_CHARS, not after: _DASH_CHARS ends with a
    # literal "-" that must stay last in the class, or it reads as a
    # range against whatever follows it (e.g. "·-/" as a codepoint range).
    digits = re.sub(r"[\s/" + _DASH_CHARS + r"]", "", raw)
    if not re.fullmatch(r"\d{4}|\d{6}", digits):
        return None
    month, year = digits[:2], digits[2:]
    if not (1 <= int(month) <= 12):
        return None
    return f"{month}/{year}"


def luhn_valid(digits: str) -> bool:
    digits = re.sub(r"[\s" + _DASH_CHARS + r"]", "", digits)
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
