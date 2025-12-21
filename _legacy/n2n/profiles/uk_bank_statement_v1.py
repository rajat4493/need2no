from __future__ import annotations

from n2n.models import PiiCategory
from n2n.profiles.base import FieldDefinition, Profile, register_profile

PROFILE_ID = "uk.bank_statement.v1"

FIELDS = [
    FieldDefinition(
        id="sort_code",
        primitive="uk_sort_code",
        category=PiiCategory.BANK_IDENTIFIERS,
        options={"context_keywords": ["sort code", "sort-code", "sc"]},
    ),
    FieldDefinition(
        id="account_number",
        primitive="uk_account_number_8d",
        category=PiiCategory.BANK_IDENTIFIERS,
        options={"context_keywords": ["account number", "account no", "a/c no", "acc no"]},
    ),
    FieldDefinition(
        id="iban_gb",
        primitive="iban_gb",
        category=PiiCategory.BANK_IDENTIFIERS,
        options={"context_keywords": ["iban"]},
    ),
    FieldDefinition(
        id="card_pan",
        primitive="card_pan",
        category=PiiCategory.CARD_NUMBERS,
        options={"context_keywords": ["card number", "credit card", "debit card"]},
    ),
]


register_profile(Profile(profile_id=PROFILE_ID, fields=FIELDS))

__all__ = ["PROFILE_ID"]
