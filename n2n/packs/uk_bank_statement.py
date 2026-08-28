"""uk.bank_statement.share_with_ai — Phase 1's first purpose pack.

Built first because N2N controls its own acceptance criteria: unlike
proof_of_address or proof_of_funds, no external institution's rules apply.
"""

from n2n.policy import Pack

SHARE_WITH_AI = Pack(
    pack_id="uk.bank_statement.share_with_ai",
    version="1.0.0",
    description=(
        "Certify a UK bank statement as safe to hand to an AI provider or "
        "downstream automated pipeline: structured account identifiers are "
        "irreversibly removed; free-text name/address candidates are never "
        "auto-resolved."
    ),
    must_hide=frozenset({"sort_code", "account_number", "iban", "card_number"}),
    must_preserve=frozenset(),
)

PACKS = {
    SHARE_WITH_AI.pack_id: SHARE_WITH_AI,
}
