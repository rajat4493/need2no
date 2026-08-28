"""uk.bank_statement.share_with_ai — Phase 1's first purpose pack.

Built first because N2N controls its own acceptance criteria: unlike
proof_of_address or proof_of_funds, no external institution's rules apply.
"""

from n2n.policy import Pack

SHARE_WITH_AI = Pack(
    pack_id="uk.bank_statement.share_with_ai",
    # 1.1.0: added card_expiry to must_hide — a linked debit/credit card's
    # expiry date can appear on a bank statement alongside its number, and
    # was previously left undetected. Version bumped per spec 5.7/5.8: a
    # pack's detection scope changing is a behavior change, not silent.
    version="1.1.0",
    description=(
        "Certify a UK bank statement as safe to hand to an AI provider or "
        "downstream automated pipeline: structured account identifiers are "
        "irreversibly removed; free-text name/address candidates are never "
        "auto-resolved."
    ),
    must_hide=frozenset(
        {"sort_code", "account_number", "iban", "card_number", "card_expiry"}
    ),
    must_preserve=frozenset(),
)

PACKS = {
    SHARE_WITH_AI.pack_id: SHARE_WITH_AI,
}
