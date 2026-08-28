"""pci.card_data.share_with_ai — second purpose pack.

Card-number detection (Luhn-validated) already existed as a supporting
field for the bank-statement pack; this promotes it to a first-class
target so any document carrying payment-card data (receipts, order
confirmations, chargeback disputes, cardholder forms — not just bank
statements) can be certified on its own. Reuses the exact same pipeline,
detectors, transform, and independent-verification machinery the
bank-statement pack already went through adversarial testing on — no new
document-type handling required, which is the point: depth on one proven
engine before breadth, not a second engine.

Deliberately pure regex + checksum, no ML/vision model — avoids the
AGPL-licensing trap flagged in the original build spec for any future
face/ID-detection feature, and keeps this pack fully offline like the
rest of the product.
"""

from n2n.policy import Pack

SHARE_WITH_AI = Pack(
    pack_id="pci.card_data.share_with_ai",
    version="1.0.0",
    description=(
        "Certify a document as safe to hand to an AI provider or "
        "downstream automated pipeline: payment card numbers (Luhn-"
        "validated, clearly formatted) and their expiry dates are "
        "irreversibly removed; free-text name/address candidates are "
        "never auto-resolved. Does not detect CVV/CVC — those should "
        "never be present on a stored document under PCI DSS, and a "
        "reliable label-free detector for a bare 3-4 digit code doesn't "
        "exist yet; see README known limitations."
    ),
    must_hide=frozenset({"card_number", "card_expiry"}),
    must_preserve=frozenset(),
)

PACKS = {
    SHARE_WITH_AI.pack_id: SHARE_WITH_AI,
}
