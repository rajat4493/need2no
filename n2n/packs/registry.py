from __future__ import annotations

from n2n.packs.pci_card_data import PACKS as PCI_PACKS
from n2n.packs.uk_bank_statement import PACKS as UK_BANK_PACKS
from n2n.policy import Pack

PACKS: dict[str, Pack] = {**UK_BANK_PACKS, **PCI_PACKS}


def list_packs() -> dict[str, Pack]:
    return dict(PACKS)


def get_pack(pack_id: str) -> Pack:
    try:
        return PACKS[pack_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pack: {pack_id}") from exc
