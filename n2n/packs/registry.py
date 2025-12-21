from __future__ import annotations

from pathlib import Path
from typing import Dict, Protocol

from n2n.packs.global_card_photo_v1 import run_pack as run_card_photo
from n2n.packs.global_id_photo_v1 import run_pack as run_id_photo
from n2n.packs.global_pci_lite_v1 import run_pack as run_pci_lite


class PackRunner(Protocol):
    def __call__(self, input_path: Path, outdir: Path, **kwargs) -> dict: ...

_PACKS: Dict[str, PackRunner] = {
    "global.pci_lite.v1": run_pci_lite,
    "global.card_photo.v1": run_card_photo,
    "global.id_photo.v1": run_id_photo,
}


def list_packs() -> Dict[str, PackRunner]:
    return dict(_PACKS)


def run_pack(pack_id: str, input_path: str | Path, outdir: str | Path, **kwargs):
    try:
        runner = _PACKS[pack_id]
    except KeyError as exc:
        raise ValueError(f"Unknown pack: {pack_id}") from exc
    return runner(Path(input_path), Path(outdir), **kwargs)


__all__ = ["list_packs", "run_pack"]
