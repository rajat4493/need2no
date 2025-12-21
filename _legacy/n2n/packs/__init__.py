from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

from n2n.packs.global_pci_lite_v1 import run_pci_lite_pack
from n2n.packs.uk_bank_statement_v1 import run_uk_bank_statement_pack

PackRunner = Callable[[Path, Path], object]

PACKS: Dict[str, PackRunner] = {
    "global.pci_lite.v1": run_pci_lite_pack,
    "uk.bank_statement.v1": run_uk_bank_statement_pack,
}


def list_packs() -> List[str]:
    return sorted(PACKS.keys())


def get_pack(pack_name: str) -> Optional[PackRunner]:
    return PACKS.get(pack_name)


def run_pack(pack_name: str, input_path: Path, config_dir: Path):
    runner = get_pack(pack_name)
    if runner is None:
        available = ", ".join(list_packs())
        raise ValueError(
            f"Unknown pack '{pack_name}'. Available packs: {available or 'none registered'}."
        )
    return runner(input_path, config_dir)


__all__ = ["list_packs", "get_pack", "run_pack"]
