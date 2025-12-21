from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_global_config(base_dir: Path) -> Dict[str, Any]:
    cfg_path = base_dir / "config" / "n2n.defaults.yaml"
    return _load_yaml(cfg_path)


def load_profile_config(base_dir: Path, country: str, profile: str) -> Dict[str, Any]:
    cfg_path = base_dir / "config" / "profiles" / country / f"{profile}.yaml"
    return _load_yaml(cfg_path)


__all__ = ["load_global_config", "load_profile_config"]
