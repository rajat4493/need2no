from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parent
CONFIG_ROOT = PACKAGE_ROOT.parent / "config"


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_defaults(config_root: Path | None = None) -> Dict[str, Any]:
    root = config_root or CONFIG_ROOT
    return _load_yaml(root / "n2n.defaults.yaml")


def load_profile(country: str, profile: str, config_root: Path | None = None) -> Dict[str, Any]:
    root = config_root or CONFIG_ROOT
    profile_path = root / "profiles" / country / f"{profile}.yaml"
    return _load_yaml(profile_path)


@lru_cache(maxsize=1)
def load_active_profile(config_root: Path | None = None) -> Dict[str, Any]:
    defaults = load_defaults(config_root)
    profile_cfg = load_profile(defaults["country_pack"], defaults["profile"], config_root)
    return {"defaults": defaults, "profile": profile_cfg}


__all__ = ["load_defaults", "load_profile", "load_active_profile", "CONFIG_ROOT"]
