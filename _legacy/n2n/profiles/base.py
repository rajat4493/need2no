from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from n2n.models import PiiCategory


@dataclass
class FieldDefinition:
    id: str
    primitive: str
    category: PiiCategory
    options: Dict[str, object] = field(default_factory=dict)


@dataclass
class Profile:
    profile_id: str
    fields: List[FieldDefinition]


_PROFILES: Dict[str, Profile] = {}


def register_profile(profile: Profile) -> None:
    _PROFILES[profile.profile_id] = profile


def get_profile(profile_id: str) -> Profile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        raise ValueError(f"Unknown profile: {profile_id}") from exc


def list_profiles() -> Dict[str, Profile]:
    return dict(_PROFILES)


__all__ = ["FieldDefinition", "Profile", "register_profile", "get_profile", "list_profiles"]
