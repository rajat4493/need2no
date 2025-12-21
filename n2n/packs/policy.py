from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Decision(str, Enum):
    CONFIRMED = "CONFIRMED"
    REVIEW = "REVIEW"
    REJECTED = "REJECTED"


@dataclass
class PolicyConfig:
    ocr_conf_strict: float = 0.70
    blur_max: float = 20.0


__all__ = ["Decision", "PolicyConfig"]
