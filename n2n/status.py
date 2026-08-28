"""The five release states. No other return state is allowed.

Only PASS_AUTO may ever result in an output file being written — see
n2n/output_gate.py for the structural enforcement of that invariant.
"""

from __future__ import annotations

from enum import Enum


class ReleaseStatus(str, Enum):
    PASS_AUTO = "PASS_AUTO"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    UNSUPPORTED = "UNSUPPORTED"
    FAILED_VERIFY = "FAILED_VERIFY"
    PROCESSING_ERROR = "PROCESSING_ERROR"


RELEASABLE_STATUSES = frozenset({ReleaseStatus.PASS_AUTO})
