"""Best-effort repair pass for malformed-but-recoverable PDFs.

Uses pikepdf (built on QPDF) — far more battle-tested than anything
hand-rolled here for xref-table damage, dangling object references, and
similar structural corruption a truncated download or a buggy upstream
generator can leave behind. This is a genuine second attempt, not a
silent substitute for the real thing: the pipeline only reaches for it
after preflight's normal open attempt already found the document
unsupported, and if repair changes the outcome, that fact is recorded in
the document's extraction_methods (surfaced in the evidence manifest) —
never applied invisibly.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import pikepdf


def attempt_repair(input_path: Path) -> Optional[bytes]:
    """Returns repaired PDF bytes, or None if pikepdf can't open/repair
    the document either (in which case the original UNSUPPORTED verdict
    stands unchanged)."""
    try:
        with pikepdf.open(input_path) as pdf:
            buf = io.BytesIO()
            pdf.save(buf)
            return buf.getvalue()
    except Exception:  # noqa: BLE001 - repair failing just means "no help available"
        return None
