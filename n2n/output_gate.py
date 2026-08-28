"""Structural enforcement of the single most safety-critical invariant in
N2N: it must be physically impossible to write a certified output file
except from the PASS_AUTO code path.

This is NOT "check status, then write the file" — that pattern is a bug
waiting to happen the moment someone adds a new call site. Instead:

  * `write_certified_output` refuses to run without a `ReleaseToken`.
  * A `ReleaseToken` can only be minted by `mint_release_token`, which is
    private to this module and requires the exact bytes being released.
  * `n2n/pipeline.py` is the ONLY caller of `mint_release_token`, and it is
    only invoked inside the `if status is ReleaseStatus.PASS_AUTO:` branch.

A test (tests/test_output_gate.py) additionally asserts, by source
inspection, that no module other than n2n/pipeline.py calls
`mint_release_token` — so a future edit that tries to route around the
gate fails CI, not just code review.
"""

from __future__ import annotations

import hashlib
import hmac
import os
from pathlib import Path

_PROCESS_SECRET = os.urandom(32)


class ReleaseToken:
    """An unforgeable, single-use proof that the holder is authorized to
    release exactly `payload_hash`. Cannot be constructed directly."""

    __slots__ = ("_payload_hash", "_mac")

    def __init__(self, payload_hash: bytes, mac: bytes) -> None:
        self._payload_hash = payload_hash
        self._mac = mac


def mint_release_token(payload: bytes) -> ReleaseToken:
    """Only call this from the PASS_AUTO branch of n2n/pipeline.py."""
    payload_hash = hashlib.sha256(payload).digest()
    mac = hmac.new(_PROCESS_SECRET, payload_hash, hashlib.sha256).digest()
    return ReleaseToken(payload_hash, mac)


def _token_is_valid_for(token: ReleaseToken, payload: bytes) -> bool:
    if not isinstance(token, ReleaseToken):
        return False
    payload_hash = hashlib.sha256(payload).digest()
    if not hmac.compare_digest(payload_hash, token._payload_hash):
        return False
    expected_mac = hmac.new(_PROCESS_SECRET, payload_hash, hashlib.sha256).digest()
    return hmac.compare_digest(expected_mac, token._mac)


def write_certified_output(
    token: ReleaseToken,
    output_payload: bytes,
    output_path: Path,
    manifest_payload: bytes,
    manifest_path: Path,
) -> None:
    """The only function in the codebase permitted to write a certified
    output file. Raises PermissionError if the token doesn't match the
    exact bytes being written."""
    if not _token_is_valid_for(token, output_payload):
        raise PermissionError(
            "Refusing to write output: no valid release token for this payload. "
            "Output can only be released from the PASS_AUTO decision path."
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(output_payload)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(manifest_payload)
