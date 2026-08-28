"""API key authentication.

Local, file-based key store — deliberately not a database or external
auth service, matching the "nothing leaves this process, nothing shared
beyond this run" model the rest of the product follows (n2n/keys.py,
n2n/webapp/sessions.py). This is the right storage for a local/single-
tenant deployment; a hosted multi-tenant SaaS would need a real
database-backed store instead — that's a genuinely different
architecture, not solved here (see README's security-layer section).

Keys are never stored in plaintext or logged: only a SHA-256 hash is
persisted, and the plaintext is shown to the caller exactly once, at
creation time — standard practice for API key systems (GitHub, Stripe,
etc. all work this way).
"""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

DEFAULT_KEYS_FILE = Path.home() / ".n2n" / "keys" / "api_keys.json"
KEY_PREFIX = "n2n_live_"


@dataclass
class ApiKeyRecord:
    id: str
    name: str
    hashed_key: str
    created_at: float
    last_used_at: Optional[float] = None
    revoked: bool = False

    def public_dict(self) -> dict:
        """Everything EXCEPT the hash — safe to return from a list/status
        endpoint or print to a terminal."""
        d = asdict(self)
        d.pop("hashed_key")
        return d


def _hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


class ApiKeyStore:
    def __init__(self, path: Path = DEFAULT_KEYS_FILE) -> None:
        self.path = path

    def _load(self) -> list[ApiKeyRecord]:
        if not self.path.exists():
            return []
        raw = json.loads(self.path.read_text())
        return [ApiKeyRecord(**entry) for entry in raw]

    def _save(self, records: list[ApiKeyRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps([asdict(r) for r in records], indent=2))
        self.path.chmod(0o600)

    def create(self, name: str) -> tuple[str, ApiKeyRecord]:
        plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
        record = ApiKeyRecord(
            id=secrets.token_hex(8),
            name=name,
            hashed_key=_hash_key(plaintext),
            created_at=time.time(),
        )
        records = self._load()
        records.append(record)
        self._save(records)
        return plaintext, record

    def list(self) -> list[ApiKeyRecord]:
        return self._load()

    def revoke(self, key_id: str) -> bool:
        records = self._load()
        found = False
        for record in records:
            if record.id == key_id:
                record.revoked = True
                found = True
        if found:
            self._save(records)
        return found

    def verify(self, plaintext: str) -> Optional[ApiKeyRecord]:
        if not plaintext:
            return None
        hashed = _hash_key(plaintext)
        records = self._load()
        for record in records:
            if record.hashed_key == hashed and not record.revoked:
                record.last_used_at = time.time()
                self._save(records)
                return record
        return None

    def is_empty(self) -> bool:
        return len(self._load()) == 0


store = ApiKeyStore()
