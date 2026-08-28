"""In-memory, single-process session store for certified downloads.

Deliberately not a database or cache service — this product's whole pitch
is "no telemetry, nothing leaves the process" (spec section 6), so a
release token maps to a local temp directory that gets cleaned up, not to
any persisted or external record.
"""

from __future__ import annotations

import secrets
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

SESSION_TTL_SECONDS = 30 * 60


@dataclass
class DownloadSession:
    token: str
    output_path: Path
    manifest_path: Path
    created_at: float


class SessionStore:
    def __init__(self, ttl_seconds: float = SESSION_TTL_SECONDS) -> None:
        self._sessions: dict[str, DownloadSession] = {}
        self._ttl_seconds = ttl_seconds

    def create(self, output_path: Path, manifest_path: Path) -> str:
        self._purge_expired()
        token = secrets.token_urlsafe(24)
        self._sessions[token] = DownloadSession(
            token=token,
            output_path=output_path,
            manifest_path=manifest_path,
            created_at=time.time(),
        )
        return token

    def get(self, token: str) -> DownloadSession | None:
        self._purge_expired()
        return self._sessions.get(token)

    def _purge_expired(self) -> None:
        now = time.time()
        expired = [
            token
            for token, session in self._sessions.items()
            if now - session.created_at > self._ttl_seconds
        ]
        for token in expired:
            session = self._sessions.pop(token)
            shutil.rmtree(session.output_path.parent, ignore_errors=True)

    def new_work_dir(self) -> Path:
        return Path(tempfile.mkdtemp(prefix="n2n_session_"))


# One store per process — matches the "nothing persisted, nothing shared
# beyond this run" model; restarting the server drops all pending
# downloads, which is the correct behaviour for a tool that promises nothing
# survives it.
store = SessionStore()
