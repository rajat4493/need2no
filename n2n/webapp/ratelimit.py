"""In-memory per-API-key rate limiting.

Deliberately in-process, not Redis-backed — correct for a single-process
local/self-hosted deployment (matches the rest of the product's no-
external-dependency model) but NOT sufficient for a multi-instance SaaS
deployment, where limits would need to be shared across instances. That's
a real, known gap — see README's security-layer section — not solved
here.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque

DEFAULT_LIMIT = 30
DEFAULT_WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, limit: int = DEFAULT_LIMIT, window_seconds: float = DEFAULT_WINDOW_SECONDS) -> None:
        self.limit = limit
        self.window_seconds = window_seconds
        self._hits: dict[str, deque] = defaultdict(deque)

    def check(self, key: str) -> bool:
        """Returns True and records the hit if the request is allowed;
        returns False (recording nothing) if the caller is over the
        limit for the current window."""
        now = time.time()
        hits = self._hits[key]
        while hits and now - hits[0] > self.window_seconds:
            hits.popleft()
        if len(hits) >= self.limit:
            return False
        hits.append(now)
        return True

    def retry_after_seconds(self, key: str) -> float:
        hits = self._hits[key]
        if not hits:
            return 0.0
        return max(0.0, self.window_seconds - (time.time() - hits[0]))


limiter = RateLimiter()
