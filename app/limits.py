"""In-memory sliding-window rate limit keyed by ``sha256(salt + ip)``.

The client IP is never stored or logged: only the digest is used as a dict key,
so a dump of the process memory does not leak addresses (nor does any log line).

An **empty** salt would make the digest brute-forceable (the IPv4 space is
small), so it is replaced by a random per-process salt (``secrets.token_hex``)
that is never printed or logged; a configured salt is used as is (digests stay
stable across restarts).  Expired keys are swept periodically instead of living
in the dict forever.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from collections import deque
from typing import Callable

#: Calls between two sweeps of stale keys (cheap and bounded).
SWEEP_EVERY = 512


class RateLimiter:
    """Allow ``limit`` requests per ``window_s`` seconds for each hashed IP."""

    def __init__(
        self,
        limit: int,
        window_s: float,
        salt: str = "",
        clock: Callable[[], float] | None = None,
        sweep_every: int = SWEEP_EVERY,
    ) -> None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        self.limit = int(limit)
        self.window_s = float(window_s)
        # An empty salt is replaced by a random per-process one: never logged.
        self._salt = salt or secrets.token_hex(32)
        self._clock = clock or time.monotonic
        self._sweep_every = max(1, int(sweep_every))
        self._calls = 0
        self._hits: dict[str, deque[float]] = {}

    def __repr__(self) -> str:  # the salt never shows up in repr
        return f"RateLimiter(limit={self.limit}, window_s={self.window_s})"

    def key(self, ip: str) -> str:
        """Opaque key for ``ip``; the address itself is never kept."""
        return hashlib.sha256((self._salt + (ip or "")).encode("utf-8")).hexdigest()

    def allow(self, ip: str) -> bool:
        """Record one request for ``ip``; ``False`` when the window is full."""
        now = self._clock()
        key = self.key(ip)
        hits = self._hits.setdefault(key, deque())
        cutoff = now - self.window_s
        while hits and hits[0] <= cutoff:
            hits.popleft()
        if len(hits) >= self.limit:
            allowed = False
        else:
            hits.append(now)
            allowed = True

        self._calls += 1
        if self._calls % self._sweep_every == 0:
            self._sweep(cutoff)
        return allowed

    def _sweep(self, cutoff: float) -> None:
        """Drop keys whose window is empty (no request from that IP is live)."""
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale:
            del self._hits[key]

    def count(self, ip: str) -> int:
        """Accepted requests still inside the window (for tests/diagnostics)."""
        return len(self._hits.get(self.key(ip), ()))
