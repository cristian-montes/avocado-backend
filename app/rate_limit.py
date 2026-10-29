"""Per-user rate limiting.

NOTE (MVP): this is an in-memory (per-process) limiter. It is not a security
boundary — the Clerk JWT check is. Its job is to throttle over-consumption by
authorized users, not to stop spoofers (they can't produce a valid token).

Caveats of the in-memory approach:
- Resets when a container instance cold-starts or scales down (min-instances=0).
- With max-instances=N, an individual user could in principle spread across N
  instances and multiply their effective limit by N (trivial at N=2).

Post-MVP: move to a shared store (Firestore TTL or Redis/Memorystore) so limits
are global across instances.
"""
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException

from app import config


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        now = time.monotonic()
        with self._lock:
            dq = self._hits[key]
            while dq and now - dq[0] > self.window_seconds:
                dq.popleft()
            if len(dq) >= self.max_requests:
                raise HTTPException(
                    status_code=429,
                    detail="Too many requests. Please try again later.",
                )
            dq.append(now)

    def clear(self, key: str) -> None:
        with self._lock:
            self._hits.pop(key, None)


limiter = RateLimiter(config.RATE_LIMIT_REQUESTS, config.RATE_LIMIT_WINDOW_SECONDS)
