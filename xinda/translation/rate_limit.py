"""Token-bucket rate limiter honoring (rpm, tpm) provider caps."""

from __future__ import annotations

import asyncio


class RateLimiter:
    """Async token bucket. Call `reserve(tokens)` before each LLM request."""

    def __init__(self, rpm: int, tpm: int):
        self.rpm = rpm
        self.tpm = tpm
        self._bucket_req: float = float(rpm)
        self._bucket_tok: float = float(tpm)
        self._lock = asyncio.Lock()
        self._refill_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        """Kick off the refill task. Idempotent."""
        if self._refill_task is None or self._refill_task.done():
            self._refill_task = asyncio.create_task(self._refill_loop())

    async def _refill_loop(self) -> None:
        while True:
            await asyncio.sleep(1)
            async with self._lock:
                self._bucket_req = min(self.rpm, self._bucket_req + self.rpm / 60)
                self._bucket_tok = min(self.tpm, self._bucket_tok + self.tpm / 60)

    async def reserve(self, tokens_needed: int) -> None:
        """Block until both a request slot and `tokens_needed` tokens are available."""
        self.start()
        while True:
            async with self._lock:
                if self._bucket_req >= 1 and self._bucket_tok >= tokens_needed:
                    self._bucket_req -= 1
                    self._bucket_tok -= tokens_needed
                    return
            await asyncio.sleep(0.1)
