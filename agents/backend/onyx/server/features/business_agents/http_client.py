import time
import asyncio
from typing import Optional, Dict, Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type


class CircuitBreaker:
    def __init__(self, fail_threshold: int, recovery_seconds: int):
        self.fail_threshold = fail_threshold
        self.recovery_seconds = recovery_seconds
        self.fail_count = 0
        self.opened_at: Optional[float] = None
        self._lock = asyncio.Lock()

    async def allow(self) -> bool:
        async with self._lock:
            if self.opened_at is None:
                return True
            if (time.time() - self.opened_at) >= self.recovery_seconds:
                # Half-open: allow one request and reset on success
                return True
            return False

    async def on_success(self):
        async with self._lock:
            self.fail_count = 0
            self.opened_at = None

    async def on_failure(self):
        async with self._lock:
            self.fail_count += 1
            if self.fail_count >= self.fail_threshold:
                self.opened_at = time.time()


class ResilientHTTPClient:
    def __init__(self, timeout_seconds: float, retries: int, cb_fail_threshold: int, cb_recovery_seconds: int):
        self.client = httpx.AsyncClient(timeout=timeout_seconds)
        self.cb = CircuitBreaker(cb_fail_threshold, cb_recovery_seconds)
        self.retries = retries

    async def close(self):
        await self.client.aclose()

    def _retry(self):
        return retry(
            reraise=True,
            stop=stop_after_attempt(self.retries),
            wait=wait_exponential(multiplier=0.2, min=0.2, max=2),
            retry=retry_if_exception_type((httpx.ReadTimeout, httpx.ConnectTimeout, httpx.NetworkError)),
        )

    async def get_json(self, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        if not await self.cb.allow():
            raise httpx.HTTPError("circuit_open")

        @_wrap := self._retry()
        @_wrap
        async def _do() -> Dict[str, Any]:
            try:
                resp = await self.client.get(url, headers=headers)
                resp.raise_for_status()
                await self.cb.on_success()
                return resp.json()
            except Exception:
                await self.cb.on_failure()
                raise

        return await _do()







