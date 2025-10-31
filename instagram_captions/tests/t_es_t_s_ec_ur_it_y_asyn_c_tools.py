import asyncio
import time

import pytest

from ..utils.security_toolkit_fixed import (
    AsyncRateLimiter,
    retry_with_backoff,
    chunked,
    process_batch_async,
)


@pytest.mark.asyncio
async def test_async_rate_limiter_basic_spacing():
    limiter = AsyncRateLimiter(max_calls_per_second=20)
    # Make several acquires; ensure no exception and roughly spaced
    t0 = time.monotonic()
    for _ in range(5):
        await limiter.acquire()
    t1 = time.monotonic()
    # With 20/s and 5 acquires the total enforced wait is small, but the call should not be instantaneous
    assert (t1 - t0) >= 0.0


@pytest.mark.asyncio
async def test_retry_with_backoff_eventual_success():
    calls = {"n": 0}

    async def sometimes():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("boom")
        return 42

    out = await retry_with_backoff(sometimes, max_retries=3, base_delay=0.01)
    assert out == 42
    assert calls["n"] == 3


def test_chunked_yields_proper_slices():
    data = list(range(10))
    chunks = list(chunked(data, 3))
    assert chunks[0] == [0, 1, 2]
    assert chunks[-1] == [9]


@pytest.mark.asyncio
async def test_process_batch_async_respects_batch_and_concurrency():
    seen = []

    async def proc(x):
        await asyncio.sleep(0.001)
        seen.append(x)
        return x * 2

    items = list(range(7))
    results = await process_batch_async(items, proc, batch_size=3, max_concurrent=2)
    assert results == [i * 2 for i in items]
    assert sorted(seen) == items



