from __future__ import annotations

import asyncio
import time
import pytest

from agents.backend.onyx.server.features.blaze_ai.engines import EngineManager


async def _always_fail(payload):  # type: ignore[no-untyped-def]
    raise RuntimeError("fail")


@pytest.mark.asyncio
async def test_circuit_breaker_opens_and_cools_down():
    mgr = EngineManager(
        default_timeout_seconds=0.5,
        breaker_error_threshold=1,
        breaker_min_calls=1,
        breaker_cooldown_seconds=0.2,
    )
    mgr.register("bad", _always_fail)

    # first call -> error
    r1 = await mgr.process({"_engine": "bad"})
    assert r1.get("ok") is False and r1.get("error")

    # immediately calling again should be circuit open
    r2 = await mgr.process({"_engine": "bad"})
    assert r2.get("ok") is False and r2.get("error") == "circuit_open"

    # after cooldown, circuit closes and engine executed (still errors)
    await asyncio.sleep(0.25)
    r3 = await mgr.process({"_engine": "bad"})
    assert r3.get("ok") is False and r3.get("error") != "circuit_open"


