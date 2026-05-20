from __future__ import annotations

import asyncio
import time
import pytest

from agents.backend.onyx.server.features.blaze_ai.engines import EngineManager
from agents.backend.onyx.server.features.blaze_ai.utils.runtime_metrics import runtime_metrics


async def _ok_engine(payload):  # type: ignore[no-untyped-def]
    return {"ok": True, "echo": payload}


async def _error_engine(payload):  # type: ignore[no-untyped-def]
    raise RuntimeError("boom")


async def _slow_engine(payload):  # type: ignore[no-untyped-def]
    await asyncio.sleep(0.2)
    return {"ok": True}


@pytest.mark.asyncio
async def test_engine_manager_success_and_metrics():
    mgr = EngineManager(default_timeout_seconds=1.0)
    mgr.register("ok", _ok_engine)
    resp = await mgr.process({"_engine": "ok", "data": 1})
    assert resp.get("ok") is True
    assert resp.get("engine") == "ok"
    assert isinstance(resp.get("elapsed_ms"), (int, float))

    snap = runtime_metrics.snapshot()
    assert "ok" in snap
    assert snap["ok"]["count"] >= 1


@pytest.mark.asyncio
async def test_engine_manager_error_and_timeout():
    mgr = EngineManager(default_timeout_seconds=0.1)
    mgr.register("err", _error_engine)
    mgr.register("slow", _slow_engine)

    # error case
    resp_err = await mgr.process({"_engine": "err"})
    assert resp_err.get("ok") is False
    assert resp_err.get("error")

    # timeout case
    t0 = time.perf_counter()
    resp_to = await mgr.process({"_engine": "slow"})
    dt = time.perf_counter() - t0
    assert resp_to.get("ok") is False and resp_to.get("error") == "timeout"
    assert dt < 0.5  # ensured timeout kicked in


