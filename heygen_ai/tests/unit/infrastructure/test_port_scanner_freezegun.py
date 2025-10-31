import pytest

freezegun = pytest.importorskip("freezegun")  # skip if freezegun not installed
from freezegun import freeze_time  # type: ignore

from port_scanner import AsyncPortScanner
import asyncio


@pytest.mark.asyncio
@freeze_time("2025-01-01 00:00:00")
async def test_scan_single_port_timestamp_freeze(monkeypatch):
    async def fake_open_connection(host, port):
        class W:
            def close(self):
                pass

            async def wait_closed(self):
                pass

        return object(), W()

    import port_scanner as mod
    monkeypatch.setattr(mod.asyncio, "open_connection", fake_open_connection)

    s = AsyncPortScanner()
    res = await s.scan_single_port("localhost", 80)
    assert res.scan_timestamp.isoformat().startswith("2025-01-01T00:00:00")













