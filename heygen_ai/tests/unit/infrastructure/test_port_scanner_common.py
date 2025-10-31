import asyncio
import pytest

from port_scanner import AsyncPortScanner


@pytest.mark.asyncio
async def test_scan_common_ports_delegates_to_range(monkeypatch):
    called = {"count": 0}

    async def fake_scan_range(self, host, start, end):
        called["count"] += 1
        return []

    monkeypatch.setattr(AsyncPortScanner, "scan_port_range", fake_scan_range)

    scanner = AsyncPortScanner()
    res = await scanner.scan_common_ports("127.0.0.1")
    assert res == []
    assert called["count"] == 1













