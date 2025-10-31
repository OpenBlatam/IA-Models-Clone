import asyncio
import pytest

from port_scanner import AsyncPortScanner


@pytest.mark.asyncio
async def test_scan_single_port_generic_exception(monkeypatch):
    async def fake_open_connection(host, port):
        raise RuntimeError("boom")

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    scanner = AsyncPortScanner(timeout_seconds=0.05)
    res = await scanner.scan_single_port("127.0.0.1", 9999)
    assert res.is_port_open is False
    assert res.error_message == "boom"













