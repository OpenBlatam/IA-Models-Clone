import asyncio
import pytest

from port_scanner import AsyncPortScanner


@pytest.mark.asyncio
async def test_scan_port_range_ignores_exceptions(monkeypatch):
    async def fake_open_connection(host, port):
        if port % 2 == 0:
            raise ConnectionRefusedError
        if port % 3 == 0:
            raise asyncio.TimeoutError
        # simulate open
        class W:
            def close(self):
                pass

            async def wait_closed(self):
                pass

        return object(), W()

    import port_scanner as mod
    monkeypatch.setattr(mod.asyncio, "open_connection", fake_open_connection)

    scanner = AsyncPortScanner(timeout_seconds=0.01)
    results = await scanner.scan_port_range("localhost", 1, 10)
    # All results returned without raising exceptions
    assert len(results) == 10
    # Some will be closed/open based on our fake
    assert any(r.is_port_open for r in results)













