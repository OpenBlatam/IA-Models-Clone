import asyncio
import pytest

from port_scanner import AsyncPortScanner


@pytest.mark.asyncio
async def test_scan_port_range_collects_all(monkeypatch):
    async def fake_open_connection(host, port):
        # Alternate open/closed by port parity
        if port % 2 == 0:
            class W:
                def close(self):
                    pass

                async def wait_closed(self):
                    return None

            return object(), W()
        raise ConnectionRefusedError()

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    scanner = AsyncPortScanner(timeout_seconds=0.05)
    results = await scanner.scan_port_range("127.0.0.1", 80, 85)

    # Expect 3 open ports: 80, 82, 84
    open_ports = [r.target_port for r in results if r.is_port_open]
    assert open_ports == [80, 82, 84]













