import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from port_scanner import AsyncPortScanner, PortScanResult


@pytest.mark.asyncio
async def test_scan_single_port_async_close_await():
    scanner = AsyncPortScanner(timeout_seconds=0.5)

    fake_reader = AsyncMock()
    fake_writer = AsyncMock()

    async def _close():
        return None

    async def _wait_closed():
        return None

    fake_writer.close = _close  # type: ignore[assignment]
    fake_writer.wait_closed = _wait_closed  # type: ignore[assignment]

    with patch("asyncio.open_connection", return_value=(fake_reader, fake_writer)):
        result = await scanner.scan_single_port("127.0.0.1", 80)

    assert isinstance(result, PortScanResult)
    assert result.is_port_open is True
    assert result.service_name is not None


@pytest.mark.asyncio
async def test_scan_port_range_filters_exceptions():
    scanner = AsyncPortScanner()

    async def fake_single(host, port):
        if port % 2 == 0:
            raise RuntimeError("boom")
        return PortScanResult(target_host=host, target_port=port, is_port_open=False)

    with patch.object(scanner, 'scan_single_port', side_effect=fake_single):
        results = await scanner.scan_port_range("localhost", 1, 5)

    # only odd port results should be present
    ports = {r.target_port for r in results}
    assert ports == {1, 3, 5}


