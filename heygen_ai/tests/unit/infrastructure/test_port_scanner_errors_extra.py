import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from port_scanner import AsyncPortScanner, PortScanResult


@pytest.mark.asyncio
async def test_scan_single_port_connection_refused_path():
    s = AsyncPortScanner()

    async def opener(*args, **kwargs):
        raise ConnectionRefusedError()

    with patch("asyncio.open_connection", side_effect=opener):
        res = await s.scan_single_port("127.0.0.1", 65535)
    assert isinstance(res, PortScanResult)
    assert res.is_port_open is False
    assert res.error_message == "Connection refused"


@pytest.mark.asyncio
async def test_scan_single_port_timeout_path():
    s = AsyncPortScanner(timeout_seconds=0.01)

    async def opener(*args, **kwargs):
        await asyncio.sleep(1)

    with patch("asyncio.open_connection", side_effect=opener):
        res = await s.scan_single_port("127.0.0.1", 1)
    assert res.is_port_open is False
    assert res.error_message == "Connection timeout"


