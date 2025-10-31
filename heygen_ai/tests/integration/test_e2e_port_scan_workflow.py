import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from port_scanner import AsyncPortScanner, PortScanResult


@pytest.mark.asyncio
async def test_e2e_port_scan_then_group_and_filter():
    s = AsyncPortScanner()

    fake_reader = AsyncMock()
    fake_writer = AsyncMock()
    fake_writer.close = AsyncMock()
    fake_writer.wait_closed = AsyncMock()

    # Simulate open on 80 and 22, refused on others
    async def opener(host, port):
        if port in (80, 22):
            return fake_reader, fake_writer
        raise ConnectionRefusedError()

    with patch("asyncio.open_connection", side_effect=opener):
        results = await s.scan_port_range("127.0.0.1", 22, 80)

    open_ports = s.filter_open_ports(results)
    grouped = s.group_by_service(open_ports)

    assert any(r.target_port == 22 for r in open_ports)
    assert any(r.target_port == 80 for r in open_ports)
    # Should have at least 'http' and 'ssh' groups
    assert 'http' in grouped and 'ssh' in grouped

