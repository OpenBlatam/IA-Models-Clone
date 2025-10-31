import asyncio
from unittest.mock import AsyncMock, patch
import pytest

from network_utils import NetworkUtils
from vulnerability_scanner import WebVulnerabilityScanner


@pytest.mark.asyncio
async def test_e2e_network_connectivity_then_scan():
    net = NetworkUtils()
    scanner = WebVulnerabilityScanner()

    # Simulate reachable host/port
    fake_reader = AsyncMock()
    fake_writer = AsyncMock()
    fake_writer.close = AsyncMock()
    fake_writer.wait_closed = AsyncMock()

    with patch("asyncio.open_connection", return_value=(fake_reader, fake_writer)):
        conn = await net.check_host_connectivity('example.com', 80)
    assert conn.is_connection_successful is True

    # Follow-up: scanning an HTTP URL for missing headers
    from aioresponses import aioresponses
    with aioresponses() as m:
        m.get('http://example.com/', status=200, body='<html/>', headers={})
        findings = await scanner.scan_single_url('http://example.com/')

    # Expect at least one missing header finding
    assert any(f.vulnerability_type == 'missing_security_header' for f in findings)

