import asyncio
import pytest

from port_scanner import AsyncPortScanner, PortScanResult


@pytest.mark.asyncio
async def test_scan_single_port_open(monkeypatch):
    class FakeWriter:
        def close(self):
            pass

        async def wait_closed(self):
            return None

    async def fake_open_connection(host, port):
        return object(), FakeWriter()

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    scanner = AsyncPortScanner(timeout_seconds=0.1)
    result = await scanner.scan_single_port("127.0.0.1", 80)
    assert isinstance(result, PortScanResult)
    assert result.is_port_open is True
    assert result.target_port == 80


@pytest.mark.asyncio
async def test_scan_single_port_refused(monkeypatch):
    async def fake_open_connection(host, port):
        raise ConnectionRefusedError()

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    scanner = AsyncPortScanner(timeout_seconds=0.1)
    result = await scanner.scan_single_port("127.0.0.1", 1)
    assert result.is_port_open is False
    assert result.error_message == "Connection refused"


@pytest.mark.asyncio
async def test_filter_and_group_helpers():
    scanner = AsyncPortScanner()
    results = [
        PortScanResult(target_host="h", target_port=80, is_port_open=True, service_name="http"),
        PortScanResult(target_host="h", target_port=22, is_port_open=False, service_name="ssh"),
        PortScanResult(target_host="h", target_port=443, is_port_open=True, service_name="https"),
    ]

    open_ports = scanner.filter_open_ports(results)
    assert len(open_ports) == 2

    grouped = scanner.group_by_service(open_ports)
    assert set(grouped.keys()) == {"http", "https"}



