from port_scanner import AsyncPortScanner, PortScanResult
from datetime import datetime


def test_group_by_service_includes_unknown():
    scanner = AsyncPortScanner()
    results = [
        PortScanResult(target_host="h", target_port=12345, is_port_open=True, service_name="unknown", scan_timestamp=datetime.utcnow()),
        PortScanResult(target_host="h", target_port=80, is_port_open=True, service_name="http", scan_timestamp=datetime.utcnow()),
    ]
    grouped = scanner.group_by_service(results)
    assert "unknown" in grouped and "http" in grouped
    assert len(grouped["unknown"]) == 1













