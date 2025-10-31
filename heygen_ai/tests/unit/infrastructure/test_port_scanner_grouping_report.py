from port_scanner import AsyncPortScanner, PortScanResult


def test_group_by_service_aggregates_correctly():
    s = AsyncPortScanner()
    results = [
        PortScanResult(target_host='h', target_port=80, is_port_open=True, service_name='http'),
        PortScanResult(target_host='h', target_port=8080, is_port_open=True, service_name='http_alt'),
        PortScanResult(target_host='h', target_port=443, is_port_open=False, service_name='https'),
        PortScanResult(target_host='h', target_port=22, is_port_open=True, service_name='ssh'),
    ]
    grouped = s.group_by_service(results)
    assert 'http' in grouped and any(r.target_port == 80 for r in grouped['http'])
    assert 'ssh' in grouped and len(grouped['ssh']) == 1


def test_filter_open_ports_only():
    s = AsyncPortScanner()
    results = [
        PortScanResult(target_host='h', target_port=80, is_port_open=True),
        PortScanResult(target_host='h', target_port=443, is_port_open=False),
    ]
    open_only = s.filter_open_ports(results)
    assert len(open_only) == 1 and open_only[0].target_port == 80


