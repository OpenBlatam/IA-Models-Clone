from port_scanner import AsyncPortScanner, PortScanResult


def test_group_by_service_ignores_none_service_names():
    s = AsyncPortScanner()
    results = [
        PortScanResult(target_host='h', target_port=1, is_port_open=True, service_name=None),
        PortScanResult(target_host='h', target_port=22, is_port_open=True, service_name='ssh'),
    ]
    grouped = s.group_by_service(results)
    assert 'ssh' in grouped and all(r.service_name is not None for r in grouped['ssh'])


