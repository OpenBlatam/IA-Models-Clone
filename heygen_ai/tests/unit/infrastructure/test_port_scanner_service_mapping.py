from port_scanner import AsyncPortScanner


def test_service_port_mapping_contains_expected_services():
    s = AsyncPortScanner()
    for p, name in [(22, 'ssh'), (80, 'http'), (443, 'https'), (8080, 'http_alt'), (8443, 'https_alt')]:
        assert s.service_port_mapping.get(p) == name


