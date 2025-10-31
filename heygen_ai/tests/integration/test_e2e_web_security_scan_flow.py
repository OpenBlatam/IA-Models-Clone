import pytest
from aioresponses import aioresponses

from vulnerability_scanner import WebVulnerabilityScanner


@pytest.mark.asyncio
async def test_e2e_full_scan_and_report_generation():
    scanner = WebVulnerabilityScanner(max_concurrent_scans=5, request_timeout=5.0)

    urls = [
        'http://e2e-sql.test/?id=1\' OR 1=1--',
        'http://e2e-xss.test/?q=<script>alert(1)</script>',
        'https://e2e-secure.test/'
    ]

    with aioresponses() as m:
        # SQLi: body shows error; no security headers
        m.get(urls[0], status=200, body="error in your SQL syntax", headers={})
        # XSS: body contains <script>; weak headers
        m.get(urls[1], status=200, body='<html><script>alert(1)</script></html>', headers={'X-Frame-Options': 'DENY'})
        # Secure-ish: provide strong headers
        m.get(urls[2], status=200, body='<html>OK</html>', headers={
            'X-Frame-Options': 'DENY',
            'X-Content-Type-Options': 'nosniff',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'"
        })

        results = await scanner.scan_multiple_urls(urls)

    report = scanner.generate_scan_report(results)

    assert report['total_targets_scanned'] == 3
    assert report['total_vulnerabilities_found'] >= 2
    assert 'severity_distribution' in report
    # Expect at least a HIGH due to SQLi/XSS
    assert any(k in report['severity_distribution'] for k in ('high', 'critical'))
    assert isinstance(report['recommendations'], list) and len(report['recommendations']) > 0

