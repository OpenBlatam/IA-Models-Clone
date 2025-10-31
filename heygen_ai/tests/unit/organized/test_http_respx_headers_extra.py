import pytest
from aioresponses import aioresponses

from vulnerability_scanner import WebVulnerabilityScanner


@pytest.mark.asyncio
async def test_detects_missing_headers_with_aioresponses():
    s = WebVulnerabilityScanner()
    with aioresponses() as m:
        m.get('http://h.test/', status=200, headers={})
        res = await s.scan_single_url('http://h.test/')
    assert any(f.vulnerability_type == 'missing_security_header' for f in res)


