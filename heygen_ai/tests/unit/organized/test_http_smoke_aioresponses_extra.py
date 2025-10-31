import asyncio

import pytest
from aioresponses import aioresponses

from vulnerability_scanner import WebVulnerabilityScanner


@pytest.mark.asyncio
async def test_http_smoke_with_aioresponses_minimal():
    scanner = WebVulnerabilityScanner()
    with aioresponses() as m:
        m.get('http://smoke.test/', status=200, body='<html>ok</html>')
        res = await scanner.scan_single_url('http://smoke.test/')
    assert isinstance(res, list)


