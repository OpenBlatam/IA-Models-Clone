import asyncio
from aioresponses import aioresponses
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_5xx_response():
    u = NetworkUtils()
    with aioresponses() as m:
        m.get('http://err.test/', status=503, headers={'server': 'ut'}, body='err')
        info = await u.check_http_status('http://err.test/')
    assert info['is_accessible'] is True
    assert info['status_code'] == 503
    assert info['server_header'] == 'ut'


@pytest.mark.asyncio
async def test_check_http_status_timeout_error():
    u = NetworkUtils()
    with aioresponses() as m:
        m.get('http://timeout.test/', exception=asyncio.TimeoutError())
        info = await u.check_http_status('http://timeout.test/')
    assert info['is_accessible'] is False
    assert 'error_message' in info


