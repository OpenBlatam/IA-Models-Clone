import pytest

aiorskip = pytest.importorskip("aioresponses")  # skip if aioresponses not installed
from aioresponses import aioresponses  # type: ignore

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_200_with_headers():
    utils = NetworkUtils()
    url = "https://example.com/path"
    with aioresponses() as mocked:
        mocked.get(url, status=200, headers={"content-type": "text/html", "server": "nginx"})
        info = await utils.check_http_status(url)
        assert info["is_accessible"] is True
        assert info["status_code"] == 200
        assert info["content_type"] == "text/html"
        assert info["server_header"] == "nginx"


@pytest.mark.asyncio
async def test_check_http_status_500_error():
    utils = NetworkUtils()
    url = "https://example.com/error"
    with aioresponses() as mocked:
        mocked.get(url, status=500)
        info = await utils.check_http_status(url)
        assert info["is_accessible"] is True
        assert info["status_code"] == 500













