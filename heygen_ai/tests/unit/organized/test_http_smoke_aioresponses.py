import pytest

aiorskip = pytest.importorskip("aioresponses")
from aioresponses import aioresponses  # type: ignore

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_http_smoke_ok():
    u = NetworkUtils()
    url = "https://api.service.local/health"
    with aioresponses() as mocked:
        mocked.get(url, status=204)
        info = await u.check_http_status(url)
        assert info["status_code"] == 204
        assert info["is_accessible"] is True













