import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_missing_headers(monkeypatch):
    class FakeResponse:
        status = 204
        headers = {}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return FakeResponse()

    import network_utils as mod
    monkeypatch.setattr(mod, "aiohttp", type("x", (), {"ClientSession": lambda timeout=None: FakeSession(), "ClientTimeout": lambda total: total}))

    u = NetworkUtils()
    info = await u.check_http_status("https://example.com/resource")
    assert info["is_accessible"] is True
    assert info["status_code"] == 204
    assert info["content_type"] is None
    assert info["server_header"] is None













