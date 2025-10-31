import asyncio
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_timeout(monkeypatch):
    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            class _R:
                async def __aenter__(self_inner):
                    raise asyncio.TimeoutError("timeout")

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return _R()

    import network_utils as mod
    monkeypatch.setattr(mod, "aiohttp", type("x", (), {"ClientSession": lambda timeout=None: FakeSession(), "ClientTimeout": lambda total: total}))

    u = NetworkUtils()
    info = await u.check_http_status("https://example.org", timeout=0.01)
    assert info["is_accessible"] is False
    assert isinstance(info["error_message"], str)













