import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_raises(monkeypatch):
    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):  # context manager that raises during __aenter__
            class _R:
                async def __aenter__(self_inner):
                    raise RuntimeError("boom")

                async def __aexit__(self_inner, exc_type, exc, tb):
                    return False

            return _R()

    import network_utils as mod
    monkeypatch.setattr(mod, "aiohttp", type("x", (), {"ClientSession": lambda timeout=None: FakeSession(), "ClientTimeout": lambda total: total}))

    utils = NetworkUtils()
    info = await utils.check_http_status("https://example.com")
    assert info["is_accessible"] is False
    assert isinstance(info["error_message"], str) and "boom" in info["error_message"]













