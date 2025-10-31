import asyncio
from types import SimpleNamespace
import ssl
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_http_status_success(monkeypatch):
    class FakeResponse:
        def __init__(self):
            self.status = 200
            self.headers = {"content-type": "text/html", "server": "fake"}

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    class FakeSession:
        def __init__(self, *_, **__):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            return FakeResponse()

    # Patch aiohttp.ClientSession used inside module
    import network_utils as mod
    monkeypatch.setattr(mod, "aiohttp", SimpleNamespace(ClientSession=FakeSession, ClientTimeout=lambda total: total))

    utils = NetworkUtils(default_timeout=1.0)
    info = await utils.check_http_status("https://example.com")
    assert info["is_accessible"] is True
    assert info["status_code"] == 200
    assert info["content_type"] == "text/html"
    assert info["server_header"] == "fake"


@pytest.mark.asyncio
async def test_check_http_status_error(monkeypatch):
    class FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def get(self, url):
            raise RuntimeError("network error")

    import network_utils as mod
    monkeypatch.setattr(mod, "aiohttp", SimpleNamespace(ClientSession=lambda timeout: FakeSession(), ClientTimeout=lambda total: total))

    utils = NetworkUtils(default_timeout=1.0)
    info = await utils.check_http_status("https://example.com")
    assert info["is_accessible"] is False
    assert "network error" in info["error_message"]


def test_is_valid_ip_and_hostname():
    utils = NetworkUtils()
    assert utils.is_valid_ip_address("127.0.0.1") is True
    assert utils.is_valid_ip_address("999.0.0.1") is False
    assert utils.is_valid_hostname("example.com") is True
    assert utils.is_valid_hostname("") is False
    assert utils.is_valid_hostname("bad host") is False


@pytest.mark.asyncio
async def test_check_host_connectivity_success(monkeypatch):
    class FakeWriter:
        def get_extra_info(self, name):
            return None

        def close(self):
            pass

        async def wait_closed(self):
            return None

    async def fake_open_connection(host, port):
        return object(), FakeWriter()

    import network_utils as mod
    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)
    # Avoid real DNS
    monkeypatch.setattr(mod, "socket", SimpleNamespace(gethostbyname=lambda h: "93.184.216.34"))

    utils = NetworkUtils(default_timeout=0.1)
    info = await utils.check_host_connectivity("example.com", 80)
    assert info.is_connection_successful is True
    assert info.hostname == "example.com"
    assert info.port == 80
    assert info.ip_address == "93.184.216.34"


@pytest.mark.asyncio
async def test_check_ssl_certificate_handles_errors(monkeypatch):
    # Force an error path
    def fake_create_connection(addr):
        raise OSError("fail")

    import network_utils as mod
    monkeypatch.setattr(mod, "socket", SimpleNamespace(create_connection=fake_create_connection))
    monkeypatch.setattr(mod, "ssl", ssl)

    utils = NetworkUtils()
    info = await utils.check_ssl_certificate("example.com")
    assert info["is_certificate_valid"] is False
    assert len(info["validation_errors"]) >= 1



