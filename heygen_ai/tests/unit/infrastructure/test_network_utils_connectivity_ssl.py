import asyncio
import types
import ssl as _ssl
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_host_connectivity_with_ssl_info(monkeypatch):
    class FakeSSLObj:
        def version(self):
            return "TLSv1.3"

        def cipher(self):
            return ("TLS_AES_128_GCM_SHA256", "TLSv1.3", 128)

        verify_mode = _ssl.CERT_REQUIRED

    class FakeWriter:
        def get_extra_info(self, key):
            if key == "ssl_object":
                return FakeSSLObj()
            return None

        async def wait_closed(self):
            return None

        def close(self):
            return None

    async def fake_open_connection(host, port):
        return object(), FakeWriter()

    import network_utils as mod
    monkeypatch.setattr(mod.asyncio, "open_connection", fake_open_connection)
    # Avoid real DNS
    monkeypatch.setattr(
        NetworkUtils,
        "resolve_hostname_to_ip",
        lambda self, h: asyncio.sleep(0, result="127.0.0.1"),
    )

    u = NetworkUtils()
    info = await u.check_host_connectivity("example.com", 443)
    assert info.is_connection_successful is True
    assert info.ssl_info and info.ssl_info["ssl_version"] == "TLSv1.3"
    assert info.ssl_info["certificate_verified"] is True


@pytest.mark.asyncio
async def test_check_host_connectivity_timeout(monkeypatch):
    async def fake_open_connection(host, port):
        raise asyncio.TimeoutError

    import network_utils as mod
    monkeypatch.setattr(mod.asyncio, "open_connection", fake_open_connection)
    monkeypatch.setattr(
        NetworkUtils, "resolve_hostname_to_ip", lambda self, h: asyncio.sleep(0, result="unresolved")
    )

    u = NetworkUtils()
    info = await u.check_host_connectivity("nohost", 80, timeout=0.01)
    assert info.is_connection_successful is False
    assert info.error_message == "Connection timeout"













