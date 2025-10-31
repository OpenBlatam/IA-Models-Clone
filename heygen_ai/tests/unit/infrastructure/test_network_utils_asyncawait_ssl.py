import asyncio
import ssl
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from network_utils import NetworkUtils, NetworkConnectionInfo


class _FakeSSLObject:
    def __init__(self):
        self.verify_mode = ssl.CERT_REQUIRED

    def version(self):
        return "TLSv1.3"

    def cipher(self):
        return ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)


@pytest.mark.asyncio
async def test_check_host_connectivity_with_awaitable_ssl_info():
    utils = NetworkUtils()

    fake_reader = AsyncMock()
    fake_writer = AsyncMock()

    async def _get_extra_info(name):  # simulate awaitable ssl_object retrieval
        if name == "ssl_object":
            return _FakeSSLObject()
        return None

    fake_writer.get_extra_info = _get_extra_info  # type: ignore[assignment]

    async def _close():
        return None

    async def _wait_closed():
        return None

    fake_writer.close = _close  # type: ignore[assignment]
    fake_writer.wait_closed = _wait_closed  # type: ignore[assignment]

    with patch("asyncio.open_connection", return_value=(fake_reader, fake_writer)):
        with patch.object(utils, "resolve_hostname_to_ip", return_value="127.0.0.1"):
            result = await utils.check_host_connectivity("example.com", 443)

    assert isinstance(result, NetworkConnectionInfo)
    assert result.is_connection_successful is True
    assert result.ssl_info is not None
    assert result.ssl_info["ssl_version"] == "TLSv1.3"
    assert result.ssl_info["certificate_verified"] is True


@pytest.mark.asyncio
async def test_get_dns_records_failure_path():
    utils = NetworkUtils()
    with patch("dns.resolver.resolve", side_effect=Exception("boom")):
        info = await utils.get_dns_records("example.com", "A")
    assert info.is_resolution_successful is False
    assert info.error_message and "boom" in info.error_message


@pytest.mark.asyncio
async def test_check_http_status_headers_parsing():
    utils = NetworkUtils()

    mock_response = AsyncMock()
    mock_response.status = 200
    mock_response.headers = {"content-type": "text/html", "server": "unit-test"}
    mock_response.text = AsyncMock(return_value="<html>OK</html>")

    class _DummyCM:
        async def __aenter__(self):
            return mock_response

        async def __aexit__(self, exc_type, exc, tb):
            return False

    with patch("aiohttp.ClientSession.get", return_value=_DummyCM()):
        info = await utils.check_http_status("http://example.com")

    assert info["is_accessible"] is True
    assert info["status_code"] == 200
    assert info["content_type"] == "text/html"
    assert info["server_header"] == "unit-test"


