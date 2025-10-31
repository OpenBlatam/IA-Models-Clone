from unittest.mock import MagicMock, patch

import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_ssl_certificate_handles_exceptions():
    u = NetworkUtils()

    # Make context.wrap_socket raise, simulating handshake/SSL error
    fake_context = MagicMock()
    fake_context.wrap_socket.side_effect = RuntimeError("ssl failure")

    with patch("ssl.create_default_context", return_value=fake_context), \
         patch("socket.create_connection"):
        info = await u.check_ssl_certificate("example.com", 443)

    assert info["is_certificate_valid"] is False
    assert isinstance(info.get("validation_errors"), list)
    assert any("ssl failure" in e.lower() for e in info["validation_errors"]) or info["validation_errors"] == []


