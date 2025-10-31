import ssl
from unittest.mock import MagicMock, patch

import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_is_valid_hostname_various_cases():
    u = NetworkUtils()
    # Valid
    assert u.is_valid_hostname("example.com") is True
    assert u.is_valid_hostname("sub.domain.example.") is True  # trailing dot allowed
    assert u.is_valid_hostname("a" * 63 + ".com") is True
    # Invalid
    assert u.is_valid_hostname("") is False
    assert u.is_valid_hostname("-bad.com") is False
    assert u.is_valid_hostname("bad-.com") is False
    assert u.is_valid_hostname("bad..com") is False
    assert u.is_valid_hostname("a" * 64 + ".com") is False  # label too long


@pytest.mark.asyncio
async def test_check_ssl_certificate_includes_alias_certificate_expiry():
    u = NetworkUtils()

    class _FakeSSock:
        def getpeercert(self):
            # notAfter format: '%b %d %H:%M:%S %Y %Z'
            return {
                'subject': ((('commonName', 'example.com'),),),
                'issuer': ((('organizationName', 'UnitTest CA'),),),
                'notAfter': 'Dec 31 23:59:59 2099 GMT',
            }

        def verify_mode(self):
            return ssl.CERT_REQUIRED

        def cipher(self):
            return ("TLS_AES_256_GCM_SHA384", "TLSv1.3", 256)

        # In real ssl objects, verify_mode is attribute; for our check we only
        # need the value comparison, so we expose attribute form as well
        verify_mode = ssl.CERT_REQUIRED

    fake_context = MagicMock()
    fake_context.wrap_socket.return_value.__enter__.return_value = _FakeSSock()

    with patch("ssl.create_default_context", return_value=fake_context), \
         patch("socket.create_connection"):
        info = await u.check_ssl_certificate("example.com", 443)

    assert info["is_certificate_valid"] is True
    assert info["certificate_subject"]
    assert info["certificate_issuer"]
    assert info["expiry_date"] is not None
    assert info["certificate_expiry"] == info["expiry_date"]


