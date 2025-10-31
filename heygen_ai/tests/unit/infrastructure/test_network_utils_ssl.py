import ssl as std_ssl
from datetime import datetime, timedelta
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_ssl_certificate_success(monkeypatch):
    class FakeSSLSocket:
        def __init__(self):
            self._cipher = ("TLS_AES_128_GCM_SHA256", None, None)

        def getpeercert(self):
            future = (datetime.utcnow() + timedelta(days=365)).strftime('%b %d %H:%M:%S %Y GMT')
            return {
                'subject': ((('commonName', 'example.com'),),),
                'issuer': ((('organizationName', 'Test CA'),),),
                'notAfter': future,
            }

        def cipher(self):
            return self._cipher

        @property
        def verify_mode(self):
            return std_ssl.CERT_REQUIRED

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeContext:
        def wrap_socket(self, sock, server_hostname=None):
            return FakeSSLSocket()

    class FakeSocket:
        def __init__(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    import network_utils as mod
    # Patch create_default_context and create_connection only
    monkeypatch.setattr(mod.ssl, "create_default_context", lambda: FakeContext())
    monkeypatch.setattr(mod.socket, "create_connection", lambda addr: FakeSocket())

    utils = NetworkUtils()
    info = await utils.check_ssl_certificate("example.com", 443)
    assert info['is_certificate_valid'] is True
    assert info['is_hostname_matching'] is True
    assert info['has_strong_cipher'] is True


