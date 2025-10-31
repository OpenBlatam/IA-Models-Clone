import pytest
import ssl as _ssl

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_check_ssl_certificate_no_cert(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSSock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getpeercert(self):
            return None

        verify_mode = _ssl.CERT_REQUIRED

    class FakeContext:
        def wrap_socket(self, sock, server_hostname=None):
            return FakeSSock()

    import network_utils as mod
    monkeypatch.setattr(mod.socket, "create_connection", lambda *_args, **_kw: FakeSocket())
    monkeypatch.setattr(mod.ssl, "create_default_context", lambda: FakeContext())

    utils = NetworkUtils()
    info = await utils.check_ssl_certificate("example.com", 443)
    assert info["is_certificate_valid"] is False
    assert info["validation_errors"] == []


@pytest.mark.asyncio
async def test_check_ssl_certificate_weak_cipher(monkeypatch):
    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    class FakeSSock:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def getpeercert(self):
            return {"subject": [( ("CN", "ex"), )], "issuer": [( ("CN", "ca"), )]}

        def cipher(self):
            return ("RC4-MD5", "TLSv1.0", 128)

        verify_mode = _ssl.CERT_REQUIRED

    class FakeContext:
        def wrap_socket(self, sock, server_hostname=None):
            return FakeSSock()

    import network_utils as mod
    monkeypatch.setattr(mod.socket, "create_connection", lambda *_args, **_kw: FakeSocket())
    monkeypatch.setattr(mod.ssl, "create_default_context", lambda: FakeContext())

    utils = NetworkUtils()
    info = await utils.check_ssl_certificate("example.com", 443)
    assert info["is_certificate_valid"] is True
    assert info["has_strong_cipher"] is False













