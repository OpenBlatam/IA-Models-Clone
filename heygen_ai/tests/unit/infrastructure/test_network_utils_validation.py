import types
import pytest

from network_utils import NetworkUtils


def test_is_valid_hostname_various_cases():
    u = NetworkUtils()
    assert u.is_valid_hostname("example.com") is True
    assert u.is_valid_hostname("EXAMPLE.COM") is True  # upper allowed via lower() check
    assert u.is_valid_hostname("") is False
    assert u.is_valid_hostname("a" * 254) is False
    assert u.is_valid_hostname("bad host") is False
    assert u.is_valid_hostname("good-host.example") is True


@pytest.mark.asyncio
async def test_get_dns_records_error(monkeypatch):
    import socket as std_socket
    import network_utils as mod

    # Make resolver.resolve raise
    class FakeResolver:
        def resolve(self, *args, **kwargs):
            raise std_socket.gaierror()

    monkeypatch.setattr(mod.dns, "resolver", types.SimpleNamespace(Resolver=FakeResolver))

    u = NetworkUtils()
    info = await u.get_dns_records("nonexistent.example", "AAAA")
    assert info.is_resolution_successful is False
    assert info.resolved_addresses == []
    assert info.error_message is not None













