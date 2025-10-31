import types
import pytest

from network_utils import NetworkUtils
import network_utils as mod


@pytest.mark.asyncio
async def test_dns_records_mock_via_resolver_instance(monkeypatch):
    class FakeRRSet:
        ttl = 42

    class FakeAnswers(list):
        rrset = FakeRRSet()

    class FakeResolver:
        def resolve(self, hostname, record_type):
            return FakeAnswers(["8.8.8.8"])

    monkeypatch.setattr(mod.dns.resolver, "Resolver", lambda: FakeResolver())

    u = NetworkUtils()
    info = await u.get_dns_records("example.com", "A")
    assert info.is_resolution_successful is True
    assert info.resolved_addresses == ["8.8.8.8"]
    assert info.ttl_value == 42


