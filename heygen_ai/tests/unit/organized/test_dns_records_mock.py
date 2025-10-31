import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_get_dns_records_mocked(monkeypatch):
    class FakeRRSet:
        ttl = 300

    class FakeAnswers(list):
        rrset = FakeRRSet()

    class FakeResolver:
        def resolve(self, hostname, record_type):
            return FakeAnswers(["1.2.3.4", "5.6.7.8"])

    import network_utils as mod
    monkeypatch.setattr(mod.dns.resolver, "Resolver", lambda: FakeResolver())

    u = NetworkUtils()
    info = await u.get_dns_records("example.com", "A")
    assert info.is_resolution_successful is True
    assert info.ttl_value == 300
    assert info.resolved_addresses == ["1.2.3.4", "5.6.7.8"]













