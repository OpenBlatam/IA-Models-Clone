import types
import pytest

import network_utils as mod
from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_dns_records_failure_via_function_mock(monkeypatch):
    def boom(hostname, record_type):
        raise RuntimeError("dns fail")

    monkeypatch.setattr(mod.dns.resolver, "resolve", boom, raising=True)
    u = NetworkUtils()
    info = await u.get_dns_records("x", "A")
    assert info.is_resolution_successful is False
    assert info.resolved_addresses == []


