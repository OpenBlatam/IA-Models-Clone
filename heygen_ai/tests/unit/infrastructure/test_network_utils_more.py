import asyncio
import types
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_get_dns_records_success(monkeypatch):
    class FakeAnswers(list):
        class RRSET:
            ttl = 123

        rrset = RRSET()

    class FakeResolver:
        def resolve(self, hostname, record_type):
            return FakeAnswers(["93.184.216.34"])

    import network_utils as mod
    monkeypatch.setattr(mod.dns, "resolver", types.SimpleNamespace(Resolver=FakeResolver))

    utils = NetworkUtils()
    info = await utils.get_dns_records("example.com", "A")
    assert info.is_resolution_successful is True
    assert info.ttl_value == 123
    assert info.resolved_addresses == ["93.184.216.34"]


@pytest.mark.asyncio
async def test_ping_host_success(monkeypatch):
    class FakeProc:
        returncode = 0

        async def communicate(self):
            # Minimal ping output; our implementation uses simplified parsing
            return ("Reply from 127.0.0.1: bytes=32 time<1ms TTL=128\n".encode(), b"")

    async def fake_create_subprocess_exec(*args, **kwargs):
        return FakeProc()

    import network_utils as mod
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    utils = NetworkUtils()
    stats = await utils.ping_host("127.0.0.1", count=2)
    assert stats["is_host_reachable"] is True
    assert stats["packets_sent"] == 2
    assert stats["packets_received"] == 2
    assert stats["packet_loss_percentage"] == 0.0













