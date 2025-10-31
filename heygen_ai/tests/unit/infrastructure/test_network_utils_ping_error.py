import asyncio
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_ping_host_process_failure(monkeypatch):
    async def fake_create_subprocess_exec(*args, **kwargs):
        class P:
            returncode = 1
            async def communicate(self):
                return (b"", b"err")
        return P()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)

    u = NetworkUtils()
    stats = await u.ping_host("bad.host", count=1)
    assert stats["is_host_reachable"] is False
    assert "error_message" in stats or stats["packets_received"] == 0













