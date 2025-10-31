import asyncio
import types
import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_resolve_hostname_to_ip_unresolved(monkeypatch):
    import socket as std_socket
    import network_utils as mod

    def raise_gaierror(host: str):
        raise std_socket.gaierror()

    monkeypatch.setattr(mod.socket, "gethostbyname", raise_gaierror)

    utils = NetworkUtils()
    ip = await utils.resolve_hostname_to_ip("nonexistent.example")
    assert ip == "unresolved"


@pytest.mark.asyncio
async def test_check_host_connectivity_timeout(monkeypatch):
    async def fake_open_connection(host, port):
        await asyncio.sleep(0)  # ensure awaitable
        raise asyncio.TimeoutError()

    monkeypatch.setattr(asyncio, "open_connection", fake_open_connection)

    import network_utils as mod
    monkeypatch.setattr(mod, "socket", types.SimpleNamespace(gethostbyname=lambda h: "0.0.0.0"))

    utils = NetworkUtils(default_timeout=0.01)
    info = await utils.check_host_connectivity("example.com", 1)
    assert info.is_connection_successful is False
    assert info.error_message == "Connection timeout"


