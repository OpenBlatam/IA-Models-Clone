import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_ping_host_windows_path_parsing():
    utils = NetworkUtils()

    async def fake_create_subprocess_exec(*args, **kwargs):
        class _P:
            returncode = 0

            async def communicate(self):
                return (
                    b"Pinging 127.0.0.1 with 32 bytes of data:\nReply from 127.0.0.1: time=1ms\n",
                    b"",
                )

        return _P()

    with patch("platform.system", return_value="Windows"), \
         patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        res = await utils.ping_host("127.0.0.1")

    assert res["is_host_reachable"] is True
    assert res["packets_received"] == 4


