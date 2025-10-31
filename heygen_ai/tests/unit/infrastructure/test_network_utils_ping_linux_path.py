import asyncio
from unittest.mock import patch

import pytest

from network_utils import NetworkUtils


@pytest.mark.asyncio
async def test_ping_host_linux_path_parsing():
    utils = NetworkUtils()

    async def fake_create_subprocess_exec(*args, **kwargs):
        class _P:
            returncode = 0

            async def communicate(self):
                return (
                    b"PING 127.0.0.1 (127.0.0.1): 56 data bytes\n64 bytes from 127.0.0.1: icmp_seq=0 time=1.23 ms\n",
                    b"",
                )

        return _P()

    with patch("platform.system", return_value="Linux"), \
         patch("asyncio.create_subprocess_exec", side_effect=fake_create_subprocess_exec):
        res = await utils.ping_host("127.0.0.1")

    assert res["is_host_reachable"] is True
    assert res["packets_sent"] == 4


