import asyncio
import pytest

from ..utils.u_ti_ls import timeout_operation


@pytest.mark.asyncio
async def test_timeout_operation_propagates_non_timeout_errors():
    class Boom(Exception):
        pass

    async def bad():
        await asyncio.sleep(0.001)
        raise Boom("fail")

    with pytest.raises(Boom):
        await timeout_operation(operation=bad, timeout_seconds=1.0)



