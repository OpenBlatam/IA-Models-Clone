import asyncio
import pytest

from ..utils.u_ti_ls import measure_execution_time


@pytest.mark.asyncio
async def test_measure_execution_time_raises_on_error():
    class Boom(Exception):
        pass

    @measure_execution_time
    async def bad():
        await asyncio.sleep(0.001)
        raise Boom("x")

    with pytest.raises(Boom):
        await bad()



