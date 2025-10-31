import asyncio
import pytest

from ..utils.u_ti_ls import measure_execution_time


@pytest.mark.asyncio
async def test_measure_execution_time_decorator_wraps_and_measures():
    @measure_execution_time
    async def do_work(x):
        await asyncio.sleep(0.01)
        return x * 2

    wrapped = await do_work(5)
    # Decorator returns dict with result and execution_time
    assert wrapped["result"] == 10
    assert wrapped["execution_time"] >= 0.0



