import asyncio
from typing import Callable
from .dispatcher import get_webhook_dispatcher


async def start_webhook_workers(handler: Callable):
    dispatcher = await get_webhook_dispatcher()
    await dispatcher.start(handler)


async def stop_webhook_workers():
    dispatcher = await get_webhook_dispatcher()
    await dispatcher.stop()







