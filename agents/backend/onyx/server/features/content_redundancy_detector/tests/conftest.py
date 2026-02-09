import asyncio
import pytest
from httpx import AsyncClient

# Import the FastAPI app
from agents.backend.onyx.server.features.content_redundancy_detector.app import app


@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def async_client():
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client







