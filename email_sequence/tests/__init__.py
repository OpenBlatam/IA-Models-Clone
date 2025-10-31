"""
Test suite for Email Sequence AI System

This module contains comprehensive tests for the email sequence system.
"""

import pytest
import asyncio
from typing import AsyncGenerator
from fastapi.testclient import TestClient
from httpx import AsyncClient

from main import create_app
from core.dependencies import get_database, get_redis, get_engine
from core.database import init_database, close_database
from core.cache import init_cache, close_cache


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
async def app():
    """Create test application."""
    app = create_app()
    yield app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


@pytest.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
async def db_session():
    """Create database session for testing."""
    await init_database()
    async with get_database() as session:
        yield session
    await close_database()


@pytest.fixture
async def redis_client():
    """Create Redis client for testing."""
    await init_cache()
    from core.cache import cache_manager
    yield cache_manager
    await close_cache()


@pytest.fixture
async def engine():
    """Create email sequence engine for testing."""
    from core.dependencies import init_services
    await init_services()
    engine = await get_engine()
    yield engine
    await engine.stop()


# Test data fixtures
@pytest.fixture
def sample_sequence_data():
    """Sample sequence data for testing."""
    return {
        "name": "Test Welcome Sequence",
        "description": "A test welcome sequence",
        "target_audience": "New subscribers",
        "goals": ["Welcome", "Onboard"],
        "tone": "friendly",
        "steps": [
            {
                "step_type": "email",
                "order": 1,
                "name": "Welcome Email",
                "subject": "Welcome to our community!",
                "content": "Thank you for joining us..."
            }
        ]
    }


@pytest.fixture
def sample_subscriber_data():
    """Sample subscriber data for testing."""
    return {
        "email": "test@example.com",
        "first_name": "John",
        "last_name": "Doe",
        "company": "Test Company",
        "interests": ["technology", "business"]
    }


@pytest.fixture
def sample_template_data():
    """Sample template data for testing."""
    return {
        "name": "Welcome Template",
        "description": "A welcome email template",
        "subject": "Welcome {{first_name}}!",
        "html_content": "<h1>Welcome {{first_name}} {{last_name}}!</h1>",
        "text_content": "Welcome {{first_name}} {{last_name}}!",
        "variables": [
            {
                "name": "first_name",
                "type": "string",
                "required": True
            },
            {
                "name": "last_name", 
                "type": "string",
                "required": False
            }
        ]
    }






























