"""
Test configuration and fixtures
"""

import pytest
import asyncio
from typing import AsyncGenerator, Generator
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from httpx import AsyncClient

from ..main import create_app
from ..config.database import get_db_session
from ..models.database import Base
from ..config.settings import get_settings

# Test database URL
TEST_DATABASE_URL = "sqlite+aiosqlite:///./test.db"


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def test_engine():
    """Create test database engine."""
    engine = create_async_engine(TEST_DATABASE_URL, echo=True)
    
    # Create tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    yield engine
    
    # Cleanup
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    await engine.dispose()


@pytest.fixture
async def test_session(test_engine) -> AsyncGenerator[AsyncSession, None]:
    """Create test database session."""
    async_session = sessionmaker(
        test_engine, class_=AsyncSession, expire_on_commit=False
    )
    
    async with async_session() as session:
        yield session


@pytest.fixture
def client(test_session: AsyncSession) -> TestClient:
    """Create test client."""
    app = create_app()
    
    # Override database dependency
    def override_get_db():
        return test_session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
async def async_client(test_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Create async test client."""
    app = create_app()
    
    # Override database dependency
    def override_get_db():
        return test_session
    
    app.dependency_overrides[get_db_session] = override_get_db
    
    async with AsyncClient(app=app, base_url="http://test") as ac:
        yield ac


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "TestPassword123!",
        "full_name": "Test User",
        "bio": "Test bio"
    }


@pytest.fixture
def sample_blog_post_data():
    """Sample blog post data for testing."""
    return {
        "title": "Test Blog Post",
        "content": "This is a test blog post content with enough text to be meaningful.",
        "excerpt": "Test excerpt",
        "category": "technology",
        "tags": ["test", "blog", "fastapi"],
        "seo_title": "Test SEO Title",
        "seo_description": "Test SEO description",
        "seo_keywords": ["test", "seo", "keywords"]
    }


@pytest.fixture
def sample_comment_data():
    """Sample comment data for testing."""
    return {
        "content": "This is a test comment with meaningful content.",
        "parent_id": None
    }


@pytest.fixture
async def authenticated_user(client: TestClient, sample_user_data: dict):
    """Create and authenticate a test user."""
    # Create user
    response = client.post("/api/v1/users/", json=sample_user_data)
    assert response.status_code == 201
    user = response.json()
    
    # In a real implementation, you would get a token from login
    # For testing, we'll mock the authentication
    return user


@pytest.fixture
async def sample_blog_post(client: TestClient, authenticated_user: dict, sample_blog_post_data: dict):
    """Create a sample blog post for testing."""
    # Mock authentication for the request
    # In a real test, you would include proper authentication headers
    response = client.post("/api/v1/blog-posts/", json=sample_blog_post_data)
    assert response.status_code == 201
    return response.json()






























