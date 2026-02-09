from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import pytest
import redis
from datetime import datetime
from pydantic import BaseModel
from ..redis_config import RedisConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Test Configuration - Onyx Integration
Shared test configuration and fixtures.
"""

# Test models
class UserModel(BaseModel):
    """Test user model."""
    id: str
    name: str
    email: str
    created_at: datetime = datetime.utcnow()

class ProductModel(BaseModel):
    """Test product model."""
    id: str
    name: str
    price: float
    category: str
    created_at: datetime = datetime.utcnow()

# Redis configuration
@pytest.fixture(scope="session")
def redis_config():
    """Create Redis configuration."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=15,  # Use a separate database for testing
        default_expire=3600
    )

@pytest.fixture(scope="session", autouse=True)
def check_redis(redis_config) -> Any:
    """Check Redis connection before tests."""
    try:
        redis_client = redis.Redis(
            host=redis_config.host,
            port=redis_config.port,
            db=redis_config.db
        )
        redis_client.ping()
        yield
    except redis.ConnectionError:
        pytest.skip("Redis server is not running")
    finally:
        redis_client.close()

# Test data fixtures
@pytest.fixture
def test_users():
    """Create test users."""
    return [
        UserModel(
            id="1",
            name="John Doe",
            email="john@example.com"
        ),
        UserModel(
            id="2",
            name="Jane Smith",
            email="jane@example.com"
        )
    ]

@pytest.fixture
def test_products():
    """Create test products."""
    return [
        ProductModel(
            id="1",
            name="Product 1",
            price=10.99,
            category="Electronics"
        ),
        ProductModel(
            id="2",
            name="Product 2",
            price=20.99,
            category="Books"
        )
    ]

@pytest.fixture
def test_data(test_users) -> Any:
    """Create test data dictionary."""
    return {
        "user_1": test_users[0],
        "user_2": test_users[1]
    } 