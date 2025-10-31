from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
from typing import Dict, Any, List
from datetime import datetime, timedelta
from pydantic import BaseModel
from ..redis_utils import RedisUtils
from ..redis_config import RedisConfig, get_config
from ..redis_middleware import RedisMiddleware
from ..redis_decorators import RedisDecorators
from fastapi import FastAPI, Request, Response
from starlette.testclient import TestClient
import redis
from ..redis_manager import RedisManager
from typing import Any, List, Dict, Optional
import logging
"""
Redis Tests - Onyx Integration
Tests for Redis integration in Onyx.
"""

# Test models
class UserModel(BaseModel):
    """Test user model."""
    id: str
    name: str
    email: str
    created_at: datetime = datetime.utcnow()

class TestModel(BaseModel):
    id: str
    name: str
    value: int

@pytest.fixture
def redis_config():
    """Create a test Redis configuration."""
    return RedisConfig(
        host="localhost",
        port=6379,
        db=15,  # Use a different DB for testing
        default_expire=300,
        log_level="DEBUG"
    )

@pytest.fixture
def redis_manager(redis_config) -> Any:
    """Create a Redis manager instance for testing."""
    manager = RedisManager(
        host=redis_config.host,
        port=redis_config.port,
        db=redis_config.db
    )
    yield manager
    # Cleanup after tests
    manager.clear_prefix("test")

class TestRedisUtils:
    """Tests for Redis utilities."""
    
    @pytest.fixture
    def redis_utils(self, redis_config) -> Any:
        """Create Redis utilities instance."""
        return RedisUtils(redis_config)
    
    @pytest.fixture
    def test_data(self) -> Any:
        """Create test data."""
        return {
            "user_1": UserModel(
                id="1",
                name="John Doe",
                email="john@example.com"
            ),
            "user_2": UserModel(
                id="2",
                name="Jane Smith",
                email="jane@example.com"
            )
        }
    
    def test_cache_data(self, redis_utils, test_data) -> Any:
        """Test caching data."""
        # Cache data
        redis_utils.cache_data(
            data=test_data["user_1"],
            prefix="test",
            identifier="user_1"
        )
        
        # Get cached data
        cached_data = redis_utils.get_cached_data(
            prefix="test",
            identifier="user_1",
            model_class=UserModel
        )
        
        assert cached_data is not None
        assert cached_data.id == test_data["user_1"].id
        assert cached_data.name == test_data["user_1"].name
        assert cached_data.email == test_data["user_1"].email
    
    def test_cache_batch(self, redis_utils, test_data) -> Any:
        """Test caching batch data."""
        # Cache batch data
        redis_utils.cache_batch(
            data_dict=test_data,
            prefix="test"
        )
        
        # Get cached batch data
        cached_batch = redis_utils.get_cached_batch(
            prefix="test",
            identifiers=["user_1", "user_2"],
            model_class=UserModel
        )
        
        assert len(cached_batch) == 2
        assert cached_batch["user_1"].id == test_data["user_1"].id
        assert cached_batch["user_2"].id == test_data["user_2"].id
    
    def test_delete_batch(self, redis_utils, test_data) -> Any:
        """Test deleting batch keys."""
        # Cache batch data
        redis_utils.cache_batch(
            data_dict=test_data,
            prefix="test"
        )
        
        # Delete batch keys
        redis_utils.delete_batch(
            prefix="test",
            identifiers=["user_1", "user_2"]
        )
        
        # Get cached batch data
        cached_batch = redis_utils.get_cached_batch(
            prefix="test",
            identifiers=["user_1", "user_2"]
        )
        
        assert len(cached_batch) == 0
    
    def test_scan_keys(self, redis_utils, test_data) -> Any:
        """Test scanning keys."""
        # Cache data
        redis_utils.cache_data(
            data=test_data["user_1"],
            prefix="test",
            identifier="user_1"
        )
        
        # Scan keys
        keys = redis_utils.scan_keys(
            prefix="test",
            pattern="user_*"
        )
        
        assert len(keys) == 1
        assert keys[0].endswith("user_1")
    
    def test_get_memory_usage(self, redis_utils) -> Optional[Dict[str, Any]]:
        """Test getting memory usage."""
        memory_usage = redis_utils.get_memory_usage()
        
        assert isinstance(memory_usage, dict)
        assert "used_memory" in memory_usage
        assert "used_memory_peak" in memory_usage
    
    def test_get_stats(self, redis_utils) -> Optional[Dict[str, Any]]:
        """Test getting Redis stats."""
        stats = redis_utils.get_stats()
        
        assert isinstance(stats, dict)
        assert "clients" in stats
        assert "memory" in stats
        assert "stats" in stats

class TestRedisMiddleware:
    """Tests for Redis middleware."""
    
    @pytest.fixture
    def app(self) -> Any:
        """Create FastAPI app."""
        app = FastAPI()
        
        # Add Redis middleware
        app.add_middleware(
            RedisMiddleware,
            config={
                "cache_ttl": 3600,
                "exclude_paths": ["/admin"],
                "include_paths": ["/api"],
                "cache_headers": True
            }
        )
        
        # Add test routes
        @app.get("/api/test")
        async def test_route():
            
    """test_route function."""
return {"message": "test"}
        
        @app.get("/admin/test")
        async def admin_route():
            
    """admin_route function."""
return {"message": "admin"}
        
        return app
    
    @pytest.fixture
    def client(self, app) -> Any:
        """Create test client."""
        return TestClient(app)
    
    def test_cached_response(self, client) -> Any:
        """Test cached response."""
        # First request
        response1 = client.get("/api/test")
        assert response1.status_code == 200
        assert response1.json() == {"message": "test"}
        assert "X-Cache" not in response1.headers
        
        # Second request (should be cached)
        response2 = client.get("/api/test")
        assert response2.status_code == 200
        assert response2.json() == {"message": "test"}
        assert response2.headers["X-Cache"] == "HIT"
    
    def test_excluded_path(self, client) -> Any:
        """Test excluded path."""
        # First request
        response1 = client.get("/admin/test")
        assert response1.status_code == 200
        assert response1.json() == {"message": "admin"}
        assert "X-Cache" not in response1.headers
        
        # Second request (should not be cached)
        response2 = client.get("/admin/test")
        assert response2.status_code == 200
        assert response2.json() == {"message": "admin"}
        assert "X-Cache" not in response2.headers

class TestRedisDecorators:
    """Tests for Redis decorators."""
    
    @pytest.fixture
    def redis_decorators(self, redis_config) -> Any:
        """Create Redis decorators instance."""
        return RedisDecorators(redis_config)
    
    @pytest.fixture
    def test_data(self) -> Any:
        """Create test data."""
        return {
            "user_1": UserModel(
                id="1",
                name="John Doe",
                email="john@example.com"
            ),
            "user_2": UserModel(
                id="2",
                name="Jane Smith",
                email="jane@example.com"
            )
        }
    
    @pytest.mark.asyncio
    async def test_cache_decorator(self, redis_decorators, test_data) -> Any:
        """Test cache decorator."""
        @redis_decorators.cache(
            prefix="test",
            ttl=3600
        )
        async def get_user_data(user_id: str) -> Dict[str, Any]:
            return test_data[user_id].model_dump()
        
        # First call
        data1 = await get_user_data("user_1")
        assert data1["id"] == test_data["user_1"].id
        
        # Second call (should be cached)
        data2 = await get_user_data("user_1")
        assert data2["id"] == test_data["user_1"].id
    
    @pytest.mark.asyncio
    async def test_cache_model_decorator(self, redis_decorators, test_data) -> Any:
        """Test cache model decorator."""
        @redis_decorators.cache_model(
            prefix="test",
            ttl=3600
        )
        async def get_user_model(user_id: str) -> UserModel:
            return test_data[user_id]
        
        # First call
        model1 = await get_user_model("user_1")
        assert model1.id == test_data["user_1"].id
        
        # Second call (should be cached)
        model2 = await get_user_model("user_1")
        assert model2.id == test_data["user_1"].id
    
    @pytest.mark.asyncio
    async def test_cache_batch_decorator(self, redis_decorators, test_data) -> Any:
        """Test cache batch decorator."""
        @redis_decorators.cache_batch(
            prefix="test",
            ttl=3600
        )
        async def get_batch_data(user_ids: List[str]) -> Dict[str, UserModel]:
            return {user_id: test_data[user_id] for user_id in user_ids}
        
        # First call
        batch1 = await get_batch_data(["user_1", "user_2"])
        assert len(batch1) == 2
        assert batch1["user_1"].id == test_data["user_1"].id
        
        # Second call (should be cached)
        batch2 = await get_batch_data(["user_1", "user_2"])
        assert len(batch2) == 2
        assert batch2["user_1"].id == test_data["user_1"].id
    
    @pytest.mark.asyncio
    async def test_invalidate_decorator(self, redis_decorators, test_data) -> bool:
        """Test invalidate decorator."""
        @redis_decorators.cache(
            prefix="test",
            ttl=3600
        )
        async def get_user_data(user_id: str) -> Dict[str, Any]:
            return test_data[user_id].model_dump()
        
        @redis_decorators.invalidate(
            prefix="test"
        )
        async def update_user_data(user_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
            return data
        
        # Cache data
        await get_user_data("user_1")
        
        # Update data
        updated_data = {"name": "Updated Name"}
        await update_user_data("user_1", updated_data)
        
        # Get updated data
        data = await get_user_data("user_1")
        assert data["name"] == "Updated Name"

# Test Redis Configuration
def test_redis_config_defaults():
    """Test default Redis configuration values."""
    config = RedisConfig()
    assert config.host == "localhost"
    assert config.port == 6379
    assert config.db == 0
    assert config.password == ""
    assert config.max_connections == 10
    assert config.default_expire == 3600

def test_redis_config_custom():
    """Test custom Redis configuration values."""
    config = RedisConfig(
        host="test-host",
        port=6380,
        db=1,
        password="test-password",
        max_connections=20,
        default_expire=1800
    )
    assert config.host == "test-host"
    assert config.port == 6380
    assert config.db == 1
    assert config.password == "test-password"
    assert config.max_connections == 20
    assert config.default_expire == 1800

def test_get_config():
    """Test getting configuration for different environments."""
    dev_config = get_config("development")
    assert dev_config.host == "localhost"
    assert dev_config.port == 6379
    
    test_config = get_config("testing")
    assert test_config.db == 15
    assert test_config.default_expire == 300
    
    prod_config = get_config("production")
    assert prod_config.host == "redis.production"
    assert prod_config.max_connections == 50

# Test Redis Manager
def test_cache_model(redis_manager) -> Any:
    """Test caching and retrieving a model."""
    model = TestModel(id="1", name="Test", value=42)
    
    # Cache the model
    redis_manager.cache_model(
        model=model,
        prefix="test",
        identifier="model_1",
        expire=300
    )
    
    # Retrieve the model
    cached_model = redis_manager.get_cached_model(
        model_class=TestModel,
        prefix="test",
        identifier="model_1"
    )
    
    assert cached_model is not None
    assert cached_model.id == model.id
    assert cached_model.name == model.name
    assert cached_model.value == model.value

def test_cache_context(redis_manager) -> Any:
    """Test caching and retrieving context data."""
    context = {"user_id": "123", "action": "test"}
    
    # Cache the context
    redis_manager.cache_context(
        context=context,
        prefix="test",
        identifier="context_1",
        expire=300
    )
    
    # Retrieve the context
    cached_context = redis_manager.get_cached_context(
        prefix="test",
        identifier="context_1"
    )
    
    assert cached_context is not None
    assert cached_context["user_id"] == context["user_id"]
    assert cached_context["action"] == context["action"]

def test_cache_prompt(redis_manager) -> Any:
    """Test caching and retrieving a prompt."""
    prompt = "Test prompt"
    
    # Cache the prompt
    redis_manager.cache_prompt(
        prompt=prompt,
        prefix="test",
        identifier="prompt_1",
        expire=300
    )
    
    # Retrieve the prompt
    cached_prompt = redis_manager.get_cached_prompt(
        prefix="test",
        identifier="prompt_1"
    )
    
    assert cached_prompt == prompt

def test_cache_metrics(redis_manager) -> Any:
    """Test caching and retrieving metrics."""
    metrics = {"clicks": 100, "views": 1000}
    
    # Cache the metrics
    redis_manager.cache_metrics(
        metrics=metrics,
        prefix="test",
        identifier="metrics_1",
        expire=300
    )
    
    # Retrieve the metrics
    cached_metrics = redis_manager.get_cached_metrics(
        prefix="test",
        identifier="metrics_1"
    )
    
    assert cached_metrics is not None
    assert cached_metrics["clicks"] == metrics["clicks"]
    assert cached_metrics["views"] == metrics["views"]

def test_increment_counter(redis_manager) -> Any:
    """Test counter operations."""
    # Increment counter
    value = redis_manager.increment_counter(
        prefix="test",
        identifier="counter_1",
        amount=1
    )
    assert value == 1
    
    # Get counter value
    counter_value = redis_manager.get_counter(
        prefix="test",
        identifier="counter_1"
    )
    assert counter_value == 1

def test_set_operations(redis_manager) -> Any:
    """Test set operations."""
    # Add to set
    redis_manager.add_to_set(
        prefix="test",
        identifier="set_1",
        value="value1"
    )
    
    # Get set members
    members = redis_manager.get_set_members(
        prefix="test",
        identifier="set_1"
    )
    assert "value1" in members

def test_delete_key(redis_manager) -> Any:
    """Test key deletion."""
    # Cache something
    redis_manager.cache_model(
        model=TestModel(id="1", name="Test", value=42),
        prefix="test",
        identifier="model_1"
    )
    
    # Delete the key
    redis_manager.delete_key(
        prefix="test",
        identifier="model_1"
    )
    
    # Try to retrieve the deleted key
    cached_model = redis_manager.get_cached_model(
        model_class=TestModel,
        prefix="test",
        identifier="model_1"
    )
    assert cached_model is None

def test_clear_prefix(redis_manager) -> Any:
    """Test clearing all keys with a prefix."""
    # Cache multiple items
    redis_manager.cache_model(
        model=TestModel(id="1", name="Test1", value=42),
        prefix="test",
        identifier="model_1"
    )
    redis_manager.cache_model(
        model=TestModel(id="2", name="Test2", value=43),
        prefix="test",
        identifier="model_2"
    )
    
    # Clear all test keys
    redis_manager.clear_prefix("test")
    
    # Verify keys are deleted
    assert redis_manager.get_cached_model(
        model_class=TestModel,
        prefix="test",
        identifier="model_1"
    ) is None
    assert redis_manager.get_cached_model(
        model_class=TestModel,
        prefix="test",
        identifier="model_2"
    ) is None 