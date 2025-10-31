from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
import pytest
from datetime import datetime
from pydantic import BaseModel
import redis
from ..redis_indexer import RedisIndexer
from ..redis_config import RedisConfig
from typing import Any, List, Dict, Optional
import logging
import asyncio
"""
Redis Indexer Tests - Onyx Integration
Tests for Redis indexer in Onyx.
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

class TestRedisIndexer:
    """Tests for Redis indexer."""
    
    @pytest.fixture(autouse=True)
    def setup_redis(self) -> Any:
        """Check Redis connection before tests."""
        try:
            redis_client = redis.Redis(host="localhost", port=6379, db=15)
            redis_client.ping()
            yield
        except redis.ConnectionError:
            pytest.skip("Redis server is not running")
        finally:
            redis_client.close()
    
    @pytest.fixture
    def redis_indexer(self, redis_config) -> Any:
        """Create Redis indexer instance."""
        return RedisIndexer(redis_config)
    
    @pytest.fixture
    def test_users(self) -> Any:
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
    def test_products(self) -> Any:
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
    
    def test_index_model(self, redis_indexer, test_users) -> Any:
        """Test indexing a single model."""
        # Index user
        redis_indexer.index_model(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Find user by email
        found_user = redis_indexer.find_by_index(
            model_name="user",
            field="email",
            value="john@example.com",
            model_class=UserModel
        )
        
        assert found_user is not None
        assert found_user.id == test_users[0].id
        assert found_user.name == test_users[0].name
        assert found_user.email == test_users[0].email
    
    def test_index_batch(self, redis_indexer, test_users) -> Any:
        """Test indexing multiple models."""
        # Index users
        redis_indexer.index_batch(
            models=test_users,
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Find users by email
        found_users = redis_indexer.find_batch_by_index(
            model_name="user",
            field="email",
            values=["john@example.com", "jane@example.com"],
            model_class=UserModel
        )
        
        assert len(found_users) == 2
        assert found_users["john@example.com"].id == test_users[0].id
        assert found_users["jane@example.com"].id == test_users[1].id
    
    def test_remove_index(self, redis_indexer, test_users) -> Any:
        """Test removing an index."""
        # Index user
        redis_indexer.index_model(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Remove index
        redis_indexer.remove_index(
            model_name="user",
            field="email",
            value="john@example.com"
        )
        
        # Try to find user
        found_user = redis_indexer.find_by_index(
            model_name="user",
            field="email",
            value="john@example.com",
            model_class=UserModel
        )
        
        assert found_user is None
    
    def test_remove_model(self, redis_indexer, test_users) -> Any:
        """Test removing a model and its indexes."""
        # Index user
        redis_indexer.index_model(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Remove model
        redis_indexer.remove_model(
            model_name="user",
            model_id="1"
        )
        
        # Try to find user by email
        found_user = redis_indexer.find_by_index(
            model_name="user",
            field="email",
            value="john@example.com",
            model_class=UserModel
        )
        
        assert found_user is None
    
    def test_update_index(self, redis_indexer, test_users) -> Any:
        """Test updating a model's indexes."""
        # Index user
        redis_indexer.index_model(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Update user
        test_users[0].name = "John Updated"
        redis_indexer.update_index(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Find updated user
        found_user = redis_indexer.find_by_index(
            model_name="user",
            field="email",
            value="john@example.com",
            model_class=UserModel
        )
        
        assert found_user is not None
        assert found_user.name == "John Updated"
    
    def test_get_index_stats(self, redis_indexer, test_users, test_products) -> Optional[Dict[str, Any]]:
        """Test getting index statistics."""
        # Index users and products
        redis_indexer.index_batch(
            models=test_users,
            model_name="user",
            index_fields=["id", "email"]
        )
        redis_indexer.index_batch(
            models=test_products,
            model_name="product",
            index_fields=["id", "category"]
        )
        
        # Get stats
        stats = redis_indexer.get_index_stats()
        
        assert isinstance(stats, dict)
        assert "user" in stats
        assert "product" in stats
        assert stats["user"]["count"] == 2
        assert stats["product"]["count"] == 2
    
    def test_multiple_index_fields(self, redis_indexer, test_products) -> Any:
        """Test indexing multiple fields."""
        # Index product with multiple fields
        redis_indexer.index_model(
            model=test_products[0],
            model_name="product",
            index_fields=["id", "category", "price"]
        )
        
        # Find by category
        found_by_category = redis_indexer.find_by_index(
            model_name="product",
            field="category",
            value="Electronics",
            model_class=ProductModel
        )
        assert found_by_category is not None
        assert found_by_category.id == test_products[0].id
        
        # Find by price
        found_by_price = redis_indexer.find_by_index(
            model_name="product",
            field="price",
            value=10.99,
            model_class=ProductModel
        )
        assert found_by_price is not None
        assert found_by_price.id == test_products[0].id
    
    def test_none_values(self, redis_indexer) -> Any:
        """Test handling of None values."""
        # Create model with None value
        model = UserModel(
            id="3",
            name="Test User",
            email=None
        )
        
        # Index model
        redis_indexer.index_model(
            model=model,
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Try to find by None email
        found_user = redis_indexer.find_by_index(
            model_name="user",
            field="email",
            value=None,
            model_class=UserModel
        )
        
        assert found_user is None
    
    def test_invalid_model_class(self, redis_indexer, test_users) -> Any:
        """Test handling of invalid model class."""
        # Index user
        redis_indexer.index_model(
            model=test_users[0],
            model_name="user",
            index_fields=["id", "email"]
        )
        
        # Try to find with wrong model class
        with pytest.raises(ValueError):
            redis_indexer.find_by_index(
                model_name="user",
                field="email",
                value="john@example.com",
                model_class=ProductModel
            ) 