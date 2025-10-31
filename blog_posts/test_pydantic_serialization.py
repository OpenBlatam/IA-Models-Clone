from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_CONNECTIONS = 1000

# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

# Constants
BUFFER_SIZE = 1024

import pytest
import asyncio
import time
import json
import pickle
from typing import Dict, List, Tuple, Any, Optional
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import os
from pydantic import BaseModel, Field, ValidationError
from datetime import datetime, timedelta
from pydantic_serialization import (
    from caching_system import CacheManager, CacheConfig, create_cache_manager
        import gzip
from typing import Any, List, Dict, Optional
import logging
"""
🧪 COMPREHENSIVE PYDANTIC SERIALIZATION TESTS
============================================

Test suite for the optimized Pydantic serialization system covering:
- Serialization formats (JSON, Pickle, MessagePack, orjson)
- Compression and decompression
- Caching mechanisms
- Performance profiling
- Error handling
- Integration with caching system
- Model validation
- Performance benchmarking

Features:
- Unit tests for each serializer
- Integration tests for caching
- Performance testing
- Error scenario testing
- Mock testing for optional dependencies
- Async/await testing
- Configuration testing
"""



    SerializationConfig, SerializationFormat, CompressionLevel,
    SerializationUtils, BaseSerializer, JSONSerializer, PickleSerializer,
    PydanticModelSerializer, CachedSerializationManager, SerializationProfiler,
    OptimizedSerializationManager, serialized, deserialized,
    create_serialization_config, create_serialization_manager,
    benchmark_serialization
)

# Import caching system for integration tests
try:
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def serialization_config():
    """Create serialization configuration for testing."""
    return SerializationConfig(
        enable_caching=True,
        enable_compression=True,
        enable_profiling=True,
        cache_ttl=3600,
        cache_max_size=100
    )

@pytest.fixture
def basic_config():
    """Create basic configuration without caching."""
    return SerializationConfig(
        enable_caching=False,
        enable_compression=False,
        enable_profiling=False
    )

@pytest.fixture
def sample_models():
    """Create sample Pydantic models for testing."""
    
    class User(BaseModel):
        id: int
        name: str
        email: str
        created_at: datetime = Field(default_factory=datetime.now)
        is_active: bool = True
    
    class Post(BaseModel):
        id: int
        title: str
        content: str
        author_id: int
        created_at: datetime = Field(default_factory=datetime.now)
        tags: List[str] = []
    
    class ComplexModel(BaseModel):
        id: int
        data: Dict[str, Any]
        nested_list: List[Dict[str, Any]]
        timestamp: datetime = Field(default_factory=datetime.now)
    
    return {
        "User": User,
        "Post": Post,
        "ComplexModel": ComplexModel
    }

@pytest.fixture
def sample_data(sample_models) -> Any:
    """Create sample data for testing."""
    User = sample_models["User"]
    Post = sample_models["Post"]
    ComplexModel = sample_models["ComplexModel"]
    
    return {
        "user": User(
            id=1,
            name="John Doe",
            email="john@example.com",
            created_at=datetime(2024, 1, 15, 10, 30, 0)
        ),
        "post": Post(
            id=1,
            title="Hello World",
            content="This is a test post with some content",
            author_id=1,
            tags=["test", "example"]
        ),
        "complex": ComplexModel(
            id=1,
            data={"key1": "value1", "key2": 42, "key3": True},
            nested_list=[
                {"nested_key1": "nested_value1"},
                {"nested_key2": "nested_value2"}
            ]
        )
    }

# ============================================================================
# CONFIGURATION TESTS
# ============================================================================

class TestSerializationConfig:
    """Test serialization configuration."""
    
    def test_default_config(self) -> Any:
        """Test default configuration values."""
        config = SerializationConfig()
        
        assert config.default_format == SerializationFormat.JSON
        assert config.fallback_format == SerializationFormat.PICKLE
        assert config.enable_compression is True
        assert config.compression_level == CompressionLevel.BALANCED
        assert config.compression_threshold == 1024
        assert config.enable_caching is True
        assert config.cache_ttl == 3600
        assert config.enable_validation is True
        assert config.enable_profiling is True
    
    def test_custom_config(self) -> Any:
        """Test custom configuration values."""
        config = SerializationConfig(
            default_format=SerializationFormat.PICKLE,
            enable_compression=False,
            enable_caching=False,
            cache_ttl=1800,
            enable_profiling=False
        )
        
        assert config.default_format == SerializationFormat.PICKLE
        assert config.enable_compression is False
        assert config.enable_caching is False
        assert config.cache_ttl == 1800
        assert config.enable_profiling is False
    
    def test_config_validation(self) -> Any:
        """Test configuration validation."""
        # Should not raise validation error
        config = SerializationConfig()
        config.cache_ttl = 7200
        assert config.cache_ttl == 7200

# ============================================================================
# SERIALIZATION UTILITIES TESTS
# ============================================================================

class TestSerializationUtils:
    """Test serialization utilities."""
    
    def test_get_compression_level(self) -> Optional[Dict[str, Any]]:
        """Test compression level mapping."""
        assert SerializationUtils.get_compression_level(CompressionLevel.NONE) == 0
        assert SerializationUtils.get_compression_level(CompressionLevel.FAST) == 1
        assert SerializationUtils.get_compression_level(CompressionLevel.BALANCED) == 6
        assert SerializationUtils.get_compression_level(CompressionLevel.MAX) == 9
    
    def test_should_compress(self, serialization_config) -> Any:
        """Test compression decision logic."""
        # Should compress large data
        assert SerializationUtils.should_compress(2048, serialization_config) is True
        
        # Should not compress small data
        assert SerializationUtils.should_compress(512, serialization_config) is False
        
        # Should not compress when compression is disabled
        config_no_compression = SerializationConfig(enable_compression=False)
        assert SerializationUtils.should_compress(2048, config_no_compression) is False
        
        # Should not compress when level is NONE
        config_no_level = SerializationConfig(compression_level=CompressionLevel.NONE)
        assert SerializationUtils.should_compress(2048, config_no_level) is False
    
    def test_compress_decompress(self, serialization_config) -> Any:
        """Test compression and decompression."""
        original_data = b"x" * 2000  # Large enough to trigger compression
        
        # Compress data
        compressed = SerializationUtils.compress_data(original_data, serialization_config)
        assert compressed.startswith(b"gzip:")
        
        # Decompress data
        decompressed = SerializationUtils.decompress_data(compressed)
        assert decompressed == original_data
    
    def test_compress_small_data(self, serialization_config) -> Any:
        """Test that small data is not compressed."""
        small_data = b"small data"
        
        # Should not compress small data
        result = SerializationUtils.compress_data(small_data, serialization_config)
        assert result == small_data
        assert not result.startswith(b"gzip:")
    
    def test_decompress_uncompressed_data(self) -> Any:
        """Test decompression of uncompressed data."""
        uncompressed_data = b"uncompressed data"
        
        # Should return original data
        result = SerializationUtils.decompress_data(uncompressed_data)
        assert result == uncompressed_data
    
    def test_generate_hash(self) -> Any:
        """Test hash generation."""
        # String hash
        str_hash = SerializationUtils.generate_hash("test string")
        assert len(str_hash) == 32  # MD5 hash length
        assert str_hash.isalnum()
        
        # Bytes hash
        bytes_hash = SerializationUtils.generate_hash(b"test bytes")
        assert len(bytes_hash) == 32
        assert bytes_hash.isalnum()
        
        # Object hash
        obj_hash = SerializationUtils.generate_hash({"key": "value"})
        assert len(obj_hash) == 32
        assert obj_hash.isalnum()
        
        # Same input should produce same hash
        assert SerializationUtils.generate_hash("test") == SerializationUtils.generate_hash("test")

# ============================================================================
# BASE SERIALIZER TESTS
# ============================================================================

class TestBaseSerializer:
    """Test base serializer functionality."""
    
    def test_base_serializer_initialization(self, serialization_config) -> Any:
        """Test base serializer initialization."""
        serializer = BaseSerializer(serialization_config)
        
        assert serializer.config == serialization_config
        assert serializer.stats["serializations"] == 0
        assert serializer.stats["deserializations"] == 0
        assert serializer.stats["errors"] == 0
        assert serializer.stats["total_time"] == 0.0
    
    def test_base_serializer_stats(self, serialization_config) -> Any:
        """Test base serializer statistics."""
        serializer = BaseSerializer(serialization_config)
        
        # Initial stats
        stats = serializer.get_stats()
        assert stats["serializations"] == 0
        assert stats["deserializations"] == 0
        assert stats["errors"] == 0
        assert stats["avg_time"] == 0
        assert stats["error_rate"] == 0
        
        # Update stats
        serializer.stats["serializations"] = 10
        serializer.stats["deserializations"] = 5
        serializer.stats["errors"] = 1
        serializer.stats["total_time"] = 1.5
        
        # Check updated stats
        stats = serializer.get_stats()
        assert stats["serializations"] == 10
        assert stats["deserializations"] == 5
        assert stats["errors"] == 1
        assert stats["total_time"] == 1.5
        assert stats["avg_time"] == 0.1  # 1.5 / 15
        assert stats["error_rate"] == 0.06666666666666667  # 1 / 15

# ============================================================================
# JSON SERIALIZER TESTS
# ============================================================================

class TestJSONSerializer:
    """Test JSON serializer functionality."""
    
    def test_json_serializer_initialization(self, serialization_config) -> Any:
        """Test JSON serializer initialization."""
        serializer = JSONSerializer(serialization_config)
        
        assert serializer.config == serialization_config
        assert isinstance(serializer, BaseSerializer)
    
    def test_json_serialize_deserialize(self, serialization_config, sample_data) -> Any:
        """Test JSON serialization and deserialization."""
        serializer = JSONSerializer(serialization_config)
        
        # Test with different data types
        test_data = {
            "string": "test string",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "null": None
        }
        
        # Serialize
        serialized = serializer.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized == test_data
    
    def test_json_serialize_pydantic_model(self, serialization_config, sample_data) -> Any:
        """Test JSON serialization of Pydantic models."""
        serializer = JSONSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize Pydantic model
        serialized = serializer.serialize(user.model_dump())
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized["id"] == user.id
        assert deserialized["name"] == user.name
        assert deserialized["email"] == user.email
    
    def test_json_serialize_with_compression(self, serialization_config) -> Any:
        """Test JSON serialization with compression."""
        serializer = JSONSerializer(serialization_config)
        
        # Large data to trigger compression
        large_data = {"data": "x" * 2000}
        
        # Serialize
        serialized = serializer.serialize(large_data)
        assert serialized.startswith(b"gzip:")
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized == large_data
    
    def test_json_serialize_error_handling(self, serialization_config) -> Any:
        """Test JSON serialization error handling."""
        serializer = JSONSerializer(serialization_config)
        
        # Test with non-serializable object
        non_serializable = {"data": lambda x: x}
        
        with pytest.raises(Exception):
            serializer.serialize(non_serializable)
        
        # Check error stats
        assert serializer.stats["errors"] == 1
    
    def test_json_deserialize_error_handling(self, serialization_config) -> Any:
        """Test JSON deserialization error handling."""
        serializer = JSONSerializer(serialization_config)
        
        # Test with invalid JSON
        invalid_data = b"invalid json data"
        
        with pytest.raises(Exception):
            serializer.deserialize(invalid_data)
        
        # Check error stats
        assert serializer.stats["errors"] == 1

# ============================================================================
# PICKLE SERIALIZER TESTS
# ============================================================================

class TestPickleSerializer:
    """Test Pickle serializer functionality."""
    
    def test_pickle_serializer_initialization(self, serialization_config) -> Any:
        """Test Pickle serializer initialization."""
        serializer = PickleSerializer(serialization_config)
        
        assert serializer.config == serialization_config
        assert isinstance(serializer, BaseSerializer)
    
    def test_pickle_serialize_deserialize(self, serialization_config, sample_data) -> Any:
        """Test Pickle serialization and deserialization."""
        serializer = PickleSerializer(serialization_config)
        
        # Test with complex data
        test_data = {
            "string": "test string",
            "number": 42,
            "boolean": True,
            "list": [1, 2, 3],
            "dict": {"key": "value"},
            "set": {1, 2, 3},
            "tuple": (1, 2, 3),
            "datetime": datetime.now()
        }
        
        # Serialize
        serialized = serializer.serialize(test_data)
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized["string"] == test_data["string"]
        assert deserialized["number"] == test_data["number"]
        assert deserialized["boolean"] == test_data["boolean"]
        assert deserialized["list"] == test_data["list"]
        assert deserialized["dict"] == test_data["dict"]
        assert deserialized["set"] == test_data["set"]
        assert deserialized["tuple"] == test_data["tuple"]
    
    def test_pickle_serialize_pydantic_model(self, serialization_config, sample_data) -> Any:
        """Test Pickle serialization of Pydantic models."""
        serializer = PickleSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize Pydantic model
        serialized = serializer.serialize(user.model_dump())
        assert isinstance(serialized, bytes)
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized["id"] == user.id
        assert deserialized["name"] == user.name
        assert deserialized["email"] == user.email
    
    def test_pickle_serialize_with_compression(self, serialization_config) -> Any:
        """Test Pickle serialization with compression."""
        serializer = PickleSerializer(serialization_config)
        
        # Large data to trigger compression
        large_data = {"data": "x" * 2000}
        
        # Serialize
        serialized = serializer.serialize(large_data)
        assert serialized.startswith(b"gzip:")
        
        # Deserialize
        deserialized = serializer.deserialize(serialized)
        assert deserialized == large_data

# ============================================================================
# PYDANTIC MODEL SERIALIZER TESTS
# ============================================================================

class TestPydanticModelSerializer:
    """Test Pydantic model serializer functionality."""
    
    def test_model_serializer_initialization(self, serialization_config) -> Any:
        """Test model serializer initialization."""
        serializer = PydanticModelSerializer(serialization_config)
        
        assert serializer.config == serialization_config
        assert SerializationFormat.JSON in serializer.serializers
        assert SerializationFormat.PICKLE in serializer.serializers
        assert serializer.stats["model_serializations"] == 0
        assert serializer.stats["model_deserializations"] == 0
    
    def test_serialize_model_json(self, serialization_config, sample_data) -> Any:
        """Test model serialization with JSON format."""
        serializer = PydanticModelSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize model
        serialized = serializer.serialize_model(user, SerializationFormat.JSON)
        assert isinstance(serialized, bytes)
        
        # Deserialize model
        deserialized = serializer.deserialize_model(serialized, type(user), SerializationFormat.JSON)
        assert isinstance(deserialized, type(user))
        assert deserialized.id == user.id
        assert deserialized.name == user.name
        assert deserialized.email == user.email
    
    def test_serialize_model_pickle(self, serialization_config, sample_data) -> Any:
        """Test model serialization with Pickle format."""
        serializer = PydanticModelSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize model
        serialized = serializer.serialize_model(user, SerializationFormat.PICKLE)
        assert isinstance(serialized, bytes)
        
        # Deserialize model
        deserialized = serializer.deserialize_model(serialized, type(user), SerializationFormat.PICKLE)
        assert isinstance(deserialized, type(user))
        assert deserialized.id == user.id
        assert deserialized.name == user.name
        assert deserialized.email == user.email
    
    def test_serialize_model_default_format(self, serialization_config, sample_data) -> Any:
        """Test model serialization with default format."""
        serializer = PydanticModelSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize model (should use default JSON format)
        serialized = serializer.serialize_model(user)
        assert isinstance(serialized, bytes)
        
        # Deserialize model
        deserialized = serializer.deserialize_model(serialized, type(user))
        assert isinstance(deserialized, type(user))
        assert deserialized.id == user.id
    
    def test_serialize_model_fallback_format(self, serialization_config, sample_data) -> Any:
        """Test model serialization with fallback format."""
        serializer = PydanticModelSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Try to serialize with non-existent format (should fallback to PICKLE)
        serialized = serializer.serialize_model(user, "non_existent_format")
        assert isinstance(serialized, bytes)
        
        # Deserialize model
        deserialized = serializer.deserialize_model(serialized, type(user), "non_existent_format")
        assert isinstance(deserialized, type(user))
        assert deserialized.id == user.id
    
    def test_validation_caching(self, serialization_config, sample_data) -> Any:
        """Test validation result caching."""
        config = SerializationConfig(cache_validation=True)
        serializer = PydanticModelSerializer(config)
        
        user = sample_data["user"]
        
        # Serialize and deserialize twice
        serialized = serializer.serialize_model(user)
        deserialized1 = serializer.deserialize_model(serialized, type(user))
        deserialized2 = serializer.deserialize_model(serialized, type(user))
        
        # Both should be the same
        assert deserialized1.id == deserialized2.id
        assert deserialized1.name == deserialized2.name
        
        # Check validation cache hits
        assert serializer.stats["validation_cache_hits"] > 0
    
    def test_validation_error_handling(self, serialization_config) -> Any:
        """Test validation error handling."""
        serializer = PydanticModelSerializer(serialization_config)
        
        # Create invalid data
        invalid_data = b'{"id": "not_an_integer", "name": "test", "email": "test@example.com"}'
        
        class TestModel(BaseModel):
            id: int
            name: str
            email: str
        
        # Should raise validation error
        with pytest.raises(ValidationError):
            serializer.deserialize_model(invalid_data, TestModel)
        
        # Check error stats
        assert serializer.stats["errors"] == 1
    
    def test_get_stats(self, serialization_config, sample_data) -> Optional[Dict[str, Any]]:
        """Test model serializer statistics."""
        serializer = PydanticModelSerializer(serialization_config)
        
        user = sample_data["user"]
        
        # Perform some operations
        serialized = serializer.serialize_model(user)
        deserialized = serializer.deserialize_model(serialized, type(user))
        
        # Get stats
        stats = serializer.get_stats()
        
        assert stats["model_serializations"] == 1
        assert stats["model_deserializations"] == 1
        assert stats["validations"] == 1
        assert stats["total_time"] > 0
        assert "validation_cache_hit_rate" in stats

# ============================================================================
# CACHED SERIALIZATION MANAGER TESTS
# ============================================================================

class TestCachedSerializationManager:
    """Test cached serialization manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, serialization_config) -> Any:
        """Test manager initialization."""
        manager = CachedSerializationManager(serialization_config)
        
        assert manager.config == serialization_config
        assert manager.model_serializer is not None
        
        await manager.start()
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_serialize_model_cached(self, serialization_config, sample_data) -> Any:
        """Test cached model serialization."""
        manager = CachedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # First serialization (should not be cached)
            serialized1 = await manager.serialize_model_cached(user, SerializationFormat.JSON)
            assert isinstance(serialized1, bytes)
            
            # Second serialization (should be cached)
            serialized2 = await manager.serialize_model_cached(user, SerializationFormat.JSON)
            assert serialized1 == serialized2
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_deserialize_model_cached(self, serialization_config, sample_data) -> Any:
        """Test cached model deserialization."""
        manager = CachedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # Serialize model
            serialized = manager.model_serializer.serialize_model(user)
            
            # First deserialization (should not be cached)
            deserialized1 = await manager.deserialize_model_cached(serialized, type(user))
            assert isinstance(deserialized1, type(user))
            assert deserialized1.id == user.id
            
            # Second deserialization (should be cached)
            deserialized2 = await manager.deserialize_model_cached(serialized, type(user))
            assert deserialized2.id == user.id
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_serialize_model_no_caching(self, basic_config, sample_data) -> Any:
        """Test serialization without caching."""
        manager = CachedSerializationManager(basic_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # Serialization should work without caching
            serialized = await manager.serialize_model_cached(user)
            assert isinstance(serialized, bytes)
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_custom_cache_key(self, serialization_config, sample_data) -> Any:
        """Test serialization with custom cache key."""
        manager = CachedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            custom_key = "custom_user_key"
            
            # Serialize with custom key
            serialized1 = await manager.serialize_model_cached(user, cache_key=custom_key)
            serialized2 = await manager.serialize_model_cached(user, cache_key=custom_key)
            
            # Should be the same (cached)
            assert serialized1 == serialized2
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_get_stats(self, serialization_config, sample_data) -> Optional[Dict[str, Any]]:
        """Test manager statistics."""
        manager = CachedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # Perform operations
            await manager.serialize_model_cached(user)
            serialized = manager.model_serializer.serialize_model(user)
            await manager.deserialize_model_cached(serialized, type(user))
            
            # Get stats
            stats = manager.get_stats()
            
            assert "model_serializer" in stats
            assert "cache" in stats
            assert "config" in stats
            
        finally:
            await manager.stop()

# ============================================================================
# SERIALIZATION PROFILER TESTS
# ============================================================================

class TestSerializationProfiler:
    """Test serialization profiler functionality."""
    
    def test_profiler_initialization(self, serialization_config) -> Any:
        """Test profiler initialization."""
        profiler = SerializationProfiler(serialization_config)
        
        assert profiler.config == serialization_config
        assert profiler.profiles == {}
    
    def test_profile_operation(self, serialization_config) -> Any:
        """Test operation profiling."""
        profiler = SerializationProfiler(serialization_config)
        
        def test_function(x) -> Any:
            time.sleep(0.01)  # Simulate work
            return x * 2
        
        # Profile operation
        result = profiler.profile_operation("test_op", test_function, 5)
        assert result == 10
        
        # Check profile
        profile = profiler.get_profile("test_op")
        assert profile is not None
        assert profile["count"] == 1
        assert profile["total_time"] > 0
        assert profile["avg_time"] > 0
    
    def test_profile_operation_disabled(self, basic_config) -> Any:
        """Test profiling when disabled."""
        profiler = SerializationProfiler(basic_config)
        
        def test_function(x) -> Any:
            return x * 2
        
        # Profile operation (should not profile)
        result = profiler.profile_operation("test_op", test_function, 5)
        assert result == 10
        
        # No profile should be created
        profile = profiler.get_profile("test_op")
        assert profile is None
    
    @pytest.mark.asyncio
    async def test_profile_async_operation(self, serialization_config) -> Any:
        """Test async operation profiling."""
        profiler = SerializationProfiler(serialization_config)
        
        async def test_async_function(x) -> Any:
            await asyncio.sleep(0.01)  # Simulate async work
            return x * 2
        
        # Profile async operation
        result = await profiler.profile_async_operation("test_async_op", test_async_function, 5)
        assert result == 10
        
        # Check profile
        profile = profiler.get_profile("test_async_op")
        assert profile is not None
        assert profile["count"] == 1
        assert profile["total_time"] > 0
    
    def test_profile_slow_operation(self, serialization_config) -> Any:
        """Test profiling of slow operations."""
        config = SerializationConfig(profile_threshold=0.001)  # Very low threshold
        profiler = SerializationProfiler(config)
        
        def slow_function():
            
    """slow_function function."""
time.sleep(0.01)  # Slow operation
            return "result"
        
        # Profile slow operation
        result = profiler.profile_operation("slow_op", slow_function)
        assert result == "result"
        
        # Check profile
        profile = profiler.get_profile("slow_op")
        assert profile is not None
        assert profile["count"] == 1
    
    def test_profile_error_handling(self, serialization_config) -> Any:
        """Test profiling error handling."""
        profiler = SerializationProfiler(serialization_config)
        
        def error_function():
            
    """error_function function."""
raise ValueError("Test error")
        
        # Profile operation that raises error
        with pytest.raises(ValueError):
            profiler.profile_operation("error_op", error_function)
        
        # Profile should still be recorded
        profile = profiler.get_profile("error_op")
        assert profile is not None
        assert profile["count"] == 1
    
    def test_get_profiles(self, serialization_config) -> Optional[Dict[str, Any]]:
        """Test getting all profiles."""
        profiler = SerializationProfiler(serialization_config)
        
        def test_function(x) -> Any:
            return x * 2
        
        # Profile multiple operations
        profiler.profile_operation("op1", test_function, 1)
        profiler.profile_operation("op2", test_function, 2)
        
        # Get all profiles
        profiles = profiler.get_profiles()
        
        assert "op1" in profiles
        assert "op2" in profiles
        assert len(profiles) == 2
    
    def test_clear_profiles(self, serialization_config) -> Any:
        """Test clearing profiles."""
        profiler = SerializationProfiler(serialization_config)
        
        def test_function(x) -> Any:
            return x * 2
        
        # Profile operation
        profiler.profile_operation("test_op", test_function, 5)
        
        # Clear profiles
        profiler.clear_profiles()
        
        # Profiles should be empty
        assert profiler.profiles == {}

# ============================================================================
# DECORATOR TESTS
# ============================================================================

class TestDecorators:
    """Test serialization decorators."""
    
    @pytest.mark.asyncio
    async def test_serialized_decorator(self, sample_data) -> Any:
        """Test serialized decorator."""
        
        @serialized(SerializationFormat.JSON)
        async def get_user_model(user_id: int):
            
    """get_user_model function."""
return sample_data["user"]
        
        # Call decorated function
        result = await get_user_model(1)
        
        # Should return serialized bytes
        assert isinstance(result, bytes)
        
        # Should be valid JSON
        if result.startswith(b"gzip:"):
            decompressed = gzip.decompress(result[5:])
        else:
            decompressed = result
        
        json_data = json.loads(decompressed.decode())
        assert json_data["id"] == 1
        assert json_data["name"] == "John Doe"
    
    @pytest.mark.asyncio
    async def test_deserialized_decorator(self, sample_data) -> Any:
        """Test deserialized decorator."""
        
        @deserialized(type(sample_data["user"]), SerializationFormat.JSON)
        async def process_user_model(user) -> Any:
            return f"Processed: {user.name}"
        
        # Serialize user first
        user = sample_data["user"]
        serialized = user.model_dump_json().encode()
        
        # Call decorated function
        result = await process_user_model(serialized)
        
        # Should return processed result
        assert result == "Processed: John Doe"
    
    @pytest.mark.asyncio
    async def test_serialized_decorator_with_manager(self, serialization_config, sample_data) -> Any:
        """Test serialized decorator with custom manager."""
        manager = CachedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            @serialized(SerializationFormat.JSON, manager=manager)
            async def get_user_model(user_id: int):
                
    """get_user_model function."""
return sample_data["user"]
            
            # Call decorated function
            result = await get_user_model(1)
            
            # Should return serialized bytes
            assert isinstance(result, bytes)
            
        finally:
            await manager.stop()

# ============================================================================
# OPTIMIZED SERIALIZATION MANAGER TESTS
# ============================================================================

class TestOptimizedSerializationManager:
    """Test optimized serialization manager functionality."""
    
    @pytest.mark.asyncio
    async def test_manager_initialization(self, serialization_config) -> Any:
        """Test manager initialization."""
        manager = OptimizedSerializationManager(serialization_config)
        
        assert manager.config == serialization_config
        assert manager.cached_manager is not None
        assert manager.profiler is not None
        
        await manager.start()
        await manager.stop()
    
    @pytest.mark.asyncio
    async def test_serialize_model(self, serialization_config, sample_data) -> Any:
        """Test model serialization with profiling."""
        manager = OptimizedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # Serialize model
            serialized = await manager.serialize_model(user, SerializationFormat.JSON)
            assert isinstance(serialized, bytes)
            
            # Check profiling
            profiles = manager.profiler.get_profiles()
            assert "serialize_model" in profiles
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_deserialize_model(self, serialization_config, sample_data) -> Any:
        """Test model deserialization with profiling."""
        manager = OptimizedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # Serialize first
            serialized = manager.cached_manager.model_serializer.serialize_model(user)
            
            # Deserialize model
            deserialized = await manager.deserialize_model(serialized, type(user))
            assert isinstance(deserialized, type(user))
            assert deserialized.id == user.id
            
            # Check profiling
            profiles = manager.profiler.get_profiles()
            assert "deserialize_model" in profiles
            
        finally:
            await manager.stop()
    
    def test_serialize_sync(self, serialization_config, sample_data) -> Any:
        """Test synchronous model serialization."""
        manager = OptimizedSerializationManager(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize model
        serialized = manager.serialize_sync(user, SerializationFormat.JSON)
        assert isinstance(serialized, bytes)
        
        # Check profiling
        profiles = manager.profiler.get_profiles()
        assert "serialize_sync" in profiles
    
    def test_deserialize_sync(self, serialization_config, sample_data) -> Any:
        """Test synchronous model deserialization."""
        manager = OptimizedSerializationManager(serialization_config)
        
        user = sample_data["user"]
        
        # Serialize first
        serialized = manager.cached_manager.model_serializer.serialize_model(user)
        
        # Deserialize model
        deserialized = manager.deserialize_sync(serialized, type(user))
        assert isinstance(deserialized, type(user))
        assert deserialized.id == user.id
        
        # Check profiling
        profiles = manager.profiler.get_profiles()
        assert "deserialize_sync" in profiles
    
    def test_get_stats(self, serialization_config, sample_data) -> Optional[Dict[str, Any]]:
        """Test manager statistics."""
        manager = OptimizedSerializationManager(serialization_config)
        
        user = sample_data["user"]
        
        # Perform operations
        manager.serialize_sync(user)
        serialized = manager.cached_manager.model_serializer.serialize_model(user)
        manager.deserialize_sync(serialized, type(user))
        
        # Get stats
        stats = manager.get_stats()
        
        assert "cached_manager" in stats
        assert "profiler" in stats
        assert "config" in stats
    
    def test_get_performance_report(self, serialization_config, sample_data) -> Optional[Dict[str, Any]]:
        """Test performance report generation."""
        manager = OptimizedSerializationManager(serialization_config)
        
        user = sample_data["user"]
        
        # Perform operations
        manager.serialize_sync(user)
        serialized = manager.cached_manager.model_serializer.serialize_model(user)
        manager.deserialize_sync(serialized, type(user))
        
        # Get performance report
        report = manager.get_performance_report()
        
        assert "summary" in report
        assert "operations" in report
        assert "recommendations" in report
        
        # Check summary
        summary = report["summary"]
        assert summary["total_operations"] > 0
        assert summary["total_time"] > 0

# ============================================================================
# UTILITY FUNCTION TESTS
# ============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_serialization_config(self) -> Any:
        """Test serialization config creation."""
        config = create_serialization_config(
            default_format=SerializationFormat.PICKLE,
            enable_compression=False
        )
        
        assert config.default_format == SerializationFormat.PICKLE
        assert config.enable_compression is False
    
    def test_create_serialization_manager(self) -> Any:
        """Test serialization manager creation."""
        manager = create_serialization_manager()
        
        assert isinstance(manager, OptimizedSerializationManager)
        assert manager.config is not None
    
    def test_benchmark_serialization(self, sample_data) -> Any:
        """Test serialization benchmarking."""
        models = list(sample_data.values())
        
        # Run benchmark
        results = benchmark_serialization(models)
        
        assert "formats" in results
        assert "compression_levels" in results
        assert "model_sizes" in results
        
        # Check formats
        formats = results["formats"]
        assert "json" in formats
        assert "pickle" in formats
        
        # Check compression levels
        compression_levels = results["compression_levels"]
        assert "none" in compression_levels
        assert "balanced" in compression_levels

# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestIntegration:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_complete_workflow(self, serialization_config, sample_data) -> Any:
        """Test complete serialization workflow."""
        manager = OptimizedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            user = sample_data["user"]
            post = sample_data["post"]
            
            # Serialize models with different formats
            user_json = await manager.serialize_model(user, SerializationFormat.JSON)
            post_pickle = await manager.serialize_model(post, SerializationFormat.PICKLE)
            
            # Deserialize models
            user_deserialized = await manager.deserialize_model(user_json, type(user), SerializationFormat.JSON)
            post_deserialized = await manager.deserialize_model(post_pickle, type(post), SerializationFormat.PICKLE)
            
            # Verify data integrity
            assert user_deserialized.id == user.id
            assert user_deserialized.name == user.name
            assert post_deserialized.id == post.id
            assert post_deserialized.title == post.title
            
            # Get performance report
            report = manager.get_performance_report()
            assert report["summary"]["total_operations"] >= 4
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_caching_integration(self, serialization_config, sample_data) -> Any:
        """Test integration with caching system."""
        if not CACHING_AVAILABLE:
            pytest.skip("Caching system not available")
        
        # Create cache manager
        cache_config = CacheConfig(
            memory_cache_size=100,
            enable_multi_tier=False
        )
        cache_manager = create_cache_manager(cache_config)
        
        # Create serialization manager with cache
        manager = OptimizedSerializationManager(serialization_config, cache_manager)
        await manager.start()
        
        try:
            user = sample_data["user"]
            
            # First serialization (should cache)
            serialized1 = await manager.serialize_model(user)
            
            # Second serialization (should use cache)
            serialized2 = await manager.serialize_model(user)
            
            # Should be the same
            assert serialized1 == serialized2
            
            # Check cache stats
            stats = manager.get_stats()
            cache_stats = stats["cached_manager"]["cache"]
            assert cache_stats["l1_cache"]["hits"] > 0
            
        finally:
            await manager.stop()
    
    @pytest.mark.asyncio
    async def test_error_recovery(self, serialization_config) -> Any:
        """Test error recovery scenarios."""
        manager = OptimizedSerializationManager(serialization_config)
        await manager.start()
        
        try:
            # Test with invalid data
            invalid_data = b"invalid data"
            
            class TestModel(BaseModel):
                id: int
                name: str
            
            # Should handle error gracefully
            with pytest.raises(Exception):
                await manager.deserialize_model(invalid_data, TestModel)
            
            # Manager should still be functional
            valid_model = TestModel(id=1, name="test")
            serialized = await manager.serialize_model(valid_model)
            assert isinstance(serialized, bytes)
            
        finally:
            await manager.stop()

# ============================================================================
# PERFORMANCE TESTS
# ============================================================================

class TestPerformance:
    """Test performance characteristics."""
    
    def test_serialization_performance(self, serialization_config) -> Any:
        """Test serialization performance."""
        manager = OptimizedSerializationManager(serialization_config)
        
        # Create large model
        class LargeModel(BaseModel):
            id: int
            data: str = "x" * 10000  # Large string
            items: List[int] = [i for i in range(1000)]  # Large list
        
        large_model = LargeModel(id=1)
        
        # Measure serialization time
        start_time = time.time()
        serialized = manager.serialize_sync(large_model)
        serialization_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert serialization_time < 1.0  # Less than 1 second
        
        # Measure deserialization time
        start_time = time.time()
        deserialized = manager.deserialize_sync(serialized, LargeModel)
        deserialization_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert deserialization_time < 1.0  # Less than 1 second
        
        # Verify data integrity
        assert deserialized.id == large_model.id
        assert deserialized.data == large_model.data
        assert deserialized.items == large_model.items
    
    def test_compression_effectiveness(self, serialization_config) -> Any:
        """Test compression effectiveness."""
        manager = OptimizedSerializationManager(serialization_config)
        
        # Create model with repetitive data
        class RepetitiveModel(BaseModel):
            id: int
            data: str = "repetitive data " * 1000  # Highly compressible
        
        model = RepetitiveModel(id=1)
        
        # Serialize with compression
        serialized_compressed = manager.serialize_sync(model)
        
        # Serialize without compression
        config_no_compression = SerializationConfig(enable_compression=False)
        manager_no_compression = OptimizedSerializationManager(config_no_compression)
        serialized_uncompressed = manager_no_compression.serialize_sync(model)
        
        # Compressed should be smaller
        assert len(serialized_compressed) < len(serialized_uncompressed)
    
    def test_cache_performance(self, serialization_config, sample_data) -> Any:
        """Test cache performance."""
        manager = OptimizedSerializationManager(serialization_config)
        
        user = sample_data["user"]
        
        # First serialization (cache miss)
        start_time = time.time()
        serialized1 = manager.serialize_sync(user)
        first_time = time.time() - start_time
        
        # Second serialization (cache hit)
        start_time = time.time()
        serialized2 = manager.serialize_sync(user)
        second_time = time.time() - start_time
        
        # Cache hit should be faster
        assert second_time < first_time
        assert serialized1 == serialized2

match __name__:
    case "__main__":
    pytest.main([__file__, "-v"]) 