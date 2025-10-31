"""
Unit tests for the ads infrastructure layer.

This module consolidates tests for:
- Database management and connection pooling
- Storage strategies and file management
- Cache management and strategies
- External service integrations
- Repository implementations
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import tempfile
import os
import json

from agents.backend.onyx.server.features.ads.infrastructure.database import (
    DatabaseConfig, ConnectionPool, DatabaseManager,
    AdsRepository, CampaignRepository, GroupRepository, PerformanceRepository
)
from agents.backend.onyx.server.features.ads.infrastructure.storage import (
    StorageType, StorageConfig, StorageStrategy, LocalStorageStrategy,
    CloudStorageStrategy, FileStorageManager
)
from agents.backend.onyx.server.features.ads.infrastructure.cache import (
    CacheType, CacheConfig, CacheStrategy, MemoryCacheStrategy,
    RedisCacheStrategy, CacheManager, CacheService
)
from agents.backend.onyx.server.features.ads.infrastructure.external_services import (
    ServiceType, ServiceStatus, ExternalServiceConfig, ExternalServiceManager,
    AIProviderService, AnalyticsService, NotificationService
)
from agents.backend.onyx.server.features.ads.infrastructure.repositories import (
    AdsRepositoryImpl, CampaignRepositoryImpl, GroupRepositoryImpl,
    PerformanceRepositoryImpl, AnalyticsRepositoryImpl, OptimizationRepositoryImpl,
    RepositoryFactory
)


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""
    
    def test_database_config_creation(self):
        """Test DatabaseConfig creation with valid values."""
        config = DatabaseConfig(
            host="localhost",
            port=5432,
            database="test_db",
            username="test_user",
            password="test_pass",
            pool_size=10,
            max_overflow=20,
            pool_timeout=30,
            pool_recycle=3600
        )
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "test_db"
        assert config.username == "test_user"
        assert config.password == "test_pass"
        assert config.pool_size == 10
        assert config.max_overflow == 20
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig creation with default values."""
        config = DatabaseConfig(
            host="localhost",
            database="test_db",
            username="test_user",
            password="test_pass"
        )
        assert config.port == 5432
        assert config.pool_size == 5
        assert config.max_overflow == 10
        assert config.pool_timeout == 30
        assert config.pool_recycle == 3600


class TestConnectionPool:
    """Test ConnectionPool class."""
    
    @pytest.fixture
    def mock_engine(self):
        """Mock SQLAlchemy engine."""
        return Mock()
    
    @pytest.fixture
    def mock_session_factory(self):
        """Mock SQLAlchemy session factory."""
        return Mock()
    
    @pytest.fixture
    def connection_pool(self, mock_engine, mock_session_factory):
        """Create ConnectionPool instance with mocked dependencies."""
        return ConnectionPool(
            engine=mock_engine,
            session_factory=mock_session_factory
        )
    
    def test_connection_pool_creation(self, connection_pool):
        """Test ConnectionPool creation."""
        assert connection_pool.engine is not None
        assert connection_pool.session_factory is not None
        assert connection_pool.stats is not None
    
    def test_connection_pool_stats(self, connection_pool):
        """Test connection pool statistics."""
        stats = connection_pool.get_stats()
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "idle_connections" in stats
        assert "created_at" in stats
    
    def test_connection_pool_health_check(self, connection_pool):
        """Test connection pool health check."""
        health = connection_pool.health_check()
        assert "status" in health
        assert "message" in health
        assert "timestamp" in health


class TestDatabaseManager:
    """Test DatabaseManager class."""
    
    @pytest.fixture
    def mock_connection_pool(self):
        """Mock ConnectionPool."""
        return Mock(spec=ConnectionPool)
    
    @pytest.fixture
    def database_manager(self, mock_connection_pool):
        """Create DatabaseManager instance with mocked dependencies."""
        return DatabaseManager(connection_pool=mock_connection_pool)
    
    @pytest.mark.asyncio
    async def test_database_manager_get_session(self, database_manager, mock_connection_pool):
        """Test getting a database session."""
        mock_session = Mock()
        mock_connection_pool.get_session.return_value = mock_session
        
        session = await database_manager.get_session()
        
        assert session == mock_session
        mock_connection_pool.get_session.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_database_manager_execute_with_retry(self, database_manager):
        """Test executing database operation with retry logic."""
        mock_operation = AsyncMock()
        mock_operation.side_effect = [Exception("Connection error"), "Success"]
        
        result = await database_manager.execute_with_retry(mock_operation, max_retries=2)
        
        assert result == "Success"
        assert mock_operation.call_count == 2
    
    @pytest.mark.asyncio
    async def test_database_manager_health_check(self, database_manager, mock_connection_pool):
        """Test database manager health check."""
        mock_connection_pool.health_check.return_value = {
            "status": "healthy",
            "message": "All good",
            "timestamp": datetime.now()
        }
        
        health = await database_manager.health_check()
        
        assert health["status"] == "healthy"
        assert health["message"] == "All good"
        mock_connection_pool.health_check.assert_called_once()


class TestStorageConfig:
    """Test StorageConfig dataclass."""
    
    def test_storage_config_creation(self):
        """Test StorageConfig creation with valid values."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            base_path="/tmp/storage",
            max_file_size=10485760,  # 10MB
            allowed_extensions=[".jpg", ".png", ".pdf"],
            compression_enabled=True,
            encryption_enabled=False
        )
        assert config.storage_type == StorageType.LOCAL
        assert config.base_path == "/tmp/storage"
        assert config.max_file_size == 10485760
        assert ".jpg" in config.allowed_extensions
        assert config.compression_enabled is True
        assert config.encryption_enabled is False
    
    def test_storage_config_defaults(self):
        """Test StorageConfig creation with default values."""
        config = StorageConfig(
            storage_type=StorageType.LOCAL,
            base_path="/tmp/storage"
        )
        assert config.max_file_size == 52428800  # 50MB
        assert config.allowed_extensions == [".jpg", ".png", ".pdf", ".doc", ".txt"]
        assert config.compression_enabled is False
        assert config.encryption_enabled is False


class TestLocalStorageStrategy:
    """Test LocalStorageStrategy class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def storage_config(self, temp_dir):
        """Create StorageConfig for testing."""
        return StorageConfig(
            storage_type=StorageType.LOCAL,
            base_path=temp_dir,
            max_file_size=1024,
            allowed_extensions=[".txt", ".json"]
        )
    
    @pytest.fixture
    def local_storage(self, storage_config):
        """Create LocalStorageStrategy instance."""
        return LocalStorageStrategy(config=storage_config)
    
    @pytest.mark.asyncio
    async def test_local_storage_save_file(self, local_storage, temp_dir):
        """Test saving a file locally."""
        file_path = "test.txt"
        content = b"Hello, World!"
        
        result = await local_storage.save_file(file_path, content)
        
        assert result["success"] is True
        assert result["file_path"] == os.path.join(temp_dir, file_path)
        
        # Verify file was created
        full_path = os.path.join(temp_dir, file_path)
        assert os.path.exists(full_path)
        with open(full_path, "rb") as f:
            assert f.read() == content
    
    @pytest.mark.asyncio
    async def test_local_storage_get_file(self, local_storage, temp_dir):
        """Test getting a file from local storage."""
        # Create a test file
        file_path = "test.json"
        content = b'{"key": "value"}'
        full_path = os.path.join(temp_dir, file_path)
        
        with open(full_path, "wb") as f:
            f.write(content)
        
        result = await local_storage.get_file(file_path)
        
        assert result["success"] is True
        assert result["content"] == content
        assert result["file_path"] == full_path
    
    @pytest.mark.asyncio
    async def test_local_storage_delete_file(self, local_storage, temp_dir):
        """Test deleting a file from local storage."""
        # Create a test file
        file_path = "test.txt"
        full_path = os.path.join(temp_dir, file_path)
        
        with open(full_path, "w") as f:
            f.write("test content")
        
        assert os.path.exists(full_path)
        
        result = await local_storage.delete_file(file_path)
        
        assert result["success"] is True
        assert not os.path.exists(full_path)
    
    @pytest.mark.asyncio
    async def test_local_storage_file_validation(self, local_storage):
        """Test file validation in local storage."""
        # Test file size validation
        large_content = b"x" * 2048  # Exceeds 1024 limit
        
        result = await local_storage.save_file("large.txt", large_content)
        
        assert result["success"] is False
        assert "file size exceeds limit" in result["error"]
    
    @pytest.mark.asyncio
    async def test_local_storage_extension_validation(self, local_storage):
        """Test file extension validation in local storage."""
        # Test invalid extension
        content = b"test content"
        
        result = await local_storage.save_file("test.invalid", content)
        
        assert result["success"] is False
        assert "file extension not allowed" in result["error"]


class TestCloudStorageStrategy:
    """Test CloudStorageStrategy class."""
    
    @pytest.fixture
    def storage_config(self):
        """Create StorageConfig for testing."""
        return StorageConfig(
            storage_type=StorageType.CLOUD,
            base_path="s3://bucket/path",
            max_file_size=10485760,
            allowed_extensions=[".jpg", ".png", ".pdf"]
        )
    
    @pytest.fixture
    def cloud_storage(self, storage_config):
        """Create CloudStorageStrategy instance."""
        return CloudStorageStrategy(config=storage_config)
    
    def test_cloud_storage_creation(self, cloud_storage):
        """Test CloudStorageStrategy creation."""
        assert cloud_storage.config.storage_type == StorageType.CLOUD
        assert cloud_storage.config.base_path == "s3://bucket/path"
    
    @pytest.mark.asyncio
    async def test_cloud_storage_placeholder_methods(self, cloud_storage):
        """Test that cloud storage methods are placeholders."""
        # These methods should return NotImplementedError for now
        with pytest.raises(NotImplementedError):
            await cloud_storage.save_file("test.txt", b"content")
        
        with pytest.raises(NotImplementedError):
            await cloud_storage.get_file("test.txt")
        
        with pytest.raises(NotImplementedError):
            await cloud_storage.delete_file("test.txt")


class TestFileStorageManager:
    """Test FileStorageManager class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def storage_config(self):
        """Create StorageConfig for testing."""
        return StorageConfig(
            storage_type=StorageType.LOCAL,
            base_path="/tmp/storage",
            max_file_size=1024,
            allowed_extensions=[".txt", ".json"]
        )
    
    @pytest.fixture
    def file_storage_manager(self, storage_config, mock_redis):
        """Create FileStorageManager instance."""
        return FileStorageManager(
            config=storage_config,
            redis_client=mock_redis
        )
    
    @pytest.mark.asyncio
    async def test_file_storage_manager_creation(self, file_storage_manager):
        """Test FileStorageManager creation."""
        assert file_storage_manager.config == storage_config
        assert file_storage_manager.redis_client is not None
        assert file_storage_manager.storage_strategy is not None
    
    @pytest.mark.asyncio
    async def test_file_storage_manager_save_file(self, file_storage_manager, mock_redis):
        """Test saving a file through the manager."""
        file_path = "test.txt"
        content = b"Hello, World!"
        
        # Mock the storage strategy
        with patch.object(file_storage_manager.storage_strategy, 'save_file') as mock_save:
            mock_save.return_value = {
                "success": True,
                "file_path": "/tmp/storage/test.txt"
            }
            
            result = await file_storage_manager.save_file(file_path, content)
            
            assert result["success"] is True
            mock_save.assert_called_once_with(file_path, content)
    
    @pytest.mark.asyncio
    async def test_file_storage_manager_cache_integration(self, file_storage_manager, mock_redis):
        """Test file storage manager cache integration."""
        file_path = "test.txt"
        content = b"Hello, World!"
        
        # Mock Redis cache
        mock_redis.get.return_value = None  # Cache miss
        mock_redis.set.return_value = True
        
        # Mock the storage strategy
        with patch.object(file_storage_manager.storage_strategy, 'save_file') as mock_save:
            mock_save.return_value = {
                "success": True,
                "file_path": "/tmp/storage/test.txt"
            }
            
            await file_storage_manager.save_file(file_path, content)
            
            # Verify cache was set
            mock_redis.set.assert_called()


class TestCacheConfig:
    """Test CacheConfig dataclass."""
    
    def test_cache_config_creation(self):
        """Test CacheConfig creation with valid values."""
        config = CacheConfig(
            cache_type=CacheType.REDIS,
            ttl=3600,
            max_size=1000,
            compression_enabled=True,
            encryption_enabled=False
        )
        assert config.cache_type == CacheType.REDIS
        assert config.ttl == 3600
        assert config.max_size == 1000
        assert config.compression_enabled is True
        assert config.encryption_enabled is False
    
    def test_cache_config_defaults(self):
        """Test CacheConfig creation with default values."""
        config = CacheConfig(cache_type=CacheType.MEMORY)
        assert config.ttl == 300  # 5 minutes
        assert config.max_size == 100
        assert config.compression_enabled is False
        assert config.encryption_enabled is False


class TestMemoryCacheStrategy:
    """Test MemoryCacheStrategy class."""
    
    @pytest.fixture
    def cache_config(self):
        """Create CacheConfig for testing."""
        return CacheConfig(
            cache_type=CacheType.MEMORY,
            ttl=60,
            max_size=100
        )
    
    @pytest.fixture
    def memory_cache(self, cache_config):
        """Create MemoryCacheStrategy instance."""
        return MemoryCacheStrategy(config=cache_config)
    
    @pytest.mark.asyncio
    async def test_memory_cache_set_get(self, memory_cache):
        """Test setting and getting values from memory cache."""
        key = "test_key"
        value = {"data": "test_value"}
        
        # Set value
        result = await memory_cache.set(key, value)
        assert result["success"] is True
        
        # Get value
        result = await memory_cache.get(key)
        assert result["success"] is True
        assert result["value"] == value
    
    @pytest.mark.asyncio
    async def test_memory_cache_ttl_expiration(self, memory_cache):
        """Test TTL expiration in memory cache."""
        key = "test_key"
        value = "test_value"
        
        # Set value with short TTL
        await memory_cache.set(key, value, ttl=0.1)  # 100ms
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Try to get expired value
        result = await memory_cache.get(key)
        assert result["success"] is False
        assert "not found" in result["error"]
    
    @pytest.mark.asyncio
    async def test_memory_cache_delete(self, memory_cache):
        """Test deleting values from memory cache."""
        key = "test_key"
        value = "test_value"
        
        # Set value
        await memory_cache.set(key, value)
        
        # Verify value exists
        result = await memory_cache.get(key)
        assert result["success"] is True
        
        # Delete value
        result = await memory_cache.delete(key)
        assert result["success"] is True
        
        # Verify value is gone
        result = await memory_cache.get(key)
        assert result["success"] is False


class TestRedisCacheStrategy:
    """Test RedisCacheStrategy class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def cache_config(self):
        """Create CacheConfig for testing."""
        return CacheConfig(
            cache_type=CacheType.REDIS,
            ttl=3600,
            max_size=1000
        )
    
    @pytest.fixture
    def redis_cache(self, cache_config, mock_redis):
        """Create RedisCacheStrategy instance."""
        return RedisCacheStrategy(config=cache_config, redis_client=mock_redis)
    
    @pytest.mark.asyncio
    async def test_redis_cache_set(self, redis_cache, mock_redis):
        """Test setting values in Redis cache."""
        key = "test_key"
        value = {"data": "test_value"}
        
        mock_redis.set.return_value = True
        
        result = await redis_cache.set(key, value)
        
        assert result["success"] is True
        mock_redis.set.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_redis_cache_get(self, redis_cache, mock_redis):
        """Test getting values from Redis cache."""
        key = "test_key"
        value = '{"data": "test_value"}'
        
        mock_redis.get.return_value = value.encode()
        
        result = await redis_cache.get(key)
        
        assert result["success"] is True
        assert result["value"]["data"] == "test_value"
        mock_redis.get.assert_called_once_with(key)
    
    @pytest.mark.asyncio
    async def test_redis_cache_delete(self, redis_cache, mock_redis):
        """Test deleting values from Redis cache."""
        key = "test_key"
        
        mock_redis.delete.return_value = 1
        
        result = await redis_cache.delete(key)
        
        assert result["success"] is True
        mock_redis.delete.assert_called_once_with(key)


class TestCacheManager:
    """Test CacheManager class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def cache_config(self):
        """Create CacheConfig for testing."""
        return CacheConfig(
            cache_type=CacheType.REDIS,
            ttl=3600,
            max_size=1000
        )
    
    @pytest.fixture
    def cache_manager(self, cache_config, mock_redis):
        """Create CacheManager instance."""
        return CacheManager(
            config=cache_config,
            redis_client=mock_redis
        )
    
    @pytest.mark.asyncio
    async def test_cache_manager_set_get(self, cache_manager, mock_redis):
        """Test setting and getting values through cache manager."""
        key = "test_key"
        value = {"data": "test_value"}
        
        mock_redis.set.return_value = True
        mock_redis.get.return_value = json.dumps(value).encode()
        
        # Set value
        result = await cache_manager.set(key, value)
        assert result["success"] is True
        
        # Get value
        result = await cache_manager.get(key)
        assert result["success"] is True
        assert result["value"] == value
    
    @pytest.mark.asyncio
    async def test_cache_manager_pattern_invalidation(self, cache_manager, mock_redis):
        """Test pattern-based cache invalidation."""
        pattern = "user:*"
        
        mock_redis.scan_iter.return_value = ["user:1", "user:2", "user:3"]
        mock_redis.delete.return_value = 3
        
        result = await cache_manager.invalidate_pattern(pattern)
        
        assert result["success"] is True
        assert result["deleted_count"] == 3
        mock_redis.scan_iter.assert_called_once_with(match=pattern)


class TestCacheService:
    """Test CacheService class."""
    
    @pytest.fixture
    def mock_cache_manager(self):
        """Mock CacheManager."""
        return Mock(spec=CacheManager)
    
    @pytest.fixture
    def cache_service(self, mock_cache_manager):
        """Create CacheService instance."""
        return CacheService(cache_manager=mock_cache_manager)
    
    @pytest.mark.asyncio
    async def test_cache_service_get_user_preferences(self, cache_service, mock_cache_manager):
        """Test getting user preferences from cache."""
        user_id = "user_123"
        preferences = {"theme": "dark", "language": "en"}
        
        mock_cache_manager.get.return_value = {
            "success": True,
            "value": preferences
        }
        
        result = await cache_service.get_user_preferences(user_id)
        
        assert result == preferences
        mock_cache_manager.get.assert_called_once_with(f"user_prefs:{user_id}")
    
    @pytest.mark.asyncio
    async def test_cache_service_cache_miss_fallback(self, cache_service, mock_cache_manager):
        """Test cache miss fallback behavior."""
        user_id = "user_123"
        
        mock_cache_manager.get.return_value = {
            "success": False,
            "error": "not found"
        }
        
        # Mock fallback function
        fallback_func = AsyncMock(return_value={"theme": "light"})
        
        result = await cache_service.get_with_fallback(
            f"user_prefs:{user_id}",
            fallback_func
        )
        
        assert result["theme"] == "light"
        fallback_func.assert_called_once()


class TestExternalServiceConfig:
    """Test ExternalServiceConfig dataclass."""
    
    def test_external_service_config_creation(self):
        """Test ExternalServiceConfig creation with valid values."""
        config = ExternalServiceConfig(
            service_type=ServiceType.AI_PROVIDER,
            base_url="https://api.openai.com",
            api_key="sk-1234567890",
            timeout=30,
            max_retries=3,
            rate_limit_per_minute=60
        )
        assert config.service_type == ServiceType.AI_PROVIDER
        assert config.base_url == "https://api.openai.com"
        assert config.api_key == "sk-1234567890"
        assert config.timeout == 30
        assert config.max_retries == 3
        assert config.rate_limit_per_minute == 60


class TestExternalServiceManager:
    """Test ExternalServiceManager class."""
    
    @pytest.fixture
    def mock_redis(self):
        """Mock Redis client."""
        return AsyncMock()
    
    @pytest.fixture
    def service_config(self):
        """Create ExternalServiceConfig for testing."""
        return ExternalServiceConfig(
            service_type=ServiceType.AI_PROVIDER,
            base_url="https://api.test.com",
            api_key="test_key",
            timeout=30,
            max_retries=3,
            rate_limit_per_minute=60
        )
    
    @pytest.fixture
    def service_manager(self, service_config, mock_redis):
        """Create ExternalServiceManager instance."""
        return ExternalServiceManager(
            config=service_config,
            redis_client=mock_redis
        )
    
    @pytest.mark.asyncio
    async def test_service_manager_creation(self, service_manager):
        """Test ExternalServiceManager creation."""
        assert service_manager.config == service_config
        assert service_manager.redis_client is not None
        assert service_manager.session is not None
    
    @pytest.mark.asyncio
    async def test_service_manager_rate_limiting(self, service_manager, mock_redis):
        """Test rate limiting in service manager."""
        endpoint = "/test"
        
        # Mock Redis rate limit check
        mock_redis.get.return_value = "5"  # 5 requests in current window
        
        can_proceed = await service_manager.check_rate_limit(endpoint)
        
        assert can_proceed is True
        mock_redis.get.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_service_manager_health_check(self, service_manager):
        """Test service manager health check."""
        health = await service_manager.health_check()
        
        assert "status" in health
        assert "message" in health
        assert "timestamp" in health


class TestRepositoryImplementations:
    """Test repository implementations."""
    
    @pytest.fixture
    def mock_database_manager(self):
        """Mock DatabaseManager."""
        return Mock(spec=DatabaseManager)
    
    @pytest.fixture
    def ads_repository(self, mock_database_manager):
        """Create AdsRepositoryImpl instance."""
        return AdsRepositoryImpl(database_manager=mock_database_manager)
    
    @pytest.fixture
    def campaign_repository(self, mock_database_manager):
        """Create CampaignRepositoryImpl instance."""
        return CampaignRepositoryImpl(database_manager=mock_database_manager)
    
    def test_ads_repository_creation(self, ads_repository):
        """Test AdsRepositoryImpl creation."""
        assert ads_repository.database_manager is not None
        assert hasattr(ads_repository, 'create')
        assert hasattr(ads_repository, 'get_by_id')
        assert hasattr(ads_repository, 'get_all')
    
    def test_campaign_repository_creation(self, campaign_repository):
        """Test CampaignRepositoryImpl creation."""
        assert campaign_repository.database_manager is not None
        assert hasattr(campaign_repository, 'create')
        assert hasattr(campaign_repository, 'get_by_id')
        assert hasattr(campaign_repository, 'get_all')


class TestRepositoryFactory:
    """Test RepositoryFactory class."""
    
    @pytest.fixture
    def mock_database_manager(self):
        """Mock DatabaseManager."""
        return Mock(spec=DatabaseManager)
    
    @pytest.fixture
    def repository_factory(self, mock_database_manager):
        """Create RepositoryFactory instance."""
        return RepositoryFactory(database_manager=mock_database_manager)
    
    def test_repository_factory_creation(self, repository_factory):
        """Test RepositoryFactory creation."""
        assert repository_factory.database_manager is not None
    
    def test_repository_factory_get_ads_repository(self, repository_factory):
        """Test getting ads repository from factory."""
        ads_repo = repository_factory.get_ads_repository()
        
        assert ads_repo is not None
        assert isinstance(ads_repo, AdsRepositoryImpl)
    
    def test_repository_factory_get_campaign_repository(self, repository_factory):
        """Test getting campaign repository from factory."""
        campaign_repo = repository_factory.get_campaign_repository()
        
        assert campaign_repo is not None
        assert isinstance(campaign_repo, CampaignRepositoryImpl)
    
    def test_repository_factory_get_all_repositories(self, repository_factory):
        """Test getting all repositories from factory."""
        repositories = repository_factory.get_all_repositories()
        
        assert "ads" in repositories
        assert "campaign" in repositories
        assert "group" in repositories
        assert "performance" in repositories
        assert "analytics" in repositories
        assert "optimization" in repositories


if __name__ == "__main__":
    pytest.main([__file__])
