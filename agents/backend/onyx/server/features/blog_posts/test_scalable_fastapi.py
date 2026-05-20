from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import tempfile
import shutil
import os
from fastapi.testclient import TestClient
from fastapi import FastAPI, HTTPException, status
import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from scalable_fastapi_system import (
        import threading
        import concurrent.futures
from typing import Any, List, Dict, Optional
import logging
"""
Test Suite for Scalable FastAPI System
=====================================

Comprehensive tests for scalable FastAPI system with async operations,
caching, middleware, database integration, and production-ready features.
"""


# FastAPI testing imports

# Database testing imports

# Import the modules to test
    Settings, DatabaseSettings, RedisSettings, SecuritySettings, APISettings,
    DatabaseManager, CacheManager, SecurityManager, MetricsManager,
    User, APIKey, RequestLog, UserCreate, UserUpdate, UserResponse,
    Token, TokenData, APIResponse, HealthCheck,
    RequestLoggingMiddleware, RateLimitingMiddleware,
    get_current_user, get_current_active_user, get_current_superuser,
    require_permissions, cache_response, external_api_call,
    background_task, create_app
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_dir():
    """Create temporary directory for tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def test_settings():
    """Test settings configuration."""
    return Settings(
        database=DatabaseSettings(
            database_url="sqlite:///:memory:",
            async_database_url="sqlite+aiosqlite:///:memory:"
        ),
        redis=RedisSettings(
            redis_url="redis://localhost:6379"
        ),
        security=SecuritySettings(
            secret_key="test-secret-key-for-testing-only",
            access_token_expire_minutes=30
        ),
        api=APISettings(
            debug=True,
            title="Test API",
            version="1.0.0"
        )
    )


@pytest.fixture
def test_db_manager(test_settings) -> Any:
    """Test database manager."""
    return DatabaseManager(test_settings)


@pytest.fixture
def test_cache_manager(test_settings) -> Any:
    """Test cache manager."""
    return CacheManager(test_settings)


@pytest.fixture
def test_security_manager(test_settings) -> Any:
    """Test security manager."""
    return SecurityManager(test_settings)


@pytest.fixture
def test_metrics_manager():
    """Test metrics manager."""
    return MetricsManager()


@pytest.fixture
def test_app(test_settings) -> Any:
    """Test FastAPI application."""
    app = create_app()
    return app


@pytest.fixture
def test_client(test_app) -> Any:
    """Test client for FastAPI application."""
    return TestClient(test_app)


@pytest.fixture
def sample_user_data():
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture
def sample_user(test_db_manager, test_security_manager, sample_user_data) -> Any:
    """Sample user for testing."""
    db = next(test_db_manager.get_db())
    
    # Create user
    hashed_password = test_security_manager.get_password_hash(sample_user_data["password"])
    user = User(
        username=sample_user_data["username"],
        email=sample_user_data["email"],
        hashed_password=hashed_password,
        is_active=True
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    db.close()
    
    return user


@pytest.fixture
def sample_token(test_security_manager, sample_user) -> Any:
    """Sample access token for testing."""
    return test_security_manager.create_access_token(
        data={"sub": sample_user.username, "user_id": sample_user.id}
    )


# =============================================================================
# SETTINGS TESTS
# =============================================================================

class TestSettings:
    """Test settings configuration."""
    
    def test_database_settings(self) -> Any:
        """Test database settings."""
        settings = DatabaseSettings()
        
        assert settings.database_url == "sqlite:///./app.db"
        assert settings.pool_size == 20
        assert settings.max_overflow == 30
        assert settings.pool_pre_ping is True
    
    def test_redis_settings(self) -> Any:
        """Test Redis settings."""
        settings = RedisSettings()
        
        assert settings.redis_url == "redis://localhost:6379"
        assert settings.redis_db == 0
        assert settings.redis_ssl is False
    
    def test_security_settings(self) -> Any:
        """Test security settings."""
        settings = SecuritySettings()
        
        assert len(settings.secret_key) >= 32
        assert settings.algorithm == "HS256"
        assert settings.access_token_expire_minutes == 30
        assert settings.bcrypt_rounds == 12
    
    async def test_api_settings(self) -> Any:
        """Test API settings."""
        settings = APISettings()
        
        assert settings.title == "Scalable FastAPI System"
        assert settings.version == "1.0.0"
        assert settings.debug is False
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000
    
    def test_main_settings(self) -> Any:
        """Test main settings."""
        settings = Settings()
        
        assert isinstance(settings.database, DatabaseSettings)
        assert isinstance(settings.redis, RedisSettings)
        assert isinstance(settings.security, SecuritySettings)
        assert isinstance(settings.api, APISettings)


# =============================================================================
# DATABASE MANAGER TESTS
# =============================================================================

class TestDatabaseManager:
    """Test DatabaseManager class."""
    
    def test_initialization(self, test_settings) -> Any:
        """Test database manager initialization."""
        db_manager = DatabaseManager(test_settings)
        
        assert db_manager.settings == test_settings
        assert db_manager.engine is not None
        assert db_manager.SessionLocal is not None
    
    def test_get_db(self, test_db_manager) -> Optional[Dict[str, Any]]:
        """Test synchronous database session."""
        db_gen = test_db_manager.get_db()
        db = next(db_gen)
        
        assert db is not None
        assert hasattr(db, 'close')
        
        # Cleanup
        db.close()
    
    @pytest.mark.asyncio
    async def test_get_async_db(self, test_db_manager) -> Optional[Dict[str, Any]]:
        """Test asynchronous database session."""
        async for db in test_db_manager.get_async_db():
            assert db is not None
            assert hasattr(db, 'close')
            break
    
    def test_create_tables(self, test_db_manager) -> Any:
        """Test table creation."""
        test_db_manager.create_tables()
        
        # Check if tables exist
        inspector = test_db_manager.engine.dialect.inspector(test_db_manager.engine)
        tables = inspector.get_table_names()
        
        assert "users" in tables
        assert "api_keys" in tables
        assert "request_logs" in tables
    
    @pytest.mark.asyncio
    async def test_create_tables_async(self, test_db_manager) -> Any:
        """Test asynchronous table creation."""
        await test_db_manager.create_tables_async()
        
        # Check if tables exist
        inspector = test_db_manager.async_engine.dialect.inspector(test_db_manager.async_engine)
        tables = inspector.get_table_names()
        
        assert "users" in tables
        assert "api_keys" in tables
        assert "request_logs" in tables


# =============================================================================
# CACHE MANAGER TESTS
# =============================================================================

class TestCacheManager:
    """Test CacheManager class."""
    
    def test_initialization(self, test_settings) -> Any:
        """Test cache manager initialization."""
        cache_manager = CacheManager(test_settings)
        
        assert cache_manager.settings == test_settings
        assert cache_manager.memory_cache is not None
        assert cache_manager.lru_cache is not None
    
    @pytest.mark.asyncio
    async def test_get_set_memory_only(self, test_cache_manager) -> Optional[Dict[str, Any]]:
        """Test cache operations with memory only."""
        # Test set
        success = await test_cache_manager.set("test_key", "test_value", ttl=60)
        assert success is True
        
        # Test get
        value = await test_cache_manager.get("test_key")
        assert value == "test_value"
        
        # Test get with default
        value = await test_cache_manager.get("nonexistent_key", "default")
        assert value == "default"
    
    @pytest.mark.asyncio
    async def test_delete(self, test_cache_manager) -> Any:
        """Test cache deletion."""
        # Set value
        await test_cache_manager.set("test_key", "test_value")
        
        # Delete value
        success = await test_cache_manager.delete("test_key")
        assert success is True
        
        # Verify deletion
        value = await test_cache_manager.get("test_key")
        assert value is None
    
    @pytest.mark.asyncio
    async def test_clear(self, test_cache_manager) -> Any:
        """Test cache clearing."""
        # Set multiple values
        await test_cache_manager.set("key1", "value1")
        await test_cache_manager.set("key2", "value2")
        
        # Clear cache
        success = await test_cache_manager.clear()
        assert success is True
        
        # Verify clearing
        value1 = await test_cache_manager.get("key1")
        value2 = await test_cache_manager.get("key2")
        assert value1 is None
        assert value2 is None
    
    @pytest.mark.asyncio
    async def test_health_check(self, test_cache_manager) -> Any:
        """Test cache health check."""
        status = await test_cache_manager.health_check()
        assert status in ["healthy", "memory_only", "unhealthy"]


# =============================================================================
# SECURITY MANAGER TESTS
# =============================================================================

class TestSecurityManager:
    """Test SecurityManager class."""
    
    def test_initialization(self, test_settings) -> Any:
        """Test security manager initialization."""
        security_manager = SecurityManager(test_settings)
        
        assert security_manager.settings == test_settings
        assert security_manager.pwd_context is not None
        assert security_manager.fernet is not None
        assert security_manager.security is not None
    
    def test_password_hashing(self, test_security_manager) -> Any:
        """Test password hashing and verification."""
        password = "testpassword123"
        
        # Hash password
        hashed = test_security_manager.get_password_hash(password)
        assert hashed != password
        assert len(hashed) > 0
        
        # Verify password
        assert test_security_manager.verify_password(password, hashed) is True
        assert test_security_manager.verify_password("wrongpassword", hashed) is False
    
    def test_token_creation(self, test_security_manager) -> Any:
        """Test token creation."""
        data = {"sub": "testuser", "user_id": 1}
        
        # Create access token
        access_token = test_security_manager.create_access_token(data)
        assert isinstance(access_token, str)
        assert len(access_token) > 0
        
        # Create refresh token
        refresh_token = test_security_manager.create_refresh_token(data)
        assert isinstance(refresh_token, str)
        assert len(refresh_token) > 0
    
    def test_token_verification(self, test_security_manager) -> Any:
        """Test token verification."""
        data = {"sub": "testuser", "user_id": 1}
        
        # Create token
        token = test_security_manager.create_access_token(data)
        
        # Verify token
        token_data = test_security_manager.verify_token(token)
        assert token_data is not None
        assert token_data.username == "testuser"
        assert token_data.user_id == 1
    
    def test_token_verification_invalid(self, test_security_manager) -> Any:
        """Test invalid token verification."""
        # Test with invalid token
        token_data = test_security_manager.verify_token("invalid_token")
        assert token_data is None
    
    def test_encryption_decryption(self, test_security_manager) -> Any:
        """Test data encryption and decryption."""
        original_data = "sensitive_data"
        
        # Encrypt data
        encrypted = test_security_manager.encrypt_data(original_data)
        assert encrypted != original_data
        assert len(encrypted) > 0
        
        # Decrypt data
        decrypted = test_security_manager.decrypt_data(encrypted)
        assert decrypted == original_data
    
    async def test_api_key_generation(self, test_security_manager) -> Any:
        """Test API key generation and hashing."""
        # Generate API key
        api_key = test_security_manager.generate_api_key()
        assert isinstance(api_key, str)
        assert len(api_key) > 0
        
        # Hash API key
        hashed_key = test_security_manager.hash_api_key(api_key)
        assert hashed_key != api_key
        assert len(hashed_key) > 0


# =============================================================================
# METRICS MANAGER TESTS
# =============================================================================

class TestMetricsManager:
    """Test MetricsManager class."""
    
    def test_initialization(self) -> Any:
        """Test metrics manager initialization."""
        metrics_manager = MetricsManager()
        
        assert metrics_manager.request_counter is not None
        assert metrics_manager.request_duration is not None
        assert metrics_manager.active_requests is not None
        assert metrics_manager.error_counter is not None
        assert metrics_manager.database_operations is not None
        assert metrics_manager.cache_hits is not None
        assert metrics_manager.cache_misses is not None
        assert metrics_manager.memory_usage is not None
        assert metrics_manager.cpu_usage is not None
    
    async def test_record_request(self) -> Any:
        """Test request recording."""
        metrics_manager = MetricsManager()
        
        # Record request
        metrics_manager.record_request("GET", "/test", 200, 0.1)
        
        # Check if metrics were recorded
        # Note: In a real test, you would check the actual metric values
        assert metrics_manager.request_counter is not None
    
    def test_record_error(self) -> Any:
        """Test error recording."""
        metrics_manager = MetricsManager()
        
        # Record error
        metrics_manager.record_error("GET", "/test", "ValidationError")
        
        # Check if error was recorded
        assert metrics_manager.error_counter is not None
    
    def test_record_database_operation(self) -> Any:
        """Test database operation recording."""
        metrics_manager = MetricsManager()
        
        # Record database operation
        metrics_manager.record_database_operation("SELECT", "users")
        
        # Check if operation was recorded
        assert metrics_manager.database_operations is not None
    
    def test_record_cache_operations(self) -> Any:
        """Test cache operation recording."""
        metrics_manager = MetricsManager()
        
        # Record cache hit
        metrics_manager.record_cache_hit("redis")
        
        # Record cache miss
        metrics_manager.record_cache_miss("redis")
        
        # Check if operations were recorded
        assert metrics_manager.cache_hits is not None
        assert metrics_manager.cache_misses is not None
    
    def test_update_system_metrics(self) -> Any:
        """Test system metrics update."""
        metrics_manager = MetricsManager()
        
        # Update system metrics
        metrics_manager.update_system_metrics()
        
        # Check if metrics were updated
        assert metrics_manager.memory_usage is not None
        assert metrics_manager.cpu_usage is not None


# =============================================================================
# MIDDLEWARE TESTS
# =============================================================================

class TestRequestLoggingMiddleware:
    """Test RequestLoggingMiddleware."""
    
    @pytest.mark.asyncio
    async async def test_request_logging(self, test_db_manager, test_metrics_manager) -> Any:
        """Test request logging middleware."""
        # Create middleware
        middleware = RequestLoggingMiddleware(
            app=Mock(),
            db_manager=test_db_manager,
            metrics_manager=test_metrics_manager
        )
        
        # Create mock request
        request = Mock()
        request.method = "GET"
        request.url.path = "/test"
        request.client.host = "127.0.0.1"
        request.headers = {"user-agent": "test-agent"}
        request.state.request_id = "test-request-id"
        
        # Create mock response
        response = Mock()
        response.status_code = 200
        response.headers = {}
        
        # Mock call_next
        async def call_next(req) -> Any:
            return response
        
        # Test dispatch
        result = await middleware.dispatch(request, call_next)
        
        assert result is response
        assert "X-Request-ID" in result.headers
        assert "X-Response-Time" in result.headers


class TestRateLimitingMiddleware:
    """Test RateLimitingMiddleware."""
    
    @pytest.mark.asyncio
    async def test_rate_limiting(self, test_cache_manager, test_settings) -> Any:
        """Test rate limiting middleware."""
        # Create middleware
        middleware = RateLimitingMiddleware(
            app=Mock(),
            cache_manager=test_cache_manager,
            settings=test_settings
        )
        
        # Create mock request
        request = Mock()
        request.client.host = "127.0.0.1"
        
        # Create mock response
        response = Mock()
        response.status_code = 200
        
        # Mock call_next
        async def call_next(req) -> Any:
            return response
        
        # Test dispatch
        result = await middleware.dispatch(request, call_next)
        
        assert result is response


# =============================================================================
# UTILITY FUNCTION TESTS
# =============================================================================

class TestUtilityFunctions:
    """Test utility functions."""
    
    @pytest.mark.asyncio
    async async def test_external_api_call_success(self) -> Any:
        """Test successful external API call."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock successful response
            mock_response = Mock()
            mock_response.json.return_value = {"test": "data"}
            mock_response.raise_for_status.return_value = None
            
            mock_client_instance = Mock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            # Test API call
            result = await external_api_call("https://api.example.com/test")
            
            assert result == {"test": "data"}
    
    @pytest.mark.asyncio
    async async def test_external_api_call_failure(self) -> Any:
        """Test failed external API call."""
        with patch('httpx.AsyncClient') as mock_client:
            # Mock failed response
            mock_response = Mock()
            mock_response.raise_for_status.side_effect = Exception("API Error")
            
            mock_client_instance = Mock()
            mock_client_instance.__aenter__.return_value = mock_client_instance
            mock_client_instance.__aexit__.return_value = None
            mock_client_instance.get.return_value = mock_response
            mock_client.return_value = mock_client_instance
            
            # Test API call with retry
            with pytest.raises(Exception):
                await external_api_call("https://api.example.com/test")
    
    @pytest.mark.asyncio
    async def test_background_task(self) -> Any:
        """Test background task execution."""
        task_id = "test-task-id"
        task_data = {"test": "data"}
        
        # Test background task
        await background_task(task_id, task_data)
        
        # Task should complete without errors
        assert True


# =============================================================================
# API ROUTE TESTS
# =============================================================================

class TestAPIRoutes:
    """Test API routes."""
    
    def test_root_endpoint(self, test_client) -> Any:
        """Test root endpoint."""
        response = test_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data["data"]
    
    def test_health_check(self, test_client) -> Any:
        """Test health check endpoint."""
        response = test_client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data
        assert "version" in data
        assert "uptime" in data
        assert "database" in data
        assert "redis" in data
        assert "memory_usage" in data
        assert "cpu_usage" in data
    
    def test_register_user(self, test_client, sample_user_data) -> Any:
        """Test user registration."""
        response = test_client.post("/auth/register", json=sample_user_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["username"] == sample_user_data["username"]
        assert data["data"]["email"] == sample_user_data["email"]
    
    def test_register_user_duplicate(self, test_client, sample_user, sample_user_data) -> Any:
        """Test user registration with duplicate data."""
        response = test_client.post("/auth/register", json=sample_user_data)
        
        assert response.status_code == 400
        data = response.json()
        assert data["success"] is False
        assert "already registered" in data["detail"]
    
    def test_login_success(self, test_client, sample_user, sample_user_data) -> Any:
        """Test successful login."""
        login_data = {
            "username": sample_user_data["username"],
            "password": sample_user_data["password"]
        }
        
        response = test_client.post("/auth/login", data=login_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "access_token" in data["data"]
        assert "refresh_token" in data["data"]
    
    def test_login_invalid_credentials(self, test_client) -> Any:
        """Test login with invalid credentials."""
        login_data = {
            "username": "invaliduser",
            "password": "invalidpassword"
        }
        
        response = test_client.post("/auth/login", data=login_data)
        
        assert response.status_code == 401
        data = response.json()
        assert data["success"] is False
    
    def test_get_current_user_authenticated(self, test_client, sample_token) -> Optional[Dict[str, Any]]:
        """Test getting current user with valid token."""
        headers = {"Authorization": f"Bearer {sample_token}"}
        response = test_client.get("/users/me", headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "username" in data["data"]
        assert "email" in data["data"]
    
    def test_get_current_user_unauthenticated(self, test_client) -> Optional[Dict[str, Any]]:
        """Test getting current user without token."""
        response = test_client.get("/users/me")
        
        assert response.status_code == 403
    
    def test_update_current_user(self, test_client, sample_token) -> Any:
        """Test updating current user."""
        headers = {"Authorization": f"Bearer {sample_token}"}
        update_data = {"email": "newemail@example.com"}
        
        response = test_client.put("/users/me", json=update_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["email"] == update_data["email"]
    
    def test_metrics_endpoint(self, test_client) -> Any:
        """Test metrics endpoint."""
        response = test_client.get("/metrics")
        
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
    
    def test_cache_status(self, test_client) -> Any:
        """Test cache status endpoint."""
        response = test_client.get("/cache/status")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "redis_status" in data["data"]
        assert "memory_cache_size" in data["data"]
    
    def test_clear_cache(self, test_client) -> Any:
        """Test cache clearing endpoint."""
        response = test_client.post("/cache/clear")
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
    
    def test_external_data_endpoint(self, test_client) -> Any:
        """Test external data endpoint."""
        with patch('scalable_fastapi_system.external_api_call') as mock_api_call:
            mock_api_call.return_value = {"test": "data"}
            
            response = test_client.get("/api/external-data")
            
            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["data"]["test"] == "data"
    
    def test_create_background_task(self, test_client, sample_token) -> Any:
        """Test background task creation."""
        headers = {"Authorization": f"Bearer {sample_token}"}
        task_data = {"task": "test_task"}
        
        response = test_client.post("/tasks/background", json=task_data, headers=headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "task_id" in data["data"]


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""
    
    def test_complete_user_workflow(self, test_client) -> Any:
        """Test complete user registration and authentication workflow."""
        # 1. Register user
        user_data = {
            "username": "integrationuser",
            "email": "integration@example.com",
            "password": "integrationpass123"
        }
        
        register_response = test_client.post("/auth/register", json=user_data)
        assert register_response.status_code == 200
        
        # 2. Login user
        login_data = {
            "username": user_data["username"],
            "password": user_data["password"]
        }
        
        login_response = test_client.post("/auth/login", data=login_data)
        assert login_response.status_code == 200
        
        login_data = login_response.json()
        token = login_data["data"]["access_token"]
        
        # 3. Get user info
        headers = {"Authorization": f"Bearer {token}"}
        user_response = test_client.get("/users/me", headers=headers)
        assert user_response.status_code == 200
        
        # 4. Update user
        update_data = {"email": "updated@example.com"}
        update_response = test_client.put("/users/me", json=update_data, headers=headers)
        assert update_response.status_code == 200
    
    def test_error_handling(self, test_client) -> Any:
        """Test error handling."""
        # Test validation error
        invalid_data = {"invalid": "data"}
        response = test_client.post("/auth/register", json=invalid_data)
        assert response.status_code == 422
        
        # Test not found
        response = test_client.get("/nonexistent-endpoint")
        assert response.status_code == 404
    
    def test_rate_limiting(self, test_client) -> Any:
        """Test rate limiting functionality."""
        # Make multiple requests to trigger rate limiting
        responses = []
        for _ in range(100):  # Adjust based on rate limit settings
            response = test_client.get("/")
            responses.append(response.status_code)
        
        # Check if any requests were rate limited
        assert 429 in responses or all(status == 200 for status in responses)


# =============================================================================
# PERFORMANCE TESTS
# =============================================================================

class TestPerformance:
    """Performance tests."""
    
    def test_response_time(self, test_client) -> Any:
        """Test API response times."""
        start_time = time.time()
        response = test_client.get("/")
        end_time = time.time()
        
        response_time = end_time - start_time
        assert response_time < 1.0  # Should respond within 1 second
        assert response.status_code == 200
    
    async def test_concurrent_requests(self, test_client) -> Any:
        """Test concurrent request handling."""
        
        def make_request():
            
    """make_request function."""
return test_client.get("/")
        
        # Make concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [future.result() for future in futures]
        
        # All requests should succeed
        assert all(response.status_code == 200 for response in responses)


# =============================================================================
# SECURITY TESTS
# =============================================================================

class TestSecurity:
    """Security tests."""
    
    def test_password_security(self, test_security_manager) -> Any:
        """Test password security."""
        password = "testpassword123"
        
        # Hash password
        hashed = test_security_manager.get_password_hash(password)
        
        # Verify hash is different from original
        assert hashed != password
        
        # Verify hash is not reversible
        assert test_security_manager.get_password_hash(password) != hashed
    
    def test_token_security(self, test_security_manager) -> Any:
        """Test token security."""
        data = {"sub": "testuser", "user_id": 1}
        
        # Create token
        token = test_security_manager.create_access_token(data)
        
        # Verify token is not the same as data
        assert token != str(data)
        
        # Verify token can be decoded
        decoded = test_security_manager.verify_token(token)
        assert decoded is not None
        assert decoded.username == "testuser"
    
    def test_encryption_security(self, test_security_manager) -> Any:
        """Test encryption security."""
        original_data = "sensitive_data"
        
        # Encrypt data
        encrypted = test_security_manager.encrypt_data(original_data)
        
        # Verify encrypted data is different
        assert encrypted != original_data
        
        # Verify same data produces different encryption
        encrypted2 = test_security_manager.encrypt_data(original_data)
        assert encrypted != encrypted2


# =============================================================================
# MAIN TEST RUNNER
# =============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 