from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
MAX_RETRIES = 100

# Constants
TIMEOUT_SECONDS = 60

import asyncio
import pytest
import json
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
import aiohttp
import httpx
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient
import structlog
from dependencies.shared_resources import (
from typing import Any, List, Dict, Optional
import logging
"""
Tests for Shared Resources Dependency Injection System

This module provides comprehensive tests for the shared resources system including:
- Network session managers (HTTP, WebSocket)
- Cryptographic backend managers
- Database and Redis pool managers
- Health checks and monitoring
- Resource lifecycle management
- FastAPI dependency injection
- Context managers
- Error handling and edge cases
"""



# Import shared resources
    SharedResourceConfig,
    ResourceConfig,
    CryptoConfig,
    ResourceType,
    CryptoAlgorithm,
    NetworkProtocol,
    ResourceHealth,
    ResourceMetrics,
    BaseResourceManager,
    HTTPSessionManager,
    WebSocketSessionManager,
    CryptoBackendManager,
    DatabasePoolManager,
    RedisPoolManager,
    SharedResourcesContainer,
    get_http_session,
    get_websocket_session,
    get_crypto_backend,
    get_database_pool,
    get_redis_pool,
    http_session_context,
    crypto_backend_context,
    initialize_shared_resources,
    shutdown_shared_resources,
    get_resource_health,
    get_all_resource_health
)

# Configure logging
logger = structlog.get_logger(__name__)

# =============================================================================
# Test Configuration
# =============================================================================

@pytest.fixture
def test_config() -> SharedResourceConfig:
    """Create test configuration for shared resources."""
    return SharedResourceConfig(
        resources={
            "http_session": ResourceConfig(
                name="http_session",
                resource_type=ResourceType.HTTP_SESSION,
                max_connections=10,
                timeout=5.0,
                keepalive_timeout=10.0
            ),
            "websocket_session": ResourceConfig(
                name="websocket_session",
                resource_type=ResourceType.WEBSOCKET_SESSION,
                max_connections=5,
                timeout=5.0
            )
        },
        crypto_configs={
            "test_aes": CryptoConfig(
                algorithm=CryptoAlgorithm.AES_256_GCM,
                key_size=256
            ),
            "test_rsa": CryptoConfig(
                algorithm=CryptoAlgorithm.RSA_2048,
                key_size=2048
            ),
            "test_hash": CryptoConfig(
                algorithm=CryptoAlgorithm.SHA_256,
                key_size=256
            )
        },
        global_timeout=10.0,
        global_max_retries=2,
        enable_monitoring=True,
        enable_health_checks=True,
        resource_cleanup_interval=60.0
    )

@pytest.fixture
async def shared_resources(test_config) -> Any:
    """Create and initialize shared resources for testing."""
    container = SharedResourcesContainer(test_config)
    await container.initialize()
    yield container
    await container.shutdown()

# =============================================================================
# Test Base Resource Manager
# =============================================================================

class TestBaseResourceManager:
    """Test base resource manager functionality."""
    
    def test_base_resource_manager_initialization(self) -> Any:
        """Test base resource manager initialization."""
        config = ResourceConfig(
            name="test",
            resource_type=ResourceType.HTTP_SESSION
        )
        manager = BaseResourceManager(config)
        
        assert manager.config == config
        assert manager.health.is_healthy is True
        assert isinstance(manager.health.last_check, datetime)
        assert manager.health.response_time == 0.0
        assert manager.metrics.total_requests == 0
        assert manager.is_available() is True
    
    def test_base_resource_manager_health_update(self) -> Any:
        """Test health status update."""
        config = ResourceConfig(
            name="test",
            resource_type=ResourceType.HTTP_SESSION
        )
        manager = BaseResourceManager(config)
        
        # Update health status
        new_health = ResourceHealth(
            is_healthy=False,
            last_check=datetime.utcnow(),
            response_time=1.5,
            error_count=1,
            last_error="Test error"
        )
        manager.health = new_health
        
        assert manager.health.is_healthy is False
        assert manager.health.response_time == 1.5
        assert manager.health.error_count == 1
        assert manager.health.last_error == "Test error"
        assert manager.is_available() is False
    
    def test_base_resource_manager_metrics_update(self) -> Any:
        """Test metrics update."""
        config = ResourceConfig(
            name="test",
            resource_type=ResourceType.HTTP_SESSION
        )
        manager = BaseResourceManager(config)
        
        # Update metrics
        manager.metrics.total_requests = 100
        manager.metrics.successful_requests = 95
        manager.metrics.failed_requests = 5
        manager.metrics.average_response_time = 0.5
        manager.metrics.active_connections = 10
        manager.metrics.peak_connections = 20
        
        assert manager.metrics.total_requests == 100
        assert manager.metrics.successful_requests == 95
        assert manager.metrics.failed_requests == 5
        assert manager.metrics.average_response_time == 0.5
        assert manager.metrics.active_connections == 10
        assert manager.metrics.peak_connections == 20

# =============================================================================
# Test HTTP Session Manager
# =============================================================================

class TestHTTPSessionManager:
    """Test HTTP session manager functionality."""
    
    @pytest.fixture
    async def http_config(self) -> Any:
        """Create HTTP session configuration."""
        return ResourceConfig(
            name="test_http",
            resource_type=ResourceType.HTTP_SESSION,
            max_connections=10,
            timeout=5.0,
            keepalive_timeout=10.0
        )
    
    @pytest.fixture
    async def http_manager(self, http_config) -> Any:
        """Create HTTP session manager."""
        return HTTPSessionManager(http_config)
    
    @pytest.mark.asyncio
    async async def test_http_session_manager_initialization(self, http_manager) -> Any:
        """Test HTTP session manager initialization."""
        assert http_manager._session is None
        assert http_manager._connector is None
        assert http_manager._timeout is None
    
    @pytest.mark.asyncio
    async async def test_http_session_initialization(self, http_manager) -> Any:
        """Test HTTP session initialization."""
        await http_manager._initialize_session()
        
        assert http_manager._session is not None
        assert http_manager._connector is not None
        assert http_manager._timeout is not None
        assert isinstance(http_manager._session, aiohttp.ClientSession)
        assert isinstance(http_manager._connector, aiohttp.TCPConnector)
        assert isinstance(http_manager._timeout, aiohttp.ClientTimeout)
    
    @pytest.mark.asyncio
    async def test_get_session(self, http_manager) -> Optional[Dict[str, Any]]:
        """Test getting HTTP session."""
        session = await http_manager.get_session()
        
        assert session is not None
        assert isinstance(session, aiohttp.ClientSession)
        assert http_manager._session is session
    
    @pytest.mark.asyncio
    async async def test_http_session_cleanup(self, http_manager) -> Any:
        """Test HTTP session cleanup."""
        # Initialize session
        await http_manager._initialize_session()
        assert http_manager._session is not None
        
        # Cleanup
        await http_manager.cleanup()
        assert http_manager._session is None
        assert http_manager._connector is None
        assert http_manager._timeout is None
    
    @pytest.mark.asyncio
    async async def test_http_health_check_success(self, http_manager) -> Any:
        """Test successful HTTP health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_get.return_value.__aenter__.return_value = mock_response
            
            health = await http_manager.health_check()
            
            assert health.is_healthy is True
            assert health.response_time > 0
            assert health.error_count == 0
    
    @pytest.mark.asyncio
    async async def test_http_health_check_failure(self, http_manager) -> Any:
        """Test failed HTTP health check."""
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = Exception("Connection failed")
            
            health = await http_manager.health_check()
            
            assert health.is_healthy is False
            assert health.error_count == 1
            assert "Connection failed" in health.last_error

# =============================================================================
# Test Crypto Backend Manager
# =============================================================================

class TestCryptoBackendManager:
    """Test crypto backend manager functionality."""
    
    @pytest.fixture
    def aes_config(self) -> Any:
        """Create AES crypto configuration."""
        return CryptoConfig(
            algorithm=CryptoAlgorithm.AES_256_GCM,
            key_size=256
        )
    
    @pytest.fixture
    def rsa_config(self) -> Any:
        """Create RSA crypto configuration."""
        return CryptoConfig(
            algorithm=CryptoAlgorithm.RSA_2048,
            key_size=2048
        )
    
    @pytest.fixture
    def hash_config(self) -> Any:
        """Create hash crypto configuration."""
        return CryptoConfig(
            algorithm=CryptoAlgorithm.SHA_256,
            key_size=256
        )
    
    @pytest.fixture
    def aes_manager(self, aes_config) -> Any:
        """Create AES crypto backend manager."""
        return CryptoBackendManager(aes_config)
    
    @pytest.fixture
    def rsa_manager(self, rsa_config) -> Any:
        """Create RSA crypto backend manager."""
        return CryptoBackendManager(rsa_config)
    
    @pytest.fixture
    def hash_manager(self, hash_config) -> Any:
        """Create hash crypto backend manager."""
        return CryptoBackendManager(hash_config)
    
    @pytest.mark.asyncio
    async def test_crypto_backend_initialization(self, aes_manager) -> Any:
        """Test crypto backend initialization."""
        assert aes_manager._private_key is None
        assert aes_manager._public_key is None
        assert aes_manager._symmetric_key is None
        assert aes_manager._key_generated_at is None
    
    @pytest.mark.asyncio
    async def test_aes_encryption_decryption(self, aes_manager) -> Any:
        """Test AES encryption and decryption."""
        test_data = b"Hello, Crypto Backend!"
        
        # Encrypt
        encrypted = await aes_manager.encrypt(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt
        decrypted = await aes_manager.decrypt(encrypted)
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_rsa_encryption_decryption(self, rsa_manager) -> Any:
        """Test RSA encryption and decryption."""
        test_data = b"Hello, RSA!"
        
        # Encrypt
        encrypted = await rsa_manager.encrypt(test_data)
        assert encrypted != test_data
        assert len(encrypted) > len(test_data)
        
        # Decrypt
        decrypted = await rsa_manager.decrypt(encrypted)
        assert decrypted == test_data
    
    @pytest.mark.asyncio
    async def test_hashing(self, hash_manager) -> Any:
        """Test hashing operations."""
        test_data = b"Hello, Hashing!"
        
        # Hash
        hashed = await hash_manager.hash(test_data)
        assert hashed != test_data
        assert len(hashed) == 32  # SHA-256 produces 32 bytes
        
        # Hash same data again should produce same result
        hashed2 = await hash_manager.hash(test_data)
        assert hashed == hashed2
    
    @pytest.mark.asyncio
    async def test_rsa_signing_verification(self, rsa_manager) -> Any:
        """Test RSA signing and verification."""
        test_data = b"Hello, Signing!"
        
        # Sign
        signature = await rsa_manager.sign(test_data)
        assert signature != test_data
        assert len(signature) > 0
        
        # Verify
        is_valid = await rsa_manager.verify(test_data, signature)
        assert is_valid is True
        
        # Verify with wrong data
        wrong_data = b"Wrong data!"
        is_valid = await rsa_manager.verify(wrong_data, signature)
        assert is_valid is False
    
    @pytest.mark.asyncio
    async def test_crypto_health_check(self, aes_manager) -> Any:
        """Test crypto backend health check."""
        health = await aes_manager.health_check()
        
        assert health.is_healthy is True
        assert health.response_time > 0
        assert health.error_count == 0
    
    @pytest.mark.asyncio
    async def test_crypto_cleanup(self, aes_manager) -> Any:
        """Test crypto backend cleanup."""
        # Generate keys first
        await aes_manager._generate_keys()
        assert aes_manager._private_key is not None
        
        # Cleanup
        await aes_manager.cleanup()
        assert aes_manager._private_key is None
        assert aes_manager._public_key is None
        assert aes_manager._symmetric_key is None

# =============================================================================
# Test Shared Resources Container
# =============================================================================

class TestSharedResourcesContainer:
    """Test shared resources container functionality."""
    
    @pytest.mark.asyncio
    async def test_container_initialization(self, test_config) -> Any:
        """Test container initialization."""
        container = SharedResourcesContainer(test_config)
        await container.initialize()
        
        assert len(container._resources) > 0
        assert len(container._crypto_backends) > 0
        assert container._health_check_task is not None
        assert container._cleanup_task is not None
        
        await container.shutdown()
    
    @pytest.mark.asyncio
    async def test_get_resource(self, shared_resources) -> Optional[Dict[str, Any]]:
        """Test getting resource by name."""
        http_manager = await shared_resources.get_resource("http_session")
        assert isinstance(http_manager, HTTPSessionManager)
        
        ws_manager = await shared_resources.get_resource("websocket_session")
        assert isinstance(ws_manager, WebSocketSessionManager)
    
    @pytest.mark.asyncio
    async def test_get_crypto_backend(self, shared_resources) -> Optional[Dict[str, Any]]:
        """Test getting crypto backend by name."""
        aes_backend = await shared_resources.get_crypto_backend("test_aes")
        assert isinstance(aes_backend, CryptoBackendManager)
        assert aes_backend.crypto_config.algorithm == CryptoAlgorithm.AES_256_GCM
        
        rsa_backend = await shared_resources.get_crypto_backend("test_rsa")
        assert isinstance(rsa_backend, CryptoBackendManager)
        assert rsa_backend.crypto_config.algorithm == CryptoAlgorithm.RSA_2048
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_resource(self, shared_resources) -> Optional[Dict[str, Any]]:
        """Test getting non-existent resource."""
        with pytest.raises(ValueError, match="Resource 'nonexistent' not found"):
            await shared_resources.get_resource("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_nonexistent_crypto_backend(self, shared_resources) -> Optional[Dict[str, Any]]:
        """Test getting non-existent crypto backend."""
        with pytest.raises(ValueError, match="Crypto backend 'nonexistent' not found"):
            await shared_resources.get_crypto_backend("nonexistent")
    
    @pytest.mark.asyncio
    async def test_container_shutdown(self, test_config) -> Any:
        """Test container shutdown."""
        container = SharedResourcesContainer(test_config)
        await container.initialize()
        
        # Verify resources are initialized
        assert len(container._resources) > 0
        
        # Shutdown
        await container.shutdown()
        
        # Verify tasks are cancelled
        assert container._health_check_task.cancelled()
        assert container._cleanup_task.cancelled()

# =============================================================================
# Test FastAPI Dependencies
# =============================================================================

class TestFastAPIDependencies:
    """Test FastAPI dependency injection."""
    
    @pytest.fixture
    def app(self) -> Any:
        """Create FastAPI test app."""
        app = FastAPI()
        
        @app.get("/test-http")
        async def test_http(http_session=Depends(get_http_session)):
            return {"session_type": type(http_session).__name__}
        
        @app.get("/test-crypto")
        async def test_crypto(crypto_backend=Depends(get_crypto_backend)):
            return {"backend_type": type(crypto_backend).__name__}
        
        @app.get("/test-crypto-named")
        async def test_crypto_named(crypto_backend=Depends(lambda: get_crypto_backend("test_aes"))):
            return {"backend_type": type(crypto_backend).__name__}
        
        return app
    
    @pytest.mark.asyncio
    async async def test_http_session_dependency(self, app, test_config) -> Any:
        """Test HTTP session dependency injection."""
        await initialize_shared_resources(test_config)
        
        try:
            client = TestClient(app)
            response = client.get("/test-http")
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_type"] == "ClientSession"
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_crypto_backend_dependency(self, app, test_config) -> Any:
        """Test crypto backend dependency injection."""
        await initialize_shared_resources(test_config)
        
        try:
            client = TestClient(app)
            response = client.get("/test-crypto")
            
            assert response.status_code == 200
            data = response.json()
            assert data["backend_type"] == "CryptoBackendManager"
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_named_crypto_backend_dependency(self, app, test_config) -> Any:
        """Test named crypto backend dependency injection."""
        await initialize_shared_resources(test_config)
        
        try:
            client = TestClient(app)
            response = client.get("/test-crypto-named")
            
            assert response.status_code == 200
            data = response.json()
            assert data["backend_type"] == "CryptoBackendManager"
        
        finally:
            await shutdown_shared_resources()

# =============================================================================
# Test Context Managers
# =============================================================================

class TestContextManagers:
    """Test context managers for resource management."""
    
    @pytest.mark.asyncio
    async async def test_http_session_context(self, test_config) -> Any:
        """Test HTTP session context manager."""
        await initialize_shared_resources(test_config)
        
        try:
            async with http_session_context() as session:
                assert session is not None
                assert isinstance(session, aiohttp.ClientSession)
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_crypto_backend_context(self, test_config) -> Any:
        """Test crypto backend context manager."""
        await initialize_shared_resources(test_config)
        
        try:
            async with crypto_backend_context() as backend:
                assert backend is not None
                assert isinstance(backend, CryptoBackendManager)
            
            # Test with named backend
            async with crypto_backend_context("test_aes") as backend:
                assert backend is not None
                assert isinstance(backend, CryptoBackendManager)
                assert backend.crypto_config.algorithm == CryptoAlgorithm.AES_256_GCM
        
        finally:
            await shutdown_shared_resources()

# =============================================================================
# Test Health Monitoring
# =============================================================================

class TestHealthMonitoring:
    """Test health monitoring functionality."""
    
    @pytest.mark.asyncio
    async def test_get_resource_health(self, test_config) -> Optional[Dict[str, Any]]:
        """Test getting resource health."""
        await initialize_shared_resources(test_config)
        
        try:
            # Wait for health checks to run
            await asyncio.sleep(1)
            
            health = get_resource_health("http_session")
            assert health is not None
            assert isinstance(health, ResourceHealth)
            
            # Test non-existent resource
            health = get_resource_health("nonexistent")
            assert health is None
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_get_all_resource_health(self, test_config) -> Optional[Dict[str, Any]]:
        """Test getting all resource health."""
        await initialize_shared_resources(test_config)
        
        try:
            # Wait for health checks to run
            await asyncio.sleep(1)
            
            all_health = get_all_resource_health()
            assert isinstance(all_health, dict)
            assert len(all_health) > 0
            
            for name, health in all_health.items():
                assert isinstance(health, ResourceHealth)
                assert isinstance(name, str)
        
        finally:
            await shutdown_shared_resources()

# =============================================================================
# Test Error Handling
# =============================================================================

class TestErrorHandling:
    """Test error handling scenarios."""
    
    @pytest.mark.asyncio
    async async def test_http_session_error_handling(self, http_config) -> Any:
        """Test HTTP session error handling."""
        manager = HTTPSessionManager(http_config)
        
        # Test health check with network error
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.side_effect = aiohttp.ClientError("Network error")
            
            health = await manager.health_check()
            assert health.is_healthy is False
            assert health.error_count == 1
            assert "Network error" in health.last_error
    
    @pytest.mark.asyncio
    async def test_crypto_backend_error_handling(self, aes_config) -> Any:
        """Test crypto backend error handling."""
        manager = CryptoBackendManager(aes_config)
        
        # Test encryption with invalid data
        with pytest.raises(Exception):
            await manager.encrypt(None)
    
    @pytest.mark.asyncio
    async def test_container_error_handling(self, test_config) -> Any:
        """Test container error handling."""
        container = SharedResourcesContainer(test_config)
        
        # Test getting non-existent resource
        with pytest.raises(ValueError):
            await container.get_resource("nonexistent")
        
        # Test getting non-existent crypto backend
        with pytest.raises(ValueError):
            await container.get_crypto_backend("nonexistent")

# =============================================================================
# Test Configuration Validation
# =============================================================================

class TestConfigurationValidation:
    """Test configuration validation."""
    
    def test_valid_configuration(self) -> Any:
        """Test valid configuration."""
        config = SharedResourceConfig(
            resources={
                "test": ResourceConfig(
                    name="test",
                    resource_type=ResourceType.HTTP_SESSION
                )
            },
            crypto_configs={
                "test": CryptoConfig(
                    algorithm=CryptoAlgorithm.AES_256_GCM
                )
            }
        )
        
        assert config.resources["test"].name == "test"
        assert config.crypto_configs["test"].algorithm == CryptoAlgorithm.AES_256_GCM
    
    def test_invalid_resource_config(self) -> Any:
        """Test invalid resource configuration."""
        with pytest.raises(ValueError):
            SharedResourceConfig(
                resources={
                    "test": "invalid_config"
                }
            )
    
    def test_resource_config_validation(self) -> Any:
        """Test resource configuration validation."""
        config = ResourceConfig(
            name="test",
            resource_type=ResourceType.HTTP_SESSION,
            max_connections=100,
            timeout=30.0
        )
        
        assert config.name == "test"
        assert config.resource_type == ResourceType.HTTP_SESSION
        assert config.max_connections == 100
        assert config.timeout == 30.0
    
    def test_crypto_config_validation(self) -> Any:
        """Test crypto configuration validation."""
        config = CryptoConfig(
            algorithm=CryptoAlgorithm.AES_256_GCM,
            key_size=256,
            salt_length=32
        )
        
        assert config.algorithm == CryptoAlgorithm.AES_256_GCM
        assert config.key_size == 256
        assert config.salt_length == 32

# =============================================================================
# Test Performance and Concurrency
# =============================================================================

class TestPerformanceAndConcurrency:
    """Test performance and concurrency scenarios."""
    
    @pytest.mark.asyncio
    async async def test_concurrent_http_sessions(self, test_config) -> Any:
        """Test concurrent HTTP session usage."""
        await initialize_shared_resources(test_config)
        
        try:
            async def fetch_data(session_id: int):
                
    """fetch_data function."""
async with http_session_context() as session:
                    # Simulate some work
                    await asyncio.sleep(0.1)
                    return f"session_{session_id}"
            
            # Run multiple concurrent sessions
            tasks = [fetch_data(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(result.startswith("session_") for result in results)
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_concurrent_crypto_operations(self, test_config) -> Any:
        """Test concurrent crypto operations."""
        await initialize_shared_resources(test_config)
        
        try:
            async def crypto_operation(data_id: int):
                
    """crypto_operation function."""
async with crypto_backend_context() as backend:
                    test_data = f"data_{data_id}".encode('utf-8')
                    encrypted = await backend.encrypt(test_data)
                    decrypted = await backend.decrypt(encrypted)
                    return decrypted == test_data
            
            # Run multiple concurrent crypto operations
            tasks = [crypto_operation(i) for i in range(5)]
            results = await asyncio.gather(*tasks)
            
            assert len(results) == 5
            assert all(result is True for result in results)
        
        finally:
            await shutdown_shared_resources()

# =============================================================================
# Test Integration Scenarios
# =============================================================================

class TestIntegrationScenarios:
    """Test integration scenarios."""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, test_config) -> Any:
        """Test complete workflow with multiple resources."""
        await initialize_shared_resources(test_config)
        
        try:
            # Use HTTP session
            async with http_session_context() as session:
                assert session is not None
            
            # Use crypto backend
            async with crypto_backend_context() as backend:
                test_data = b"integration test"
                encrypted = await backend.encrypt(test_data)
                decrypted = await backend.decrypt(encrypted)
                assert decrypted == test_data
            
            # Check health
            health = get_all_resource_health()
            assert len(health) > 0
        
        finally:
            await shutdown_shared_resources()
    
    @pytest.mark.asyncio
    async def test_resource_lifecycle(self, test_config) -> Any:
        """Test complete resource lifecycle."""
        # Initialize
        await initialize_shared_resources(test_config)
        
        # Use resources
        async with http_session_context() as session:
            assert session is not None
        
        # Check health
        health = get_all_resource_health()
        assert len(health) > 0
        
        # Shutdown
        await shutdown_shared_resources()
        
        # Verify shutdown
        health = get_all_resource_health()
        assert len(health) == 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 