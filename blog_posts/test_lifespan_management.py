from typing_extensions import Literal, TypedDict
from typing import Any, List, Dict, Optional, Union, Tuple
# Constants
TIMEOUT_SECONDS = 60

import pytest
import asyncio
import signal
import sys
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from typing import Dict, Any, List
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from .lifespan_management import (
from typing import Any, List, Dict, Optional
import logging
"""
🧪 Test Suite for FastAPI Lifespan Management
============================================

Comprehensive tests for:
- Lifespan context manager functionality
- Resource initialization and cleanup
- Database connection management
- Redis client management
- Cache manager operations
- Background task management
- Health checks
- Signal handling
- Application state management
"""



    # Core classes
    AppState,
    HealthStatus,
    DatabaseConfig,
    RedisConfig,
    AppConfig,
    CacheManager,
    
    # Functions
    load_config,
    lifespan,
    initialize_database_pool,
    close_database_pool,
    initialize_redis_client,
    close_redis_client,
    initialize_cache_manager,
    close_cache_manager,
    health_monitor_task,
    metrics_collector_task,
    start_background_tasks,
    stop_background_tasks,
    check_database_health,
    check_redis_health,
    check_cache_health,
    perform_health_check,
    collect_system_metrics,
    setup_signal_handlers,
    setup_logging,
    create_app,
    setup_middleware,
    setup_routes,
    setup_exception_handlers,
    get_app_state,
    get_database_session,
    get_cache_manager
)

# ============================================================================
# Test Data
# ============================================================================

SAMPLE_CONFIG = AppConfig(
    app_name="Test API",
    version="1.0.0",
    debug=True,
    database=DatabaseConfig(
        url="postgresql+asyncpg://test:test@localhost/testdb",
        pool_size=5,
        max_overflow=10
    ),
    redis=RedisConfig(
        url="redis://localhost:6379",
        max_connections=10
    ),
    cors_origins=["http://localhost:3000"],
    trusted_hosts=["localhost"],
    log_level="DEBUG"
)

# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Test configuration management."""
    
    def test_load_config(self) -> Any:
        """Test configuration loading."""
        config = load_config()
        
        assert isinstance(config, AppConfig)
        assert config.app_name == "Blatam Academy NLP API"
        assert config.version == "1.0.0"
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
    
    def test_database_config(self) -> Any:
        """Test database configuration."""
        db_config = DatabaseConfig(
            url="postgresql+asyncpg://user:pass@localhost/db",
            pool_size=20,
            max_overflow=30
        )
        
        assert db_config.url == "postgresql+asyncpg://user:pass@localhost/db"
        assert db_config.pool_size == 20
        assert db_config.max_overflow == 30
        assert db_config.pool_pre_ping is True
        assert db_config.echo is False
    
    def test_redis_config(self) -> Any:
        """Test Redis configuration."""
        redis_config = RedisConfig(
            url="redis://localhost:6379",
            max_connections=20
        )
        
        assert redis_config.url == "redis://localhost:6379"
        assert redis_config.max_connections == 20
        assert redis_config.decode_responses is True
        assert redis_config.health_check_interval == 30
    
    def test_app_config(self) -> Any:
        """Test application configuration."""
        config = SAMPLE_CONFIG
        
        assert config.app_name == "Test API"
        assert config.version == "1.0.0"
        assert config.debug is True
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.workers == 1
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.redis, RedisConfig)
        assert config.cors_origins == ["http://localhost:3000"]
        assert config.trusted_hosts == ["localhost"]
        assert config.log_level == "DEBUG"

# ============================================================================
# AppState Tests
# ============================================================================

class TestAppState:
    """Test application state management."""
    
    def test_app_state_initialization(self) -> Any:
        """Test AppState initialization."""
        app_state = AppState()
        
        assert isinstance(app_state.startup_time, datetime)
        assert app_state.is_healthy is True
        assert app_state.database_pool is None
        assert app_state.redis_client is None
        assert app_state.cache_manager is None
        assert isinstance(app_state.background_tasks, list)
        assert app_state.shutdown_event is None
        assert isinstance(app_state.config, dict)
        assert isinstance(app_state.metrics, dict)
    
    def test_app_state_with_config(self) -> Any:
        """Test AppState with configuration."""
        config = SAMPLE_CONFIG
        app_state = AppState(
            config=config.__dict__,
            shutdown_event=asyncio.Event()
        )
        
        assert app_state.config == config.__dict__
        assert isinstance(app_state.shutdown_event, asyncio.Event)
    
    def test_app_state_health_status(self) -> Any:
        """Test AppState health status management."""
        app_state = AppState()
        
        # Initially healthy
        assert app_state.is_healthy is True
        
        # Set to unhealthy
        app_state.is_healthy = False
        assert app_state.is_healthy is False
        
        # Set back to healthy
        app_state.is_healthy = True
        assert app_state.is_healthy is True

# ============================================================================
# Database Management Tests
# ============================================================================

class TestDatabaseManagement:
    """Test database management functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_database_pool_success(self) -> Any:
        """Test successful database pool initialization."""
        config = DatabaseConfig(
            url="postgresql+asyncpg://test:test@localhost/testdb",
            pool_size=5,
            max_overflow=10
        )
        
        with patch('lifespan_management.create_async_engine') as mock_create_engine:
            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine
            
            with patch('lifespan_management.async_sessionmaker') as mock_session_maker:
                mock_session_maker_instance = Mock()
                mock_session_maker.return_value = mock_session_maker_instance
                
                # Mock session context manager
                mock_session = AsyncMock()
                mock_session_maker_instance.return_value.__aenter__.return_value = mock_session
                mock_session_maker_instance.return_value.__aexit__.return_value = None
                
                result = await initialize_database_pool(config)
                
                assert result == mock_session_maker_instance
                mock_create_engine.assert_called_once()
                mock_session_maker.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_database_pool_failure(self) -> Any:
        """Test database pool initialization failure."""
        config = DatabaseConfig(
            url="invalid://url",
            pool_size=5,
            max_overflow=10
        )
        
        with pytest.raises(Exception):
            await initialize_database_pool(config)
    
    @pytest.mark.asyncio
    async def test_close_database_pool(self) -> Any:
        """Test database pool closure."""
        mock_session_maker = Mock()
        mock_engine = Mock()
        mock_session_maker.kw = {"bind": mock_engine}
        
        await close_database_pool(mock_session_maker)
        
        mock_session_maker.close_all.assert_called_once()
        mock_engine.dispose.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_database_health_success(self) -> Any:
        """Test successful database health check."""
        mock_session_maker = Mock()
        mock_session = AsyncMock()
        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_session_maker.return_value.__aexit__.return_value = None
        
        result = await check_database_health(mock_session_maker)
        
        assert result is True
        mock_session.execute.assert_called_once_with("SELECT 1")
    
    @pytest.mark.asyncio
    async def test_check_database_health_failure(self) -> Any:
        """Test database health check failure."""
        mock_session_maker = Mock()
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database error")
        mock_session_maker.return_value.__aenter__.return_value = mock_session
        mock_session_maker.return_value.__aexit__.return_value = None
        
        result = await check_database_health(mock_session_maker)
        
        assert result is False

# ============================================================================
# Redis Management Tests
# ============================================================================

class TestRedisManagement:
    """Test Redis management functionality."""
    
    @pytest.mark.asyncio
    async def test_initialize_redis_client_success(self) -> Any:
        """Test successful Redis client initialization."""
        config = RedisConfig(
            url="redis://localhost:6379",
            max_connections=10
        )
        
        with patch('lifespan_management.redis.from_url') as mock_from_url:
            mock_client = AsyncMock()
            mock_from_url.return_value = mock_client
            
            result = await initialize_redis_client(config)
            
            assert result == mock_client
            mock_from_url.assert_called_once_with(
                config.url,
                max_connections=config.max_connections,
                decode_responses=config.decode_responses,
                health_check_interval=config.health_check_interval
            )
            mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_initialize_redis_client_failure(self) -> Any:
        """Test Redis client initialization failure."""
        config = RedisConfig(
            url="redis://invalid:6379",
            max_connections=10
        )
        
        with patch('lifespan_management.redis.from_url') as mock_from_url:
            mock_client = AsyncMock()
            mock_client.ping.side_effect = Exception("Connection failed")
            mock_from_url.return_value = mock_client
            
            with pytest.raises(Exception):
                await initialize_redis_client(config)
    
    @pytest.mark.asyncio
    async def test_close_redis_client(self) -> Any:
        """Test Redis client closure."""
        mock_client = AsyncMock()
        
        await close_redis_client(mock_client)
        
        mock_client.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_redis_health_success(self) -> Any:
        """Test successful Redis health check."""
        mock_client = AsyncMock()
        
        result = await check_redis_health(mock_client)
        
        assert result is True
        mock_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_check_redis_health_failure(self) -> Any:
        """Test Redis health check failure."""
        mock_client = AsyncMock()
        mock_client.ping.side_effect = Exception("Redis error")
        
        result = await check_redis_health(mock_client)
        
        assert result is False

# ============================================================================
# Cache Manager Tests
# ============================================================================

class TestCacheManager:
    """Test cache manager functionality."""
    
    @pytest.mark.asyncio
    async def test_cache_manager_initialization(self) -> Any:
        """Test cache manager initialization."""
        mock_redis_client = AsyncMock()
        cache_manager = CacheManager(mock_redis_client)
        
        assert cache_manager.redis == mock_redis_client
        assert cache_manager.logger is not None
    
    @pytest.mark.asyncio
    async def test_cache_get_success(self) -> Optional[Dict[str, Any]]:
        """Test successful cache get operation."""
        mock_redis_client = AsyncMock()
        mock_redis_client.get.return_value = "test_value"
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.get("test_key")
        
        assert result == "test_value"
        mock_redis_client.get.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_get_failure(self) -> Optional[Dict[str, Any]]:
        """Test cache get operation failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.get.side_effect = Exception("Redis error")
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.get("test_key")
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_cache_set_success(self) -> Any:
        """Test successful cache set operation."""
        mock_redis_client = AsyncMock()
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.set("test_key", "test_value", ttl=3600)
        
        assert result is True
        mock_redis_client.set.assert_called_once_with("test_key", "test_value", ex=3600)
    
    @pytest.mark.asyncio
    async def test_cache_set_failure(self) -> Any:
        """Test cache set operation failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.set.side_effect = Exception("Redis error")
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.set("test_key", "test_value")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cache_delete_success(self) -> Any:
        """Test successful cache delete operation."""
        mock_redis_client = AsyncMock()
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.delete("test_key")
        
        assert result is True
        mock_redis_client.delete.assert_called_once_with("test_key")
    
    @pytest.mark.asyncio
    async def test_cache_delete_failure(self) -> Any:
        """Test cache delete operation failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.delete.side_effect = Exception("Redis error")
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.delete("test_key")
        
        assert result is False
    
    @pytest.mark.asyncio
    async def test_cache_health_check_success(self) -> Any:
        """Test successful cache health check."""
        mock_redis_client = AsyncMock()
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.health_check()
        
        assert result is True
        mock_redis_client.ping.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_cache_health_check_failure(self) -> Any:
        """Test cache health check failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.ping.side_effect = Exception("Redis error")
        cache_manager = CacheManager(mock_redis_client)
        
        result = await cache_manager.health_check()
        
        assert result is False

# ============================================================================
# Cache Manager Initialization Tests
# ============================================================================

class TestCacheManagerInitialization:
    """Test cache manager initialization."""
    
    @pytest.mark.asyncio
    async def test_initialize_cache_manager_success(self) -> Any:
        """Test successful cache manager initialization."""
        mock_redis_client = AsyncMock()
        
        result = await initialize_cache_manager(mock_redis_client)
        
        assert isinstance(result, CacheManager)
        assert result.redis == mock_redis_client
        
        # Verify test operations were called
        mock_redis_client.set.assert_called_once_with("health_check", "ok", ex=60)
        mock_redis_client.get.assert_called_once_with("health_check")
        mock_redis_client.delete.assert_called_once_with("health_check")
    
    @pytest.mark.asyncio
    async def test_initialize_cache_manager_failure(self) -> Any:
        """Test cache manager initialization failure."""
        mock_redis_client = AsyncMock()
        mock_redis_client.set.side_effect = Exception("Redis error")
        
        with pytest.raises(Exception):
            await initialize_cache_manager(mock_redis_client)
    
    @pytest.mark.asyncio
    async def test_close_cache_manager(self) -> Any:
        """Test cache manager closure."""
        mock_redis_client = AsyncMock()
        cache_manager = CacheManager(mock_redis_client)
        
        await close_cache_manager(cache_manager)
        
        # Should complete without error
        assert True

# ============================================================================
# Background Tasks Tests
# ============================================================================

class TestBackgroundTasks:
    """Test background task management."""
    
    @pytest.mark.asyncio
    async def test_start_background_tasks(self) -> Any:
        """Test starting background tasks."""
        app_state = AppState(shutdown_event=asyncio.Event())
        
        with patch('lifespan_management.health_monitor_task') as mock_health_task:
            with patch('lifespan_management.metrics_collector_task') as mock_metrics_task:
                await start_background_tasks(app_state)
                
                assert len(app_state.background_tasks) == 2
                assert all(isinstance(task, asyncio.Task) for task in app_state.background_tasks)
    
    @pytest.mark.asyncio
    async def test_stop_background_tasks(self) -> Any:
        """Test stopping background tasks."""
        app_state = AppState(shutdown_event=asyncio.Event())
        
        # Create mock tasks
        mock_task1 = AsyncMock()
        mock_task1.done.return_value = False
        mock_task2 = AsyncMock()
        mock_task2.done.return_value = False
        
        app_state.background_tasks = [mock_task1, mock_task2]
        
        await stop_background_tasks(app_state)
        
        # Verify shutdown event was set
        assert app_state.shutdown_event.is_set()
        
        # Verify tasks were cancelled
        mock_task1.cancel.assert_called_once()
        mock_task2.cancel.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_monitor_task(self) -> Any:
        """Test health monitor background task."""
        app_state = AppState(shutdown_event=asyncio.Event())
        
        # Mock dependencies
        app_state.database_pool = Mock()
        app_state.redis_client = AsyncMock()
        app_state.cache_manager = Mock()
        
        with patch('lifespan_management.check_database_health', return_value=True):
            with patch('lifespan_management.check_redis_health', return_value=True):
                with patch('lifespan_management.check_cache_health', return_value=True):
                    # Start task
                    task = asyncio.create_task(health_monitor_task(app_state))
                    
                    # Let it run for a short time
                    await asyncio.sleep(0.1)
                    
                    # Stop task
                    app_state.shutdown_event.set()
                    await task
                    
                    # Verify metrics were updated
                    assert "last_health_check" in app_state.metrics
                    assert app_state.metrics["overall_healthy"] is True
    
    @pytest.mark.asyncio
    async def test_metrics_collector_task(self) -> Any:
        """Test metrics collector background task."""
        app_state = AppState(shutdown_event=asyncio.Event())
        app_state.cache_manager = Mock()
        
        with patch('lifespan_management.collect_system_metrics') as mock_collect:
            mock_collect.return_value = {"cpu_percent": 50.0, "memory_percent": 60.0}
            
            # Start task
            task = asyncio.create_task(metrics_collector_task(app_state))
            
            # Let it run for a short time
            await asyncio.sleep(0.1)
            
            # Stop task
            app_state.shutdown_event.set()
            await task
            
            # Verify metrics were collected
            assert "cpu_percent" in app_state.metrics
            assert "memory_percent" in app_state.metrics

# ============================================================================
# Health Check Tests
# ============================================================================

class TestHealthChecks:
    """Test health check functionality."""
    
    @pytest.mark.asyncio
    async def test_perform_health_check_success(self) -> Any:
        """Test successful health check."""
        app_state = AppState()
        app_state.database_pool = Mock()
        app_state.redis_client = AsyncMock()
        app_state.cache_manager = Mock()
        
        with patch('lifespan_management.check_database_health', return_value=True):
            with patch('lifespan_management.check_redis_health', return_value=True):
                with patch('lifespan_management.check_cache_health', return_value=True):
                    await perform_health_check(app_state)
                    
                    assert app_state.is_healthy is True
    
    @pytest.mark.asyncio
    async def test_perform_health_check_failure(self) -> Any:
        """Test health check failure."""
        app_state = AppState()
        app_state.database_pool = Mock()
        app_state.redis_client = AsyncMock()
        app_state.cache_manager = Mock()
        
        with patch('lifespan_management.check_database_health', return_value=False):
            with patch('lifespan_management.check_redis_health', return_value=True):
                with patch('lifespan_management.check_cache_health', return_value=True):
                    await perform_health_check(app_state)
                    
                    assert app_state.is_healthy is False
    
    @pytest.mark.asyncio
    async def test_collect_system_metrics(self) -> Any:
        """Test system metrics collection."""
        with patch('lifespan_management.psutil') as mock_psutil:
            mock_psutil.cpu_percent.return_value = 50.0
            mock_psutil.virtual_memory.return_value.percent = 60.0
            mock_psutil.disk_usage.return_value.percent = 70.0
            
            metrics = await collect_system_metrics()
            
            assert "timestamp" in metrics
            assert "cpu_percent" in metrics
            assert "memory_percent" in metrics
            assert "disk_percent" in metrics
            assert "uptime" in metrics

# ============================================================================
# Signal Handling Tests
# ============================================================================

class TestSignalHandling:
    """Test signal handling functionality."""
    
    def test_setup_signal_handlers(self) -> Any:
        """Test signal handler setup."""
        shutdown_event = asyncio.Event()
        
        with patch('lifespan_management.signal.signal') as mock_signal:
            setup_signal_handlers(shutdown_event)
            
            # Verify signal handlers were registered
            assert mock_signal.call_count == 2
            calls = mock_signal.call_args_list
            assert any(call[0][0] == signal.SIGTERM for call in calls)
            assert any(call[0][0] == signal.SIGINT for call in calls)

# ============================================================================
# Logging Tests
# ============================================================================

class TestLogging:
    """Test logging setup."""
    
    def test_setup_logging(self) -> Any:
        """Test logging setup."""
        with patch('lifespan_management.structlog.configure') as mock_configure:
            with patch('lifespan_management.logging.basicConfig') as mock_basic_config:
                setup_logging("INFO")
                
                mock_configure.assert_called_once()
                mock_basic_config.assert_called_once()

# ============================================================================
# Application Factory Tests
# ============================================================================

class TestApplicationFactory:
    """Test application factory functionality."""
    
    def test_create_app(self) -> Any:
        """Test application creation."""
        with patch('lifespan_management.load_config') as mock_load_config:
            mock_load_config.return_value = SAMPLE_CONFIG
            
            app = create_app()
            
            assert isinstance(app, FastAPI)
            assert app.title == "Test API"
            assert app.version == "1.0.0"
            assert app.debug is True
    
    def test_setup_middleware(self) -> Any:
        """Test middleware setup."""
        app = FastAPI()
        config = SAMPLE_CONFIG
        
        setup_middleware(app, config)
        
        # Verify middleware was added (check by looking at middleware stack)
        assert len(app.user_middleware) > 0
    
    def test_setup_routes(self) -> Any:
        """Test route setup."""
        app = FastAPI()
        
        setup_routes(app)
        
        # Verify routes were added
        routes = [route.path for route in app.routes]
        assert "/" in routes
        assert "/health" in routes
        assert "/metrics" in routes
        assert "/analyze" in routes
    
    def test_setup_exception_handlers(self) -> Any:
        """Test exception handler setup."""
        app = FastAPI()
        
        setup_exception_handlers(app)
        
        # Verify exception handler was added
        assert app.exception_handlers.get(Exception) is not None

# ============================================================================
# Dependency Injection Tests
# ============================================================================

class TestDependencyInjection:
    """Test dependency injection functionality."""
    
    def test_get_app_state(self) -> Optional[Dict[str, Any]]:
        """Test getting application state."""
        app_state = AppState()
        request = Mock()
        request.app.state = app_state
        
        result = get_app_state(request)
        
        assert result == app_state
    
    def test_get_database_session_success(self) -> Optional[Dict[str, Any]]:
        """Test getting database session successfully."""
        app_state = AppState()
        app_state.database_pool = Mock()
        mock_session = Mock()
        app_state.database_pool.return_value = mock_session
        
        request = Mock()
        request.app.state = app_state
        
        result = get_database_session(request)
        
        assert result == mock_session
        app_state.database_pool.assert_called_once()
    
    def test_get_database_session_unavailable(self) -> Optional[Dict[str, Any]]:
        """Test getting database session when unavailable."""
        app_state = AppState()
        app_state.database_pool = None
        
        request = Mock()
        request.app.state = app_state
        
        with pytest.raises(HTTPException) as exc_info:
            get_database_session(request)
        
        assert exc_info.value.status_code == 503
        assert "Database not available" in str(exc_info.value.detail)
    
    def test_get_cache_manager_success(self) -> Optional[Dict[str, Any]]:
        """Test getting cache manager successfully."""
        app_state = AppState()
        cache_manager = CacheManager(Mock())
        app_state.cache_manager = cache_manager
        
        request = Mock()
        request.app.state = app_state
        
        result = get_cache_manager(request)
        
        assert result == cache_manager
    
    def test_get_cache_manager_unavailable(self) -> Optional[Dict[str, Any]]:
        """Test getting cache manager when unavailable."""
        app_state = AppState()
        app_state.cache_manager = None
        
        request = Mock()
        request.app.state = app_state
        
        with pytest.raises(HTTPException) as exc_info:
            get_cache_manager(request)
        
        assert exc_info.value.status_code == 503
        assert "Cache not available" in str(exc_info.value.detail)

# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for lifespan management."""
    
    @pytest.mark.asyncio
    async def test_lifespan_context_manager(self) -> Any:
        """Test lifespan context manager functionality."""
        app = FastAPI()
        
        # Mock all dependencies
        with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
            with patch('lifespan_management.setup_logging'):
                with patch('lifespan_management.initialize_database_pool') as mock_db:
                    with patch('lifespan_management.initialize_redis_client') as mock_redis:
                        with patch('lifespan_management.initialize_cache_manager') as mock_cache:
                            with patch('lifespan_management.setup_signal_handlers'):
                                with patch('lifespan_management.start_background_tasks'):
                                    with patch('lifespan_management.perform_health_check'):
                                        with patch('lifespan_management.stop_background_tasks'):
                                            with patch('lifespan_management.close_cache_manager'):
                                                with patch('lifespan_management.close_redis_client'):
                                                    with patch('lifespan_management.close_database_pool'):
                                                        async with lifespan(app) as app_state:
                                                            # Verify app state was created
                                                            assert isinstance(app_state, AppState)
                                                            assert app_state.is_healthy is True
                                                            
                                                            # Verify dependencies were initialized
                                                            mock_db.assert_called_once()
                                                            mock_redis.assert_called_once()
                                                            mock_cache.assert_called_once()
    
    async def test_fastapi_app_with_lifespan(self) -> Any:
        """Test FastAPI app with lifespan integration."""
        with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
            app = create_app()
            
            # Verify lifespan was set
            assert app.router.lifespan_context is not None
            
            # Test with TestClient
            client = TestClient(app)
            
            # Test root endpoint
            response = client.get("/")
            assert response.status_code == 200
            assert response.json()["status"] == "running"
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self) -> Any:
        """Test graceful shutdown process."""
        app_state = AppState(shutdown_event=asyncio.Event())
        
        # Mock background tasks
        mock_task1 = AsyncMock()
        mock_task1.done.return_value = False
        mock_task2 = AsyncMock()
        mock_task2.done.return_value = False
        app_state.background_tasks = [mock_task1, mock_task2]
        
        # Mock resources
        app_state.database_pool = Mock()
        app_state.redis_client = AsyncMock()
        app_state.cache_manager = Mock()
        
        with patch('lifespan_management.stop_background_tasks'):
            with patch('lifespan_management.close_cache_manager'):
                with patch('lifespan_management.close_redis_client'):
                    with patch('lifespan_management.close_database_pool'):
                        # Simulate shutdown
                        app_state.shutdown_event.set()
                        
                        # Verify shutdown event was set
                        assert app_state.shutdown_event.is_set()

# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Performance tests for lifespan management."""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_performance(self) -> Any:
        """Test lifespan startup performance."""
        app = FastAPI()
        
        start_time = datetime.now()
        
        with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
            with patch('lifespan_management.setup_logging'):
                with patch('lifespan_management.initialize_database_pool'):
                    with patch('lifespan_management.initialize_redis_client'):
                        with patch('lifespan_management.initialize_cache_manager'):
                            with patch('lifespan_management.setup_signal_handlers'):
                                with patch('lifespan_management.start_background_tasks'):
                                    with patch('lifespan_management.perform_health_check'):
                                        async with lifespan(app):
                                            pass
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Startup should be fast (less than 1 second)
        assert duration < 1.0
    
    @pytest.mark.asyncio
    async def test_background_task_performance(self) -> Any:
        """Test background task performance."""
        app_state = AppState(shutdown_event=asyncio.Event())
        
        # Mock dependencies
        app_state.database_pool = Mock()
        app_state.redis_client = AsyncMock()
        app_state.cache_manager = Mock()
        
        start_time = datetime.now()
        
        with patch('lifespan_management.check_database_health', return_value=True):
            with patch('lifespan_management.check_redis_health', return_value=True):
                with patch('lifespan_management.check_cache_health', return_value=True):
                    # Run health monitor for a short time
                    task = asyncio.create_task(health_monitor_task(app_state))
                    await asyncio.sleep(0.1)
                    app_state.shutdown_event.set()
                    await task
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Health check should be fast
        assert duration < 0.5

# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Test error handling in lifespan management."""
    
    @pytest.mark.asyncio
    async def test_lifespan_startup_failure(self) -> Any:
        """Test lifespan startup failure handling."""
        app = FastAPI()
        
        with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
            with patch('lifespan_management.setup_logging'):
                with patch('lifespan_management.initialize_database_pool', side_effect=Exception("DB Error")):
                    with pytest.raises(Exception, match="DB Error"):
                        async with lifespan(app):
                            pass
    
    @pytest.mark.asyncio
    async def test_lifespan_shutdown_failure(self) -> Any:
        """Test lifespan shutdown failure handling."""
        app = FastAPI()
        
        with patch('lifespan_management.load_config', return_value=SAMPLE_CONFIG):
            with patch('lifespan_management.setup_logging'):
                with patch('lifespan_management.initialize_database_pool'):
                    with patch('lifespan_management.initialize_redis_client'):
                        with patch('lifespan_management.initialize_cache_manager'):
                            with patch('lifespan_management.setup_signal_handlers'):
                                with patch('lifespan_management.start_background_tasks'):
                                    with patch('lifespan_management.perform_health_check'):
                                        with patch('lifespan_management.stop_background_tasks', side_effect=Exception("Shutdown Error")):
                                            with pytest.raises(Exception, match="Shutdown Error"):
                                                async with lifespan(app):
                                                    pass

# ============================================================================
# Main Test Runner
# ============================================================================

if __name__ == "__main__":
    # Run all tests
    pytest.main([__file__, "-v", "--tb=short"]) 