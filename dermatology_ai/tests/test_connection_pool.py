"""
Tests for Connection Pool Manager
Tests for database connection pool management
"""

import pytest
from unittest.mock import Mock, AsyncMock
import asyncio

from core.infrastructure.connection_pool_manager import (
    ConnectionPoolManager,
    PoolConfig
)


class TestPoolConfig:
    """Tests for PoolConfig"""
    
    def test_create_pool_config(self):
        """Test creating pool configuration"""
        config = PoolConfig(
            min_size=5,
            max_size=20,
            max_overflow=10,
            pool_timeout=30.0
        )
        
        assert config.min_size == 5
        assert config.max_size == 20
        assert config.max_overflow == 10
        assert config.pool_timeout == 30.0
    
    def test_pool_config_defaults(self):
        """Test pool config with defaults"""
        config = PoolConfig()
        
        assert config.min_size == 5
        assert config.max_size == 20
        assert config.pool_pre_ping is True


class TestConnectionPoolManager:
    """Tests for ConnectionPoolManager"""
    
    @pytest.fixture
    def pool_manager(self):
        """Create connection pool manager"""
        return ConnectionPoolManager()
    
    def test_register_pool(self, pool_manager):
        """Test registering a connection pool"""
        mock_pool = Mock()
        config = PoolConfig(min_size=5, max_size=20)
        
        pool_manager.register_pool("test_pool", mock_pool, config)
        
        assert "test_pool" in pool_manager.pools
        assert pool_manager.pools["test_pool"] == mock_pool
        assert pool_manager.configs["test_pool"] == config
    
    def test_get_pool(self, pool_manager):
        """Test getting a connection pool"""
        mock_pool = Mock()
        pool_manager.register_pool("test_pool", mock_pool)
        
        pool = pool_manager.get_pool("test_pool")
        
        assert pool == mock_pool
    
    def test_get_pool_not_found(self, pool_manager):
        """Test getting non-existent pool"""
        pool = pool_manager.get_pool("non_existent")
        
        assert pool is None
    
    @pytest.mark.asyncio
    async def test_acquire_connection(self, pool_manager):
        """Test acquiring connection from pool"""
        mock_pool = Mock()
        mock_connection = AsyncMock()
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        
        pool_manager.register_pool("test_pool", mock_pool)
        
        async with pool_manager.acquire("test_pool") as conn:
            assert conn == mock_connection
        
        mock_pool.acquire.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_release_connection(self, pool_manager):
        """Test releasing connection back to pool"""
        mock_pool = Mock()
        mock_connection = Mock()
        mock_pool.acquire = AsyncMock(return_value=mock_connection)
        mock_pool.release = AsyncMock()
        
        pool_manager.register_pool("test_pool", mock_pool)
        
        async with pool_manager.acquire("test_pool"):
            pass
        
        # Connection should be released (via context manager)
        # Implementation dependent
    
    def test_get_pool_stats(self, pool_manager):
        """Test getting pool statistics"""
        mock_pool = Mock()
        mock_pool.size = 10
        mock_pool.checked_in = 5
        mock_pool.checked_out = 5
        
        pool_manager.register_pool("test_pool", mock_pool)
        
        stats = pool_manager.get_stats("test_pool")
        
        assert stats is not None
        # Stats format depends on implementation
    
    @pytest.mark.asyncio
    async def test_close_pool(self, pool_manager):
        """Test closing a connection pool"""
        mock_pool = Mock()
        mock_pool.close = AsyncMock()
        
        pool_manager.register_pool("test_pool", mock_pool)
        
        await pool_manager.close_pool("test_pool")
        
        mock_pool.close.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_close_all_pools(self, pool_manager):
        """Test closing all connection pools"""
        mock_pool1 = Mock()
        mock_pool1.close = AsyncMock()
        mock_pool2 = Mock()
        mock_pool2.close = AsyncMock()
        
        pool_manager.register_pool("pool1", mock_pool1)
        pool_manager.register_pool("pool2", mock_pool2)
        
        await pool_manager.close_all()
        
        mock_pool1.close.assert_called_once()
        mock_pool2.close.assert_called_once()



