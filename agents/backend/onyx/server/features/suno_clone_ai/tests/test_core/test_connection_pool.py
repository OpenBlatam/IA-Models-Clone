"""
Tests para el gestor de connection pools
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, MagicMock

from core.connection_pool import (
    ConnectionPoolManager,
    get_connection_pool_manager
)


@pytest.fixture
def pool_manager():
    """Fixture para ConnectionPoolManager"""
    return ConnectionPoolManager()


@pytest.fixture
def mock_pool_factory():
    """Mock de factory de pool"""
    async def factory(min_size=2, max_size=10, **kwargs):
        pool = MagicMock()
        pool.min_size = min_size
        pool.max_size = max_size
        pool.size = Mock(return_value=5)
        pool.idle = Mock(return_value=2)
        pool.acquire = AsyncMock()
        pool.close = AsyncMock()
        return pool
    return factory


@pytest.mark.unit
@pytest.mark.core
class TestConnectionPoolManager:
    """Tests para ConnectionPoolManager"""
    
    def test_init(self, pool_manager):
        """Test de inicialización"""
        assert pool_manager._pools == {}
        assert pool_manager._pool_configs == {}
    
    def test_register_pool(self, pool_manager, mock_pool_factory):
        """Test de registro de pool"""
        pool_manager.register_pool(
            "test_pool",
            mock_pool_factory,
            min_size=3,
            max_size=15
        )
        
        assert "test_pool" in pool_manager._pool_configs
        config = pool_manager._pool_configs["test_pool"]
        assert config["min_size"] == 3
        assert config["max_size"] == 15
        assert config["factory"] == mock_pool_factory
    
    def test_register_pool_with_kwargs(self, pool_manager, mock_pool_factory):
        """Test de registro con kwargs adicionales"""
        pool_manager.register_pool(
            "test_pool",
            mock_pool_factory,
            min_size=2,
            max_size=10,
            host="localhost",
            port=5432
        )
        
        config = pool_manager._pool_configs["test_pool"]
        assert config["host"] == "localhost"
        assert config["port"] == 5432
    
    @pytest.mark.asyncio
    async def test_get_pool_lazy_init(self, pool_manager, mock_pool_factory):
        """Test de obtención de pool con inicialización lazy"""
        pool_manager.register_pool("test_pool", mock_pool_factory)
        
        pool = await pool_manager.get_pool("test_pool")
        
        assert pool is not None
        assert "test_pool" in pool_manager._pools
        assert pool.min_size == 2
        assert pool.max_size == 10
    
    @pytest.mark.asyncio
    async def test_get_pool_not_registered(self, pool_manager):
        """Test de obtención de pool no registrado"""
        with pytest.raises(ValueError, match="not registered"):
            await pool_manager.get_pool("nonexistent")
    
    @pytest.mark.asyncio
    async def test_get_pool_cached(self, pool_manager, mock_pool_factory):
        """Test de que el pool se cachea"""
        pool_manager.register_pool("test_pool", mock_pool_factory)
        
        pool1 = await pool_manager.get_pool("test_pool")
        pool2 = await pool_manager.get_pool("test_pool")
        
        assert pool1 is pool2
    
    @pytest.mark.asyncio
    async def test_acquire_with_acquire_method(self, pool_manager, mock_pool_factory):
        """Test de acquire con método acquire"""
        pool_manager.register_pool("test_pool", mock_pool_factory)
        
        connection_mock = MagicMock()
        pool = await pool_manager.get_pool("test_pool")
        pool.acquire.return_value.__aenter__ = AsyncMock(return_value=connection_mock)
        pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
        
        async with pool_manager.acquire("test_pool") as conn:
            assert conn is connection_mock
    
    @pytest.mark.asyncio
    async def test_acquire_without_acquire_method(self, pool_manager):
        """Test de acquire sin método acquire (ej: Redis)"""
        async def simple_factory(min_size=2, max_size=10, **kwargs):
            return MagicMock()  # Pool sin método acquire
        
        pool_manager.register_pool("simple_pool", simple_factory)
        
        async with pool_manager.acquire("simple_pool") as conn:
            assert conn is not None
    
    @pytest.mark.asyncio
    async def test_close_all(self, pool_manager, mock_pool_factory):
        """Test de cierre de todos los pools"""
        pool_manager.register_pool("pool1", mock_pool_factory)
        pool_manager.register_pool("pool2", mock_pool_factory)
        
        await pool_manager.get_pool("pool1")
        await pool_manager.get_pool("pool2")
        
        await pool_manager.close_all()
        
        assert len(pool_manager._pools) == 0
    
    @pytest.mark.asyncio
    async def test_close_all_with_dispose(self, pool_manager):
        """Test de cierre con método dispose"""
        async def factory_with_dispose(min_size=2, max_size=10, **kwargs):
            pool = MagicMock()
            pool.dispose = AsyncMock()
            return pool
        
        pool_manager.register_pool("pool_with_dispose", factory_with_dispose)
        await pool_manager.get_pool("pool_with_dispose")
        
        await pool_manager.close_all()
        
        assert len(pool_manager._pools) == 0
    
    @pytest.mark.asyncio
    async def test_get_pool_stats(self, pool_manager, mock_pool_factory):
        """Test de obtención de estadísticas"""
        pool_manager.register_pool("test_pool", mock_pool_factory)
        await pool_manager.get_pool("test_pool")
        
        stats = pool_manager.get_pool_stats("test_pool")
        
        assert stats["name"] == "test_pool"
        assert stats["initialized"] is True
        assert "size" in stats
        assert "idle" in stats
    
    def test_get_pool_stats_not_initialized(self, pool_manager):
        """Test de estadísticas de pool no inicializado"""
        stats = pool_manager.get_pool_stats("nonexistent")
        
        assert "error" in stats


@pytest.mark.unit
@pytest.mark.core
class TestGetConnectionPoolManager:
    """Tests para get_connection_pool_manager"""
    
    def test_get_connection_pool_manager_singleton(self):
        """Test de que retorna singleton"""
        instance1 = get_connection_pool_manager()
        instance2 = get_connection_pool_manager()
        
        assert instance1 is instance2
        assert isinstance(instance1, ConnectionPoolManager)
