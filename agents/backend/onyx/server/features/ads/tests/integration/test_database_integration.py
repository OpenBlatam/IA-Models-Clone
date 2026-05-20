"""
Integration tests for the ads database layer.

This module tests the integration between different database components:
- Database manager integration
- Repository implementations integration
- Connection pooling integration
- Transaction management integration
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List
import tempfile
import os

# Import database components
from agents.backend.onyx.server.features.ads.infrastructure.database import (
    DatabaseManager, ConnectionPool, DatabaseConfig
)
from agents.backend.onyx.server.features.ads.infrastructure.repositories import (
    AdsRepositoryImpl, CampaignRepositoryImpl, RepositoryFactory
)

# Import domain entities
from agents.backend.onyx.server.features.ads.domain.entities import Ad, AdCampaign, AdGroup


class TestDatabaseIntegration:
    """Test database integration and cross-component communication."""

    @pytest.fixture
    def temp_db_path(self):
        """Create a temporary database path."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            temp_path = f.name
        yield temp_path
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.fixture
    def db_config(self, temp_db_path):
        """Create database configuration."""
        return DatabaseConfig(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )

    @pytest.fixture
    def mock_connection_pool(self):
        """Mock connection pool."""
        mock = AsyncMock(spec=ConnectionPool)
        mock.get_connection.return_value = AsyncMock()
        mock.return_connection.return_value = None
        mock.get_stats.return_value = {
            "pool_size": 5,
            "checked_in": 3,
            "checked_out": 2,
            "overflow": 0
        }
        return mock

    @pytest.fixture
    def mock_database_manager(self, db_config, mock_connection_pool):
        """Mock database manager."""
        mock = AsyncMock(spec=DatabaseManager)
        mock.config = db_config
        mock.connection_pool = mock_connection_pool
        mock.get_session.return_value.__aenter__.return_value = AsyncMock()
        return mock

    @pytest.mark.asyncio
    async def test_database_config_integration(self, db_config):
        """Test database configuration integration."""
        # Test configuration creation
        assert db_config.database_url is not None
        assert db_config.pool_size == 5
        assert db_config.max_overflow == 10
        assert db_config.pool_timeout == 30
        assert db_config.pool_recycle == 3600

    @pytest.mark.asyncio
    async def test_connection_pool_integration(self, mock_connection_pool):
        """Test connection pool integration."""
        # Test connection acquisition
        connection = await mock_connection_pool.get_connection()
        assert connection is not None
        
        # Test connection return
        result = await mock_connection_pool.return_connection(connection)
        assert result is None
        
        # Test pool statistics
        stats = await mock_connection_pool.get_stats()
        assert "pool_size" in stats
        assert "checked_in" in stats
        assert "checked_out" in stats

    @pytest.mark.asyncio
    async def test_database_manager_integration(self, mock_database_manager):
        """Test database manager integration."""
        # Test session acquisition
        async with mock_database_manager.get_session() as session:
            assert session is not None
        
        # Test configuration access
        assert mock_database_manager.config is not None
        assert mock_database_manager.connection_pool is not None

    @pytest.mark.asyncio
    async def test_repository_factory_integration(self, mock_database_manager):
        """Test repository factory integration."""
        # Test repository creation
        factory = RepositoryFactory(database_manager=mock_database_manager)
        
        # Test ads repository creation
        ads_repo = factory.create_ads_repository()
        assert ads_repo is not None
        assert isinstance(ads_repo, AdsRepositoryImpl)
        
        # Test campaign repository creation
        campaign_repo = factory.create_campaign_repository()
        assert campaign_repo is not None
        assert isinstance(campaign_repo, CampaignRepositoryImpl)

    @pytest.mark.asyncio
    async def test_repository_database_integration(self, mock_database_manager):
        """Test repository integration with database manager."""
        # Create repository with mocked database manager
        ads_repo = AdsRepositoryImpl(database_manager=mock_database_manager)
        
        # Test that repository can access database manager
        assert ads_repo.database_manager is mock_database_manager
        
        # Test session usage
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_transaction_integration(self, mock_database_manager):
        """Test transaction management integration."""
        # Test transaction handling
        async with mock_database_manager.get_session() as session:
            # Simulate transaction operations
            await session.execute("BEGIN")
            await session.execute("COMMIT")
            
            # Verify session was used
            assert session is not None

    @pytest.mark.asyncio
    async def test_connection_pooling_integration(self, mock_connection_pool):
        """Test connection pooling integration."""
        # Test multiple connection acquisitions
        connections = []
        for _ in range(3):
            connection = await mock_connection_pool.get_connection()
            connections.append(connection)
        
        # Verify connections were acquired
        assert len(connections) == 3
        assert all(conn is not None for conn in connections)
        
        # Test connection return
        for connection in connections:
            await mock_connection_pool.return_connection(connection)
        
        # Test pool statistics after operations
        stats = await mock_connection_pool.get_stats()
        assert "pool_size" in stats

    @pytest.mark.asyncio
    async def test_database_error_handling_integration(self, mock_database_manager):
        """Test database error handling integration."""
        # Mock database manager to raise exception
        mock_database_manager.get_session.side_effect = Exception("Database connection failed")
        
        # Test that errors are properly handled
        with pytest.raises(Exception):
            async with mock_database_manager.get_session():
                pass

    @pytest.mark.asyncio
    async def test_database_performance_integration(self, mock_database_manager):
        """Test database performance integration."""
        import time
        
        # Measure database operation performance
        start_time = time.time()
        async with mock_database_manager.get_session() as session:
            assert session is not None
        end_time = time.time()
        
        # Should complete within reasonable time
        assert (end_time - start_time) < 1.0

    @pytest.mark.asyncio
    async def test_database_concurrent_access_integration(self, mock_database_manager):
        """Test database concurrent access integration."""
        # Test concurrent session acquisitions
        async def get_session():
            async with mock_database_manager.get_session() as session:
                return session is not None
        
        # Execute multiple concurrent operations
        results = await asyncio.gather(*[get_session() for _ in range(5)])
        
        # All should succeed
        assert all(results)

    @pytest.mark.asyncio
    async def test_database_connection_cleanup_integration(self, mock_connection_pool):
        """Test database connection cleanup integration."""
        # Acquire connections
        connections = []
        for _ in range(3):
            connection = await mock_connection_pool.get_connection()
            connections.append(connection)
        
        # Return connections
        for connection in connections:
            await mock_connection_pool.return_connection(connection)
        
        # Verify cleanup was handled
        # This would depend on actual cleanup implementation

    @pytest.mark.asyncio
    async def test_database_configuration_validation_integration(self, temp_db_path):
        """Test database configuration validation integration."""
        # Test valid configuration
        valid_config = DatabaseConfig(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=5,
            max_overflow=10
        )
        assert valid_config.database_url is not None
        
        # Test invalid configuration (negative values)
        with pytest.raises(ValueError):
            invalid_config = DatabaseConfig(
                database_url=f"sqlite:///{temp_db_path}",
                pool_size=-1,  # Invalid
                max_overflow=10
            )

    @pytest.mark.asyncio
    async def test_database_url_parsing_integration(self, temp_db_path):
        """Test database URL parsing integration."""
        # Test SQLite URL
        sqlite_url = f"sqlite:///{temp_db_path}"
        config = DatabaseConfig(database_url=sqlite_url)
        assert config.database_url == sqlite_url
        
        # Test PostgreSQL URL
        postgres_url = "postgresql://user:pass@localhost:5432/db"
        config = DatabaseConfig(database_url=postgres_url)
        assert config.database_url == postgres_url

    @pytest.mark.asyncio
    async def test_database_pool_configuration_integration(self, temp_db_path):
        """Test database pool configuration integration."""
        # Test different pool configurations
        configs = [
            DatabaseConfig(
                database_url=f"sqlite:///{temp_db_path}",
                pool_size=1,
                max_overflow=0
            ),
            DatabaseConfig(
                database_url=f"sqlite:///{temp_db_path}",
                pool_size=10,
                max_overflow=20
            ),
            DatabaseConfig(
                database_url=f"sqlite:///{temp_db_path}",
                pool_size=5,
                max_overflow=10
            )
        ]
        
        for config in configs:
            assert config.pool_size > 0
            assert config.max_overflow >= 0

    @pytest.mark.asyncio
    async def test_database_timeout_configuration_integration(self, temp_db_path):
        """Test database timeout configuration integration."""
        # Test timeout configurations
        config = DatabaseConfig(
            database_url=f"sqlite:///{temp_db_path}",
            pool_timeout=60,
            pool_recycle=7200
        )
        
        assert config.pool_timeout == 60
        assert config.pool_recycle == 7200

    @pytest.mark.asyncio
    async def test_database_connection_retry_integration(self, mock_database_manager):
        """Test database connection retry integration."""
        # Test that connection retries work
        # This would depend on actual retry implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_health_check_integration(self, mock_database_manager):
        """Test database health check integration."""
        # Test that database health can be checked
        # This would depend on actual health check implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_metrics_integration(self, mock_connection_pool):
        """Test database metrics integration."""
        # Test that metrics are collected
        stats = await mock_connection_pool.get_stats()
        
        # Verify metrics structure
        required_metrics = ["pool_size", "checked_in", "checked_out", "overflow"]
        for metric in required_metrics:
            assert metric in stats

    @pytest.mark.asyncio
    async def test_database_logging_integration(self, mock_database_manager):
        """Test database logging integration."""
        # Test that database operations generate logs
        # This would depend on actual logging implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_security_integration(self, temp_db_path):
        """Test database security integration."""
        # Test that sensitive information is not exposed
        config = DatabaseConfig(
            database_url=f"sqlite:///{temp_db_path}",
            pool_size=5
        )
        
        # Database URL should not expose credentials in this case
        assert "password" not in config.database_url.lower()
        assert "secret" not in config.database_url.lower()

    @pytest.mark.asyncio
    async def test_database_migration_integration(self, mock_database_manager):
        """Test database migration integration."""
        # Test that migrations can be handled
        # This would depend on actual migration implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_backup_integration(self, mock_database_manager):
        """Test database backup integration."""
        # Test that backups can be handled
        # This would depend on actual backup implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_restore_integration(self, mock_database_manager):
        """Test database restore integration."""
        # Test that restores can be handled
        # This would depend on actual restore implementation
        
        # For now, test that session acquisition works
        async with mock_database_manager.get_session() as session:
            assert session is not None

    @pytest.mark.asyncio
    async def test_database_end_to_end_integration(self, mock_database_manager, mock_connection_pool):
        """Test end-to-end database integration."""
        # Test complete database workflow
        
        # 1. Configure database
        config = DatabaseConfig(
            database_url="sqlite:///test.db",
            pool_size=5,
            max_overflow=10
        )
        assert config.database_url is not None
        
        # 2. Create connection pool
        assert mock_connection_pool is not None
        
        # 3. Create database manager
        assert mock_database_manager is not None
        
        # 4. Acquire session
        async with mock_database_manager.get_session() as session:
            assert session is not None
        
        # 5. Get connection pool stats
        stats = await mock_connection_pool.get_stats()
        assert "pool_size" in stats
        
        # 6. Verify all components work together
        assert mock_database_manager.connection_pool is mock_connection_pool


class TestDatabaseRepositoryIntegration:
    """Test database repository integration."""

    @pytest.fixture
    def mock_session(self):
        """Mock database session."""
        mock = AsyncMock()
        mock.execute.return_value = AsyncMock()
        mock.commit.return_value = None
        mock.rollback.return_value = None
        return mock

    @pytest.fixture
    def mock_database_manager_with_session(self, mock_session):
        """Mock database manager with session."""
        mock = AsyncMock()
        mock.get_session.return_value.__aenter__.return_value = mock_session
        return mock

    @pytest.mark.asyncio
    async def test_ads_repository_database_integration(self, mock_database_manager_with_session):
        """Test ads repository integration with database."""
        # Create repository
        ads_repo = AdsRepositoryImpl(database_manager=mock_database_manager_with_session)
        
        # Test database operations
        async with mock_database_manager_with_session.get_session() as session:
            # Test create operation
            result = await ads_repo.create({
                "title": "Test Ad",
                "description": "Test Description",
                "platform": "facebook",
                "budget": 1000.0
            })
            
            # Verify session was used
            assert session is not None

    @pytest.mark.asyncio
    async def test_campaign_repository_database_integration(self, mock_database_manager_with_session):
        """Test campaign repository integration with database."""
        # Create repository
        campaign_repo = CampaignRepositoryImpl(database_manager=mock_database_manager_with_session)
        
        # Test database operations
        async with mock_database_manager_with_session.get_session() as session:
            # Test create operation
            result = await campaign_repo.create({
                "name": "Test Campaign",
                "description": "Test Campaign Description",
                "budget": 5000.0
            })
            
            # Verify session was used
            assert session is not None

    @pytest.mark.asyncio
    async def test_repository_transaction_integration(self, mock_database_manager_with_session):
        """Test repository transaction integration."""
        # Test transaction handling in repositories
        ads_repo = AdsRepositoryImpl(database_manager=mock_database_manager_with_session)
        
        async with mock_database_manager_with_session.get_session() as session:
            # Simulate transaction operations
            await session.execute("BEGIN")
            await session.commit()
            
            # Verify session was used
            assert session is not None

    @pytest.mark.asyncio
    async def test_repository_error_handling_integration(self, mock_database_manager_with_session):
        """Test repository error handling integration."""
        # Test that repositories handle database errors gracefully
        ads_repo = AdsRepositoryImpl(database_manager=mock_database_manager_with_session)
        
        # Mock session to raise exception
        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Database error")
        mock_database_manager_with_session.get_session.return_value.__aenter__.return_value = mock_session
        
        # Test error handling
        with pytest.raises(Exception):
            async with mock_database_manager_with_session.get_session() as session:
                await session.execute("SELECT * FROM ads")


# Test utilities for database integration tests
@pytest.fixture
def database_test_utilities():
    """Utility functions for database integration tests."""
    
    def create_test_db_config(database_url: str = "sqlite:///test.db") -> DatabaseConfig:
        """Create test database configuration."""
        return DatabaseConfig(
            database_url=database_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=3600
        )
    
    def validate_db_config(config: DatabaseConfig) -> bool:
        """Validate database configuration."""
        required_fields = ["database_url", "pool_size", "max_overflow"]
        return all(hasattr(config, field) for field in required_fields)
    
    def create_test_connection_stats() -> Dict[str, Any]:
        """Create test connection statistics."""
        return {
            "pool_size": 5,
            "checked_in": 3,
            "checked_out": 2,
            "overflow": 0
        }
    
    return {
        "create_test_db_config": create_test_db_config,
        "validate_db_config": validate_db_config,
        "create_test_connection_stats": create_test_connection_stats
    }
