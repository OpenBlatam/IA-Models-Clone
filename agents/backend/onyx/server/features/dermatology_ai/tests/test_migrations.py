"""
Tests for Migrations
Tests for database migration management
"""

import pytest
from unittest.mock import Mock, AsyncMock
from datetime import datetime

from core.migrations.migration_manager import Migration, MigrationManager


class TestMigration:
    """Tests for Migration"""
    
    def test_create_migration(self):
        """Test creating a migration"""
        migration = Migration(
            version="001",
            name="create_analyses_table",
            up_sql="CREATE TABLE analyses...",
            down_sql="DROP TABLE analyses..."
        )
        
        assert migration.version == "001"
        assert migration.name == "create_analyses_table"
        assert migration.up_sql is not None
        assert migration.down_sql is not None
    
    @pytest.mark.asyncio
    async def test_migration_up(self):
        """Test running migration up"""
        executed = False
        
        async def up_function():
            nonlocal executed
            executed = True
        
        migration = Migration(
            version="001",
            name="test_migration",
            up=up_function
        )
        
        await migration.up()
        
        assert executed is True
    
    @pytest.mark.asyncio
    async def test_migration_down(self):
        """Test running migration down"""
        executed = False
        
        async def down_function():
            nonlocal executed
            executed = True
        
        migration = Migration(
            version="001",
            name="test_migration",
            down=down_function
        )
        
        await migration.down()
        
        assert executed is True


class TestMigrationManager:
    """Tests for MigrationManager"""
    
    @pytest.fixture
    def migration_manager(self):
        """Create migration manager"""
        return MigrationManager()
    
    @pytest.mark.asyncio
    async def test_register_migration(self, migration_manager):
        """Test registering a migration"""
        migration = Migration(
            version="001",
            name="test_migration",
            up_sql="CREATE TABLE test..."
        )
        
        migration_manager.register(migration)
        
        assert migration in migration_manager.migrations
    
    @pytest.mark.asyncio
    async def test_get_pending_migrations(self, migration_manager):
        """Test getting pending migrations"""
        migration1 = Migration(version="001", name="migration1", up_sql="...")
        migration2 = Migration(version="002", name="migration2", up_sql="...")
        
        migration_manager.register(migration1)
        migration_manager.register(migration2)
        
        # Mock that migration1 is already applied
        migration_manager.applied_migrations = ["001"]
        
        pending = migration_manager.get_pending()
        
        assert len(pending) == 1
        assert pending[0].version == "002"
    
    @pytest.mark.asyncio
    async def test_migrate_up(self, migration_manager):
        """Test running migrations up"""
        executed_migrations = []
        
        async def up_func():
            executed_migrations.append("001")
        
        migration = Migration(
            version="001",
            name="test_migration",
            up=up_func
        )
        
        migration_manager.register(migration)
        
        await migration_manager.migrate_up()
        
        assert "001" in executed_migrations
    
    @pytest.mark.asyncio
    async def test_migrate_down(self, migration_manager):
        """Test running migrations down"""
        executed_migrations = []
        
        async def down_func():
            executed_migrations.append("001")
        
        migration = Migration(
            version="001",
            name="test_migration",
            down=down_func
        )
        
        migration_manager.register(migration)
        migration_manager.applied_migrations = ["001"]
        
        await migration_manager.migrate_down("001")
        
        assert "001" in executed_migrations
    
    @pytest.mark.asyncio
    async def test_migration_ordering(self, migration_manager):
        """Test migrations are ordered correctly"""
        migration1 = Migration(version="001", name="migration1", up_sql="...")
        migration2 = Migration(version="002", name="migration2", up_sql="...")
        migration3 = Migration(version="003", name="migration3", up_sql="...")
        
        # Register in different order
        migration_manager.register(migration3)
        migration_manager.register(migration1)
        migration_manager.register(migration2)
        
        # Should be ordered by version
        ordered = migration_manager.get_ordered()
        
        assert ordered[0].version == "001"
        assert ordered[1].version == "002"
        assert ordered[2].version == "003"



